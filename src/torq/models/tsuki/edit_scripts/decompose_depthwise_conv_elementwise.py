#!/usr/bin/env python3
"""Decompose depthwise Conv1D into Slice + Mul + Add (NSS-friendly).

linalg.depthwise_conv_2d_nchw_chw is not supported on NSS at any kernel size.
This script replaces depthwise Conv1D with purely elementwise ops, tiled along
channels to fit in LRAM (512KB):

For each channel chunk:
  Slice(channels) → Pad → (Slice + Mul) × K → tree-Add → bias-Add → barrier
Then Concat all chunks.

All resulting ops (Pad, Slice, Mul, Add) are natively supported on NSS.
"""
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _add_fusion_barrier(name, inp_name, shape, elem_type, new_nodes, new_inits, new_vis):
    """Pad(+1 dim0) then Slice(1:2 dim0) — identity that blocks TileAndFuse."""
    padded = f'{name}_bar_pad'
    pad_vals_name = f'{name}_bar_pv'
    new_inits.append(numpy_helper.from_array(
        np.array([1, 0, 0, 0, 0, 0], dtype=np.int64), name=pad_vals_name))
    new_nodes.append(helper.make_node('Pad', [inp_name, pad_vals_name], [padded],
                                      name=f'{name}_bar_pad', mode='constant'))
    padded_shape = list(shape)
    padded_shape[0] += 1
    new_vis.append(helper.make_tensor_value_info(padded, elem_type, padded_shape))

    out = f'{name}_bar_out'
    st_name = f'{name}_bar_st'
    en_name = f'{name}_bar_en'
    ax_name = f'{name}_bar_ax'
    new_inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=st_name))
    new_inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name=en_name))
    new_inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name=ax_name))
    new_nodes.append(helper.make_node('Slice', [padded, st_name, en_name, ax_name],
                                      [out], name=f'{name}_bar_sl'))
    new_vis.append(helper.make_tensor_value_info(out, elem_type, shape))
    return out


def _decompose_one_chunk(prefix, inp_name, C_chunk, L, L_padded, pad_left,
                         w_raw_chunk, b_raw_chunk, K, dilation, elem_type, is_bf16,
                         out_name, new_nodes, new_inits, new_vis):
    """Decompose one channel chunk: Pad → (Slice+Mul)×K → tree-Add → bias-Add."""
    # Pad input along dim 2
    padded_name = f'{prefix}_padded'
    pad_vals = np.array([0, 0, pad_left, 0, 0, L_padded - L - pad_left], dtype=np.int64)
    pad_init_name = f'{prefix}_pv'
    new_inits.append(numpy_helper.from_array(pad_vals, name=pad_init_name))
    new_nodes.append(helper.make_node('Pad', [inp_name, pad_init_name], [padded_name],
                                      name=f'{prefix}_pad', mode='constant'))
    new_vis.append(helper.make_tensor_value_info(padded_name, elem_type, [1, C_chunk, L_padded]))

    # For each kernel position: Slice + Mul
    sub_products = []
    for k in range(K):
        slice_start = k * dilation
        slice_end = slice_start + L

        slice_out = f'{prefix}_sl_{k}'
        st = f'{prefix}_st_{k}'
        en = f'{prefix}_en_{k}'
        ax = f'{prefix}_ax_{k}'
        new_inits.append(numpy_helper.from_array(np.array([slice_start], dtype=np.int64), name=st))
        new_inits.append(numpy_helper.from_array(np.array([slice_end], dtype=np.int64), name=en))
        new_inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name=ax))
        new_nodes.append(helper.make_node('Slice', [padded_name, st, en, ax],
                                          [slice_out], name=f'{prefix}_sl_{k}'))
        new_vis.append(helper.make_tensor_value_info(slice_out, elem_type, [1, C_chunk, L]))

        wk_name = f'{prefix}_w_{k}'
        wk_data = w_raw_chunk[:, :, k:k+1]
        if is_bf16:
            wk_init = onnx.TensorProto()
            wk_init.name = wk_name
            wk_init.data_type = TensorProto.BFLOAT16
            wk_init.dims[:] = [1, C_chunk, 1]
            wk_init.raw_data = wk_data.copy().tobytes()
        else:
            wk_init = numpy_helper.from_array(wk_data.reshape(1, C_chunk, 1), name=wk_name)
        new_inits.append(wk_init)

        mul_out = f'{prefix}_mul_{k}'
        new_nodes.append(helper.make_node('Mul', [slice_out, wk_name], [mul_out],
                                          name=f'{prefix}_mul_{k}'))
        new_vis.append(helper.make_tensor_value_info(mul_out, elem_type, [1, C_chunk, L]))
        sub_products.append(mul_out)

    # Tree reduction
    level = 0
    while len(sub_products) > 1:
        next_level = []
        for i in range(0, len(sub_products), 2):
            if i + 1 < len(sub_products):
                sum_name = f'{prefix}_add_L{level}_{i // 2}'
                new_nodes.append(helper.make_node('Add', [sub_products[i], sub_products[i + 1]],
                                                  [sum_name], name=sum_name))
                new_vis.append(helper.make_tensor_value_info(sum_name, elem_type, [1, C_chunk, L]))
                next_level.append(sum_name)
            else:
                next_level.append(sub_products[i])
        sub_products = next_level
        level += 1

    # Add bias
    if b_raw_chunk is not None:
        b_name = f'{prefix}_bias'
        if is_bf16:
            b_init = onnx.TensorProto()
            b_init.name = b_name
            b_init.data_type = TensorProto.BFLOAT16
            b_init.dims[:] = [1, C_chunk, 1]
            b_init.raw_data = b_raw_chunk.copy().tobytes()
        else:
            b_init = numpy_helper.from_array(b_raw_chunk.reshape(1, C_chunk, 1), name=b_name)
        new_inits.append(b_init)
        new_nodes.append(helper.make_node('Add', [sub_products[0], b_name], [out_name],
                                          name=f'{prefix}_add_bias'))
    else:
        for n in new_nodes:
            if n.output[0] == sub_products[0]:
                n.output[0] = out_name
                break


def decompose_depthwise_conv_elementwise(model, min_kernel=1, channel_chunk=32):
    graph = model.graph
    init_map = {i.name: i for i in graph.initializer}
    vi_map = {vi.name: vi for vi in graph.value_info}
    for inp in graph.input:
        vi_map[inp.name] = inp
    for out in graph.output:
        vi_map[out.name] = out

    new_nodes = []
    new_inits = []
    new_vis = []
    replaced = 0

    for node in graph.node:
        if node.op_type != 'Conv':
            new_nodes.append(node)
            continue

        group = 1
        kernel_shape = []
        pads = []
        dilations = [1]
        strides = [1]
        for a in node.attribute:
            if a.name == 'group':
                group = a.i
            elif a.name == 'kernel_shape':
                kernel_shape = list(a.ints)
            elif a.name == 'pads':
                pads = list(a.ints)
            elif a.name == 'dilations':
                dilations = list(a.ints)
            elif a.name == 'strides':
                strides = list(a.ints)

        if len(kernel_shape) != 1 or kernel_shape[0] < min_kernel:
            new_nodes.append(node)
            continue

        inp_name = node.input[0]
        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 else None
        out_name = node.output[0]

        if w_name not in init_map:
            new_nodes.append(node)
            continue

        w_init = init_map[w_name]
        C = w_init.dims[0]
        K = w_init.dims[2]
        dilation = dilations[0]
        stride = strides[0]

        if group != C or stride != 1:
            new_nodes.append(node)
            continue

        inp_vi = vi_map.get(inp_name)
        if inp_vi is None:
            new_nodes.append(node)
            continue
        inp_shape = [d.dim_value for d in inp_vi.type.tensor_type.shape.dim]
        L = inp_shape[2]

        is_bf16 = (inp_vi.type.tensor_type.elem_type == TensorProto.BFLOAT16)
        elem_type = TensorProto.BFLOAT16 if is_bf16 else TensorProto.FLOAT

        if len(pads) == 2:
            pad_left, pad_right = pads[0], pads[1]
        elif len(pads) == 4:
            pad_left, pad_right = pads[1], pads[3]
        else:
            effective_k = (K - 1) * dilation + 1
            pad_left = (effective_k - 1) // 2
            pad_right = effective_k - 1 - pad_left

        L_padded = L + pad_left + pad_right
        prefix = f'{node.name}__dw_elem'

        if is_bf16:
            w_raw = np.frombuffer(w_init.raw_data, dtype=np.uint16).reshape(C, 1, K)
        else:
            w_raw = numpy_helper.to_array(w_init).reshape(C, 1, K)

        b_raw = None
        if b_name and b_name in init_map:
            b_init_orig = init_map[b_name]
            if is_bf16:
                b_raw = np.frombuffer(b_init_orig.raw_data, dtype=np.uint16).reshape(C)
            else:
                b_raw = numpy_helper.to_array(b_init_orig).reshape(C)

        # Barrier on input to isolate from upstream fusion
        barriered_inp = _add_fusion_barrier(f'{prefix}_inp', inp_name,
                                            [1, C, L], elem_type,
                                            new_nodes, new_inits, new_vis)

        n_chunks = (C + channel_chunk - 1) // channel_chunk
        chunk_outputs = []

        for ci in range(n_chunks):
            c_start = ci * channel_chunk
            c_end = min(c_start + channel_chunk, C)
            c_size = c_end - c_start
            chunk_prefix = f'{prefix}_ch{ci}'

            # Slice input along channels (dim 1)
            chunk_inp = f'{chunk_prefix}_inp'
            ch_st = f'{chunk_prefix}_ch_st'
            ch_en = f'{chunk_prefix}_ch_en'
            ch_ax = f'{chunk_prefix}_ch_ax'
            new_inits.append(numpy_helper.from_array(np.array([c_start], dtype=np.int64), name=ch_st))
            new_inits.append(numpy_helper.from_array(np.array([c_end], dtype=np.int64), name=ch_en))
            new_inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=ch_ax))
            new_nodes.append(helper.make_node('Slice', [barriered_inp, ch_st, ch_en, ch_ax],
                                              [chunk_inp], name=f'{chunk_prefix}_ch_sl'))
            new_vis.append(helper.make_tensor_value_info(chunk_inp, elem_type, [1, c_size, L]))

            # Decompose this chunk
            chunk_out = f'{chunk_prefix}_out'
            w_chunk = w_raw[c_start:c_end, :, :]
            b_chunk = b_raw[c_start:c_end] if b_raw is not None else None

            _decompose_one_chunk(chunk_prefix, chunk_inp, c_size, L, L_padded, pad_left,
                                 w_chunk, b_chunk, K, dilation, elem_type, is_bf16,
                                 chunk_out, new_nodes, new_inits, new_vis)

            # Fusion barrier on chunk output
            barrier_out = _add_fusion_barrier(chunk_prefix, chunk_out,
                                              [1, c_size, L], elem_type,
                                              new_nodes, new_inits, new_vis)
            chunk_outputs.append(barrier_out)

        # Concat chunks along dim 1, then barrier on final output
        if len(chunk_outputs) == 1:
            concat_out = chunk_outputs[0]
        else:
            concat_out = f'{prefix}_concat_out'
            new_nodes.append(helper.make_node('Concat', chunk_outputs, [concat_out],
                                              name=f'{prefix}_concat', axis=1))
            new_vis.append(helper.make_tensor_value_info(concat_out, elem_type, [1, C, L]))

        # Final fusion barrier to isolate DW conv from downstream ops
        final_bar = _add_fusion_barrier(f'{prefix}_final', concat_out,
                                        [1, C, L], elem_type,
                                        new_nodes, new_inits, new_vis)
        for n in new_nodes:
            if n.output[0] == final_bar:
                n.output[0] = out_name
                break

        replaced += 1
        print(f'Decomposed {node.name}: group={group}, kernel={K}, dilation={dilation}, '
              f'{n_chunks} channel chunks of {channel_chunk}')

    if replaced == 0:
        print(f'No depthwise Conv1D with kernel >= {min_kernel} found')
        return model

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    graph.value_info.extend(new_vis)
    print(f'\nDecomposed {replaced} depthwise Conv1D ops into tiled elementwise ops')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--min-kernel', type=int, default=1,
                        help='Minimum kernel size to decompose (default: 1, i.e. all DW convs)')
    parser.add_argument('--channel-chunk', type=int, default=32,
                        help='Max channels per tile (default: 32)')
    args = parser.parse_args()

    model = onnx.load(args.input)
    model = decompose_depthwise_conv_elementwise(model, args.min_kernel, args.channel_chunk)
    onnx.save(model, args.output)
    onnx.checker.check_model(args.output)
    print(f'Saved to {args.output} (ONNX checker passed)')


if __name__ == '__main__':
    main()
