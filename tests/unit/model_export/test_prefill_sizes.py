# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import sys

import pytest


def test_probe_reads_pass_dump_header():
    from torq.model_export import prefill_sizes as probe

    stderr = "// -----// IR Dump After TileAndFuse: torq-tile-and-fuse{slice-count=2} //----- //\n" + '''
func.func @main() {
  %w = arith.constant dense<1.0> : tensor<8x16xbf16>
  return
}
'''
    assert probe._parse_tile_and_fuse(stderr, 4, 8, 16) == probe.ProbeResult(4, None, False, False)


def test_probe_sweeps_narrow_nonmonotone_plateaus(monkeypatch, tmp_path):
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path)
    calls = []
    monkeypatch.setattr(probe, "run_probe", lambda *args: "")

    def result(_, m, k, n):
        calls.append(m)
        return probe.ProbeResult(m, 4 if m == 7 else 8, False, False)

    monkeypatch.setattr(probe, "_parse_tile_and_fuse", result)
    kwargs = dict(cap=12, tool=sys.executable)
    report = probe.recommend(path, "SL2610", [], tmp_path, **kwargs)
    assert sorted(calls) == [1, *range(4, 13)]
    assert [(p["b_start"], p["b_end"]) for p in report.shapes[0].plateaus] == [(4, 6), (7, 7), (8, 12)]
    assert report.chosen == [4, 7, 12]
    calls.clear()
    cached = probe.recommend(path, "SL2610", [], tmp_path, **kwargs)
    assert cached.chosen == report.chosen
    assert calls == []


def test_probe_drops_restream_and_no_fit(monkeypatch, tmp_path):
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path)
    monkeypatch.setattr(probe, "run_probe", lambda *args: "")
    monkeypatch.setattr(probe, "_parse_tile_and_fuse", lambda _, m, k, n: probe.ProbeResult(
        m, None if m == 8 else 8 if m < 6 else 4, m == 6, m == 6))
    report = probe.recommend(path, "SL2610", [], tmp_path, cap=8, tool=sys.executable)
    assert report.chosen == [4, 5, 7]
    dropped = {c["b"]: c["reason"] for c in report.candidates if not c["kept"]}
    assert "re-streamed" in dropped[6] and "no tile fit" in dropped[8] and len(dropped) == 2
    with pytest.raises(probe.ProbeError, match="max batch size"):
        probe.recommend(path, "SL2610", [], tmp_path, cap=3, tool=sys.executable)


def test_recommendation_uses_exporter_option():
    from torq.model_export.prefill_sizes import ProbeReport

    assert ProbeReport([4, 7, 12], [], 12, 16).sizes_flag() == "--native-batch-prefill --batch-prefill-sizes 4,7,12"


def _model(tmp_path, dtype=None, *, quantized=False):
    import onnx
    from onnx import helper as h, TensorProto as T

    dtype = dtype or T.FLOAT
    nodes = []
    if quantized:
        weights = [h.make_tensor('w', dtype, [8, 16], (list(range(-8, 8)) if dtype == T.INT4 else list(range(16))) * 8),
                   h.make_tensor('scale', T.BFLOAT16, [2, 16], [0.25, 0.5] * 16),
                   h.make_tensor('zero', dtype, [2, 16], [0, 1] * 16)]
        nodes.append(h.make_node('DequantizeLinear', ['w', 'scale', 'zero'], ['weight'], axis=0, block_size=4))
        activation = T.BFLOAT16
    else:
        weights = [h.make_tensor('weight', dtype, [8, 16], [0.125, 0.5] * 64)]
        activation = dtype
    nodes.append(h.make_node('MatMul', ['x', 'weight'], ['y']))
    model = h.make_model(h.make_graph(nodes, 'main',
        [h.make_tensor_value_info('x', activation, [1, 1, 8])],
        [h.make_tensor_value_info('y', activation, [1, 1, 16])], weights),
        opset_imports=[h.make_opsetid('', 21)])
    path = tmp_path / 'decoder.onnx'
    onnx.save(model, path)
    return path


@pytest.mark.parametrize('dtype', [1, 10, 16, 21, 22])
def test_projection_preserves_real_types_constants_and_quantization(tmp_path, dtype):
    from iree.compiler import ir
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path, dtype, quantized=dtype in (21, 22))
    projection, = probe.extract_projections(path)
    assert (projection.k, projection.n) == (8, 16)
    output = tmp_path / 'probe.mlirbc'
    projection.write(output, 7)
    with ir.Context():
        module = ir.Module.parse(output.read_bytes())
        text = str(module)
        elt = {1: 'f32', 10: 'f16', 16: 'bf16', 22: 'bf16', 21: 'bf16'}[dtype]
        assert f'!torch.vtensor<[1,7,8],{elt}>' in text
        assert f'!torch.vtensor<[1,7,16],{elt}>' in text
        assert 'onnx.MatMul' in text
        if dtype in (21, 22):
            assert 'onnx.DequantizeLinear' in text
            assert 'tensor<8x16x' in text and 'i4>' in text
            assert 'vtensor<[2,16],' in text
            assert 'tensor<2x16xbf16>' in text
            assert 'torch.onnx.block_size = 4' in text
        # The extracted imported graph is also a supported source.
    # Reset to decode before writing a second imported source.
    projection.write(output, 1)
    assert len(probe.extract_projections(output)) == 1
    with ir.Context():
        source = tmp_path / 'source.mlir'
        source.write_text(str(ir.Module.parse(output.read_bytes())))
    assert len(probe.extract_projections(source)) == 1


def test_same_shape_different_quantization_is_not_deduplicated(tmp_path):
    import onnx
    from onnx import helper as h, TensorProto as T
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path, T.INT4, quantized=True)
    model = onnx.load(path)
    model.graph.initializer.append(h.make_tensor('float_weight', T.BFLOAT16, [8, 16], [0.25] * 128))
    model.graph.node.append(h.make_node('MatMul', ['x', 'float_weight'], ['float_y']))
    model.graph.output.append(h.make_tensor_value_info('float_y', T.BFLOAT16, [1, 1, 16]))
    onnx.save(model, path)
    projections = probe.extract_projections(path)
    assert [(p.k, p.n) for p in projections] == [(8, 16), (8, 16)]
    assert projections[0].key != projections[1].key


def test_probe_cache_tracks_actual_projection_contents(tmp_path, monkeypatch):
    import onnx
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path)
    calls = []
    monkeypatch.setattr(probe, 'run_probe', lambda *args: '')
    monkeypatch.setattr(probe, '_parse_tile_and_fuse', lambda _, m, k, n:
        calls.append(m) or probe.ProbeResult(m, 8, False, False))
    kwargs = dict(cap=4, tool=sys.executable)
    probe.recommend(path, 'SL2610', [], tmp_path, **kwargs)
    assert sorted(calls) == [1, 4]
    calls.clear()
    probe.recommend(path, 'SL2610', [], tmp_path, **kwargs)
    assert calls == []
    model = onnx.load(path)
    model.graph.initializer[0].float_data[0] = 2.0
    onnx.save(model, path)
    probe.recommend(path, 'SL2610', [], tmp_path, **kwargs)
    assert sorted(calls) == [1, 4]


@pytest.mark.parametrize('dtype,shape,snap_grid', [
    ('i4', '2x4x16', False), ('f16', '8x16', False),
    ('f32', '8x16', False), ('bf16', '1x8x16', True),
])
@pytest.mark.parametrize('m_outer', [False, True])
def test_parser_reads_typed_and_blocked_weight_strips(dtype, shape, snap_grid, m_outer):
    from torq.model_export import prefill_sizes as probe

    n_loop = '%n = scf.for %j = %c0 to %c16 step %c8 iter_args(%out = %empty) -> (tensor<1x12x16xbf16>) {'
    m_loop = '%m = scf.for %i = %c0 to %c12 step %c4 iter_args(%out = %empty) -> (tensor<1x12x16xbf16>) {'
    offsets = '0, 0, %j' if shape.count('x') == 2 else '0, %j'
    loops = [m_loop, n_loop] if m_outer else [n_loop, m_loop]
    dump = '// -----// IR Dump After TileAndFuse: torq-tile-and-fuse //----- //\n' + '\n'.join([
        'func.func @main() {',
        f'  %w = arith.constant dense_resource<__elided__> : tensor<{shape}x{dtype}>',
        '  %c0 = arith.constant 0 : index', '  %c4 = arith.constant 4 : index',
        '  %c8 = arith.constant 8 : index', '  %c12 = arith.constant 12 : index',
        '  %c16 = arith.constant 16 : index',
        '  ' + loops[0], '    ' + loops[1],
        f'      %weight = tensor.extract_slice %w[{offsets}]',
        '      %activation = tensor.extract_slice %x[0, %i, 0]',
        '    }', '  }', '}',
    ])
    assert probe._parse_tile_and_fuse(dump, 12, 8, 16) == probe.ProbeResult(12, 8, True, m_outer, snap_grid)


def test_quantized_widths_do_not_use_bf16_snap(tmp_path, monkeypatch):
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path, 22, quantized=True)
    monkeypatch.setattr(probe, 'run_probe', lambda *args: '')
    monkeypatch.setattr(probe, '_parse_tile_and_fuse', lambda _, m, k, n:
        probe.ProbeResult(m, 96 if m == 1 else 64 if m < 6 else 48, False, False, False))
    monkeypatch.setattr(probe, 'snap', lambda *args: pytest.fail('BF16 snap used for INT4'))
    report = probe.recommend(path, 'SL2610', ['--torq-enable-split-constants-optimization'],
                             tmp_path, cap=8, tool=sys.executable)
    assert report.chosen == [4, 5, 8]


@pytest.mark.parametrize('read_only', [False, True])
def test_context_length_uses_cache_layout(read_only):
    from onnx import helper as h, TensorProto as T
    from torq.model_export import prefill_sizes as probe

    shape = [1, 256, 2, 64] if read_only else [1, 2, 256, 64]
    output_shape = [1, 1, 2, 64] if read_only else shape
    model = h.make_model(h.make_graph([], 'decoder', [
        h.make_tensor_value_info('past_key_values.0.key_value', T.BFLOAT16, shape),
        h.make_tensor_value_info('attention_mask', T.BFLOAT16, [1, 1, 1, 256]),
    ], [h.make_tensor_value_info('present.0.key_value', T.BFLOAT16, output_shape)]))
    assert probe._context_length_from_onnx(model) == 256


def test_external_int4_projection_survives_source_removal(tmp_path):
    import onnx
    from onnx import numpy_helper
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path, 22, quantized=True)
    model = onnx.load(path)
    for tensor in model.graph.initializer:
        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor), tensor.name))
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location='weights.data', size_threshold=0)
    projection, = probe.extract_projections(path)
    # Once extracted, the probe is independent of the source sidecar.
    (tmp_path / 'weights.data').unlink()
    projection.write(tmp_path / 'probe.mlirbc', 4)
    assert (tmp_path / 'probe.mlirbc').stat().st_size > 0


def test_structural_dedup_preserves_splat_and_quantization_changes(tmp_path):
    import onnx
    from onnx import helper as h, TensorProto as T
    from torq.model_export import prefill_sizes as probe

    path = _model(tmp_path)
    model = onnx.load(path)
    for name, values in [('same', [0.25, 0.75] * 64), ('splat', [0.25] * 128)]:
        model.graph.initializer.append(h.make_tensor(name, T.FLOAT, [8, 16], values))
        model.graph.node.append(h.make_node('MatMul', ['x', name], [name + '_out']))
        model.graph.output.append(h.make_tensor_value_info(name + '_out', T.FLOAT, [1, 1, 16]))
    onnx.save(model, path)
    assert len(probe.extract_projections(path)) == 2



def test_source_probe_excludes_the_independent_native_head(monkeypatch, tmp_path):
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper
    from torq.model_export import prefill_sizes as probe

    model = h.make_model(h.make_graph([
        h.make_node("MatMul", ["x", "projection"], ["hidden"]),
        h.make_node("MatMul", ["hidden", "head"], ["logits"]),
    ], "source", [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 8])],
        [h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 32])], [
        numpy_helper.from_array(np.ones((8, 16), np.float32), "projection"),
        numpy_helper.from_array(np.ones((16, 32), np.float32), "head"),
    ]))
    path = tmp_path / "source.onnx"
    onnx.save(model, path)
    assert probe.extract_weight_shapes(path) == [(8, 16), (16, 32)]
    monkeypatch.setattr(probe, "run_probe", lambda *args: "")
    monkeypatch.setattr(probe, "_parse_tile_and_fuse",
                        lambda _, m, k, n: probe.ProbeResult(m, 8, False, False))
    report = probe.recommend(path, "SL2610", [], tmp_path, cap=4, context_length=8,
                             tool=sys.executable, exclude_lm_head=True)
    assert [(shape.k, shape.n) for shape in report.shapes] == [(8, 16)]
    assert report.chosen == [4]
