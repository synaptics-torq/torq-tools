# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Opt-in supplied decoder families; model construction stays in the exporters."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import onnx
from onnx import numpy_helper
from onnx.external_data_helper import _get_initializer_tensors, uses_external_data

from ..utils.compile import _resolve_compiler, compile_mlir_for_vm
from ..utils.onnx import (
    _EXTERNAL_DATA_THRESHOLD, _read_external_data, _set_external_data, load_onnx_lazy,
)


def parse_native_sizes(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[1-9][0-9]*(,[1-9][0-9]*)*", value):
        raise argparse.ArgumentTypeError("native sizes must be comma-separated ascending decimal integers")
    sizes = tuple(map(int, value.split(",")))
    if sizes[0] <= 1 or any(a >= b for a, b in zip(sizes, sizes[1:])):
        raise argparse.ArgumentTypeError("native sizes must be unique, ascending, and greater than one")
    return sizes


def validate_native_options(sizes, *, batch_prefill, context, minimum, static,
                            embeddings, split_head, read_only, individual, unsupported,
                            supports_read_only=False):
    if sizes is None:
        return ()
    if (not sizes or any(type(b) is not int or not minimum <= b <= context for b in sizes)
            or any(a >= b for a, b in zip(sizes, sizes[1:]))):
        raise ValueError(f"--native-batch-prefill requires unique ascending sizes in [{minimum}, {context}]")
    if batch_prefill is not None:
        raise ValueError("--native-batch-prefill and --batch-prefill are mutually exclusive")
    if not static or not embeddings or not split_head or (read_only and not supports_read_only) or individual or unsupported:
        raise ValueError(
            "--native-batch-prefill supports Gemma3/FunctionGemma 270M and LFM2.5 230M only; "
            "requires static exports, --extract-embeddings, --split-lm-head and combined KV I/O. "
            "Gemma3 supports FP32 or INT4 weights and read-only caches; LFM2.5 requires FP32 "
            "and --no-read-only-kv-cache. Explicit dtype conversion, convolution overrides "
            "and topology splitting are unsupported"
        )
    return tuple(sizes)


def check_native_compiler(compiler_path=None):
    """Check the selected executable and the installed importer before destructive work."""
    try:
        from iree.compiler import ir
        from iree.compiler.extras import onnx_importer
        if not all(hasattr(onnx_importer, attr) for attr in ("NodeImporter", "ContextCache", "ModuleCache")):
            raise ImportError("multi-graph importer API is missing")
        with ir.Context():
            pass
        result = subprocess.run([_resolve_compiler(compiler_path), "--help"],
                                capture_output=True, text=True, check=True, timeout=30)
    except (ImportError, OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"native prefill needs compatible compiler Python bindings and executable: {exc}") from exc
    if "--torq-enable-multi-graph" not in result.stdout:
        raise RuntimeError(
            "selected compiler does not support supplied native prefill families; "
            "use --compiler-path or TORQ_COMPILER_PATH with a native-family build"
        )


def _signature(values):
    signature = []
    for value in values:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape") or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in tensor.shape.dim):
            raise ValueError(f"native family requires static shapes: {value.name}")
        signature.append({"name": value.name, "dtype": tensor.elem_type,
                          "shape": [d.dim_value for d in tensor.shape.dim]})
    return signature


def _constant_roles(model):
    """Classify the supported exporter ops by use, never by a byte threshold.

    Projection/kernel RHS and normalization gains are resource parameters.
    Fixed RoPE frequencies are fingerprinted like parameters but stay literal:
    their position-only computation can execute outside the family dispatch.
    Scalar arithmetic, Pow exponents and Where fills are controls. Reject
    unrecognized and mixed sites instead of guessing.
    """
    constants = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            attrs = {a.name: a for a in node.attribute}
            if "value" not in attrs:
                raise ValueError(f"unsupported native Constant encoding: {node.name}")
            constants[node.output[0]] = attrs["value"].t
    producers = {name: node for node in model.graph.node for name in node.output}

    def position_value(name):
        while name in producers:
            node = producers[name]
            if node.op_type == "Add" and len(node.input) == 2:
                variable = [value for value in node.input if value not in constants]
                if len(variable) != 1:
                    return False
                name = variable[0]
            elif node.op_type in ("Cast", "Reshape", "Squeeze", "Unsqueeze", "Identity", "Transpose"):
                name = node.input[0]
            else:
                return False
        return name == "position_ids"

    def normalized_value(name):
        node = producers.get(name)
        if node is None:
            return False
        if node.op_type == "Mul":
            return any(normalized_value(value) for value in node.input)
        if node.op_type == "Div" and len(node.input) == 2:
            denominator = producers.get(node.input[1])
            return denominator is not None and denominator.op_type == "Sqrt"
        return False

    uses = {name: set() for name in constants}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name not in constants:
                continue
            tensor = constants[name]
            if node.op_type == "DequantizeLinear":
                uses[name].add("parameter")
                continue
            if tensor.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.BFLOAT16):
                if tensor.data_type in (onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE, onnx.TensorProto.INT4, onnx.TensorProto.UINT4):
                    raise ValueError(f"unsupported native constant use: {name} at {node.op_type}")
                continue
            scalar = not tensor.dims or all(d == 1 for d in tensor.dims)
            if node.op_type in ("Mul", "MatMul") and position_value(node.input[1 - index]):
                role = "frequency"
            elif ((node.op_type in ("MatMul", "Gemm", "Conv") and index == 1)
                  or (node.op_type == "Mul" and normalized_value(node.input[1 - index]))):
                role = "parameter"
            elif ((node.op_type == "Pow" and index == 1)
                  or (node.op_type == "Where" and index in (1, 2))
                  or (node.op_type == "Add" and name in ("kv_new_mask", "kv_new_mask_bf16"))
                  or (node.op_type == "Add" and position_value(node.input[1 - index]))
                  or (scalar and node.op_type in ("Add", "Sub", "Mul", "Div"))):
                role = "control"
            else:
                raise ValueError(f"unclassified native float constant {name!r} at {node.op_type} input {index}")
            uses[name].add(role)
    parameters = {}
    for name, roles in uses.items():
        tensor = constants[name]
        if tensor.data_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
            roles.add("control")
        if len(roles) > 1:
            raise ValueError(f"native constant has mixed parameter/control uses: {name}")
        if not roles:
            continue
        data = numpy_helper.to_array(tensor)
        if roles in ({"parameter"}, {"frequency"}):
            parameters[name] = (tensor.data_type, tuple(tensor.dims), hashlib.sha256(data.tobytes()).hexdigest())
        if roles != {"parameter"}:
            # Typed fields become dense literals, allowing equal shape/mask
            # controls to CSE too. Do this after the final export transforms.
            tensor.CopyFrom(onnx.helper.make_tensor(tensor.name, tensor.data_type, tensor.dims, data.reshape(-1).tolist()))
    return parameters


def share_native_weights(paths):
    """Pack final decoder initializers into one content-deduplicated data file.

    Keep integer controls inline for ONNX data propagation. Preserve packed
    INT4/BF16 bytes without conversion. Only remove exporter-owned sidecars
    after every rewritten graph is installed; a failed write leaves the
    original graphs and their data readable.
    """
    paths = [Path(path).absolute() for path in paths]
    if not paths:
        return
    directory = paths[0].parent
    if any(path.parent != directory for path in paths):
        raise ValueError("native family ONNX files must share a directory")
    data_path = directory / "transformer.weights.data"
    spans = {}
    with tempfile.TemporaryDirectory(prefix=".native-weights-", dir=directory) as tmp:
        temporary = Path(tmp)
        temporary_data = temporary / data_path.name
        with temporary_data.open("wb") as stream:
            for index, path in enumerate(paths):
                model = load_onnx_lazy(path)
                for tensor in _get_initializer_tensors(model):
                    if tensor.data_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                        continue
                    if uses_external_data(tensor):
                        data = _read_external_data(tensor)
                    elif tensor.HasField("raw_data") and len(tensor.raw_data) >= _EXTERNAL_DATA_THRESHOLD:
                        data = tensor.raw_data
                    else:
                        continue
                    key = (tensor.data_type, tuple(tensor.dims), hashlib.sha256(data).digest())
                    if key not in spans:
                        spans[key] = (stream.tell(), len(data))
                        stream.write(data)
                    offset, length = spans[key]
                    tensor.ClearField("raw_data")
                    _set_external_data(tensor, location=data_path.name, offset=str(offset), length=str(length))
                onnx.save(model, temporary / f"{index}.onnx")
        # Publish only complete data, without replacing any existing weights.
        os.link(temporary_data, data_path)
        for index, path in enumerate(paths):
            os.replace(temporary / f"{index}.onnx", path)
        for path in paths:
            path.with_name(path.name + ".data").unlink(missing_ok=True)


def _tensor_context_cache():
    """Importer cache supporting ONNX packed integers and 16-bit float literals."""
    from iree.compiler import ir
    from iree.compiler.extras import onnx_importer

    class TensorContextCache(onnx_importer.ContextCache):
        def tensor_proto_to_attr(self, tensor):
            if tensor.data_type in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4):
                # MLIR's canonical i4 resource storage uses one byte per element.
                values = numpy_helper.to_array(tensor).astype("uint8") & 15
                return ir.DenseResourceElementsAttr.get_from_buffer(
                    values, self._sanitize_name(tensor.name), self.tensor_proto_to_builtin_type(tensor), alignment=8)
            if tensor.data_type in (onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT16) and not tensor.HasField("raw_data"):
                return ir.DenseElementsAttr.get(numpy_helper.to_array(tensor).view("uint16"),
                                                type=self.tensor_proto_to_builtin_type(tensor))
            return super().tensor_proto_to_attr(tensor)

    return TensorContextCache


def import_native_family(paths: dict[int, Path], output: Path):
    """Import decode (key 1) and static chunks into one verified bytecode module."""
    from iree.compiler import ir
    from iree.compiler.extras import onnx_importer
    from iree.compiler.tools.import_onnx.__main__ import load_onnx_model, parse_arguments

    if not paths or next(iter(paths)) != 1 or len(paths) < 2:
        raise ValueError("native family needs decode followed by prefill entries")
    entries = {}
    parameters = None
    decode_inputs = decode_outputs = None
    read_only = False
    shared_parameters = {}

    class FamilyContextCache(_tensor_context_cache()):
        def tensor_proto_to_attr(self, tensor):
            if tensor.name not in parameters:
                return super().tensor_proto_to_attr(tensor)
            if tensor.name not in shared_parameters:
                shared_parameters[tensor.name] = super().tensor_proto_to_attr(tensor)
            return shared_parameters[tensor.name]

    with ir.Context() as context, ir.Location.unknown():
        module = ir.Module.create()
        context_cache = FamilyContextCache(context)
        module_cache = onnx_importer.ModuleCache(module.operation, context_cache)
        for size, path in paths.items():
            args = parse_arguments([str(path), "--data-prop", "--large-model"])
            model = load_onnx_model(args)
            inputs, outputs = _signature(model.graph.input), _signature(model.graph.output)
            if (len(inputs) < 3 or inputs[0]["name"] != "token_embedding"
                    or inputs[0]["shape"][:2] != [1, size]
                    or inputs[0]["dtype"] not in (onnx.TensorProto.FLOAT, onnx.TensorProto.BFLOAT16)
                    or inputs[1] != {"name": "position_ids", "dtype": (onnx.TensorProto.INT64 if inputs[0]["dtype"] == onnx.TensorProto.FLOAT else onnx.TensorProto.INT32), "shape": [1, 1]}
                    or not outputs or outputs[0]["dtype"] != inputs[0]["dtype"]
                    or outputs[0]["shape"] != [1, 1, inputs[0]["shape"][-1]]
                    or len(outputs) != len(inputs) - 1):
                raise ValueError(f"unsupported native decoder ABI in {path}")
            if size == 1:
                decode_inputs, decode_outputs = inputs, outputs
                read_only = (inputs[2]["name"].startswith("past_key_values.")
                             and inputs[2]["shape"] != outputs[1]["shape"])
                for index, (inp, out) in enumerate(zip(inputs[2:], outputs[1:])):
                    expected = inp["name"].replace("past_key_values.", "present.").replace("past_conv.", "present_conv.")
                    shape = inp["shape"]
                    if read_only and inp["name"].startswith("past_key_values."):
                        if len(shape) != 4 or shape[1] <= 1:
                            raise ValueError(f"invalid time-major native cache: {inp['name']}")
                        shape = [1, 1, *shape[2:]]
                    if (inp["name"] not in (f"past_key_values.{index}.key_value", f"past_conv.{index}")
                            or out["name"] != expected or inp["dtype"] != inputs[0]["dtype"]
                            or (shape, inp["dtype"]) != (out["shape"], out["dtype"])):
                        raise ValueError(f"native state input/output name, order or shape mismatch: {inp['name']}")
            else:
                expected_outputs = [
                    {**out, "shape": [1, size, *out["shape"][2:]]}
                    if read_only and out["name"].startswith("present.") else out
                    for out in decode_outputs
                ]
                if inputs[1:] != decode_inputs[1:] or outputs != expected_outputs or inputs[0]["shape"][2:] != decode_inputs[0]["shape"][2:]:
                    raise ValueError(f"native family state ABI differs in {path}")
            if any(size > inp["shape"][1 if read_only else 2] for inp in inputs[2:]
                   if inp["name"].startswith("past_key_values.")):
                raise ValueError(f"native chunk exceeds cache capacity in {path}")
            current = _constant_roles(model)
            if parameters is None:
                parameters = current
            elif current != parameters:
                changed = sorted(k for k in parameters.keys() | current.keys() if parameters.get(k) != current.get(k))
                raise ValueError(f"native family parameters differ after export: {changed[:12]}")
            name = "main" if size == 1 else f"main_{size}"
            model.graph.name = name
            info = onnx_importer.ModelInfo(model)
            importer = onnx_importer.NodeImporter.define_function(
                info.main_graph, module.operation, context_cache=context_cache, module_cache=module_cache,
            )
            importer.import_all()
            entries[name] = {"size": size, "inputs": inputs, "outputs": outputs}
            del importer, info, model
        module.operation.verify()
        with output.open("wb") as stream:
            module.operation.write_bytecode(stream)
    return {"version": 1, "entries": entries, "kv_cache_sequence_axis": 1 if read_only else 2}


def native_compile_flags(flags):
    if any(flag.startswith("--torq-enable-multi-graph") for flag in flags):
        raise ValueError("native exporter sets multi-graph compilation; do not pass compiler family options")
    required = ["--iree-input-type=onnx-torq", "--iree-hal-target-backends=torq", "--torq-hw=SL2610",
                "--torq-convert-dtypes", "--torq-convert-io-dtype", "--torq-disable-slicing",
                "--torq-enable-split-constants-optimization", "--torq-enable-annotate-tied-operands"]
    result = list(flags)
    for flag in required:
        key = flag.split("=", 1)[0]
        allowed = {flag} if "=" in flag else {flag, flag + "=true", flag + "=1"}
        for arg in flags:
            if (arg == key or arg.startswith(key + "=")) and arg not in allowed:
                raise ValueError(f"native prefill requires {flag}; incompatible option {arg}")
        # Canonicalize true spellings and duplicates so metadata and the compiler
        # agree even when a model default and a user override name the same flag.
        result = [arg for arg in result if arg not in allowed]
        result.append(flag)
    return result + ["--torq-enable-multi-graph"]


def compile_native_family(paths, output_dir, *, compiler_args, compiler_path=None, local_compile=False):
    output_dir = Path(output_dir)
    mlir = output_dir / "transformer.mlirbc"
    metadata = import_native_family(paths, mlir)
    flags = list(compiler_args)
    compile_mlir_for_vm(mlir, output_dir / "transformer.vmfb", compiler_args=flags,
                        compiler_path=compiler_path, use_binary=True, local_compile=local_compile)
    metadata["compile_flags"] = flags
    metadata["runtime_io_converted"] = "--torq-convert-io-dtype" in flags
    (output_dir / "transformer.family.json").write_text(json.dumps(metadata, indent=2) + "\n")


def validate_native_onnx(paths, *, rtol=1e-4, atol=1e-4):
    """Compare every hidden/state output to sequential decode at cache boundaries."""
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    decode = ort.InferenceSession(str(paths[1]), opts, providers=["CPUExecutionProvider"])
    specs = decode.get_inputs()
    state_specs = specs[2:]
    read_only = state_specs[0].shape != decode.get_outputs()[1].shape
    axis = 1 if read_only else 2
    context = next(s.shape[axis] for s in state_specs if s.name.startswith("past_key_values."))
    rng = np.random.default_rng(0)
    for size, path in paths.items():
        if size == 1:
            continue
        chunk = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        for position in sorted({0, min(3, context - size), context - size}):
            embeddings = rng.normal(0, 0.1, chunk.get_inputs()[0].shape).astype(np.float32)
            state = {s.name: (np.zeros(s.shape, dtype=np.float32) if position == 0 else
                             rng.normal(0, 0.1, s.shape).astype(np.float32)) for s in state_specs}
            actual = chunk.run(None, {"token_embedding": embeddings,
                                     "position_ids": np.array([[position]], dtype=np.int64), **state})
            current = {name: value.copy() for name, value in state.items()}
            for offset in range(size):
                expected = decode.run(None, {"token_embedding": embeddings[:, offset:offset + 1],
                                            "position_ids": np.array([[position + offset]], dtype=np.int64), **current})
                for spec, value in zip(state_specs, expected[1:]):
                    if read_only and spec.name.startswith("past_key_values."):
                        current[spec.name][:, position + offset:position + offset + 1] = value
                    else:
                        current[spec.name] = value
            expected = [expected[0], *[
                current[s.name][:, position:position + size] if read_only and s.name.startswith("past_key_values.")
                else current[s.name] for s in state_specs
            ]]
            for spec, result, reference in zip(chunk.get_outputs(), actual, expected):
                message = f"B={size}, p={position}, output={spec.name}"
                tolerance = atol
                if read_only:
                    # Split past/new attention changes FP32 reduction order.
                    # Check both total relative error and peak-scaled element
                    # error, including cancellation near zero.
                    tolerance = max(atol, rtol * np.max(np.abs(reference)))
                    np.testing.assert_allclose(np.linalg.norm(result - reference), 0, rtol=0,
                                               atol=atol + rtol * np.linalg.norm(reference), err_msg=message)
                np.testing.assert_allclose(result, reference, rtol=rtol, atol=tolerance, err_msg=message)
            for spec, result in zip(state_specs, actual[1:]):
                if spec.name.startswith("past_key_values.") and not read_only:
                    np.testing.assert_array_equal(result[:, :, :position], state[spec.name][:, :, :position])
                    np.testing.assert_array_equal(result[:, :, position + size:], state[spec.name][:, :, position + size:])
