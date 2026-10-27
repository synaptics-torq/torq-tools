# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import re

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper as h, numpy_helper

from torq.model_export.prefill import (
    _constant_roles, import_native_family, parse_native_sizes, validate_native_options,
    validate_native_onnx,
)
from torq.models.gemma3 import add_gemma3_export_args
from torq.models.liquid import add_liquid_export_args


@pytest.mark.parametrize("value", ["", "1", "4,4", "7,4", "4,", "4, 7", "04,7", "-4", "4.0", "４"])
def test_invalid_sizes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_native_sizes(value)


@pytest.mark.parametrize("add_args", [add_gemma3_export_args, add_liquid_export_args])
def test_cli_sizes_replace_default_without_accepting_explicit_legacy(add_args):
    parser = argparse.ArgumentParser()
    add_args(parser)
    defaults = parser.parse_args([])
    assert defaults.batch_prefill == 64
    assert defaults.native_batch_prefill is False
    assert defaults.batch_prefill_sizes is None
    automatic = parser.parse_args(["--native-batch-prefill"])
    assert automatic.native_batch_prefill is True
    assert automatic.batch_prefill_sizes is None
    explicit = parser.parse_args(["--native-batch-prefill", "--batch-prefill-sizes", "4,7"])
    assert explicit.native_batch_prefill is True
    assert explicit.batch_prefill_sizes == (4, 7)
    for size in ("0", "64"):
        for arguments in (["--batch-prefill", size, "--native-batch-prefill"],
                          ["--native-batch-prefill", "--batch-prefill", size]):
            with pytest.raises(SystemExit):
                parser.parse_args(arguments)


@pytest.mark.parametrize("override", [dict(sizes=()), dict(sizes=(4,4)), dict(sizes=(7,4)), dict(sizes=(1,4)),
    dict(sizes=(4,17)), dict(sizes=(2,), minimum=4), dict(read_only=True), dict(static=False),
    dict(embeddings=False), dict(split_head=False), dict(individual=True), dict(unsupported=True), dict(batch_prefill=64)])
def test_reject_unsupported_options(override):
    options = dict(sizes=(4,7), enabled=True, batch_prefill=None, context=16, minimum=2, static=True,
                   embeddings=True, split_head=True, read_only=False, individual=False, unsupported=False)
    options.update(override)
    with pytest.raises(ValueError):
        validate_native_options(**options)


def test_literal_roles_are_based_on_use_and_reject_mixed():
    weight = numpy_helper.from_array(np.ones((4,4), np.float32), "weight")
    exponent = numpy_helper.from_array(np.array(2., np.float32), "exponent")
    model = h.make_model(h.make_graph([
        h.make_node("MatMul", ["x","weight"], ["y"]),
        h.make_node("Pow", ["y","exponent"], ["z"]),
    ], "main", [], [], [weight,exponent]))
    assert set(_constant_roles(model)) == {"weight"}
    assert model.graph.initializer[0].raw_data
    assert not model.graph.initializer[1].raw_data
    assert list(model.graph.initializer[1].float_data) == [2.]
    model.graph.node.append(h.make_node("Pow", ["z", "weight"], ["a"]))
    with pytest.raises(ValueError, match="mixed"):
        _constant_roles(model)


@pytest.mark.parametrize("quantized", [False, True])
def test_read_only_causal_mask_is_a_typed_control(quantized):
    import ml_dtypes

    name = "kv_new_mask_bf16" if quantized else "kv_new_mask"
    values = np.ones((1,1,8,4), dtype=ml_dtypes.bfloat16 if quantized else np.float32)
    model = h.make_model(h.make_graph([
        h.make_node("Add", ["scores", name], ["masked"]),
    ], "main", [], [], [numpy_helper.from_array(values, name)]))
    assert _constant_roles(model) == {}
    tensor = model.graph.initializer[0]
    assert not tensor.raw_data
    np.testing.assert_array_equal(numpy_helper.to_array(tensor), values)


def _family_graph(size, weight=1., width=4):
    # Full-cache identity state and a real final-token hidden output.
    inputs = [h.make_tensor_value_info("token_embedding",T.FLOAT,[1,size,width]),
              h.make_tensor_value_info("position_ids",T.INT64,[1,1]),
              h.make_tensor_value_info("past_key_values.0.key_value",T.FLOAT,[1,2,16,4])]
    outputs = [h.make_tensor_value_info("last_hidden_states",T.FLOAT,[1,1,width]),
               h.make_tensor_value_info("present.0.key_value",T.FLOAT,[1,2,16,4])]
    initializers = [numpy_helper.from_array(np.eye(width,dtype=np.float32)*weight,"weight")]
    for name, vals in [("start",[-1]),("end",[2**63-1]),("axis",[1])]:
        initializers.append(numpy_helper.from_array(np.array(vals,np.int64),name))
    return h.make_model(h.make_graph([
        h.make_node("MatMul",["token_embedding","weight"],["projected"]),
        h.make_node("Slice",["projected","start","end","axis"],["last_hidden_states"]),
        h.make_node("Identity",["past_key_values.0.key_value"],["present.0.key_value"]),
    ],"main",inputs,outputs,initializers),opset_imports=[h.make_opsetid("",18)],ir_version=10)


def test_family_import_and_numerical_gate(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    paths = {b: tmp_path/f"decoder_{b}.onnx" for b in (1,4,7)}
    for b,path in paths.items():
        onnx.save(_family_graph(b),path)
    output = tmp_path/"family.mlirbc"
    metadata = import_native_family(paths,output)
    assert output.read_bytes().startswith(b"ML\xefR")
    assert set(metadata["entries"]) == {"main","main_4","main_7"}
    from iree.compiler import ir

    with ir.Context():
        assembly = str(ir.Module.parse(output.read_bytes()))
    assert set(re.findall(r"dense_resource<(weight[^>]*)>", assembly)) == {"weight"}
    validate_native_onnx(paths)
    onnx.save(_family_graph(7,weight=2.),paths[7])
    with pytest.raises(ValueError,match="parameters differ"):
        import_native_family(paths,output)


def _read_only_family_graph(size):
    model = _family_graph(size)
    model.graph.input[2].CopyFrom(h.make_tensor_value_info("past_key_values.0.key_value", T.FLOAT, [1,16,2,4]))
    model.graph.output[1].CopyFrom(h.make_tensor_value_info("present.0.key_value", T.FLOAT, [1,size,2,4]))
    del model.graph.node[-1]
    model.graph.initializer.append(numpy_helper.from_array(np.array([2], np.int64), "rows_axis"))
    model.graph.node.extend([
        h.make_node("Unsqueeze", ["token_embedding", "rows_axis"], ["rows"]),
        h.make_node("Concat", ["rows", "rows"], ["present.0.key_value"], axis=2),
    ])
    return model


def test_read_only_family_import_and_sequential_rows(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    paths = {b: tmp_path / f"decoder_{b}.onnx" for b in (1,4,7)}
    for size, path in paths.items():
        onnx.save(_read_only_family_graph(size), path)
    metadata = import_native_family(paths, tmp_path / "family.mlirbc")
    assert metadata["kv_cache_sequence_axis"] == 1
    assert metadata["entries"]["main_7"]["outputs"][1]["shape"] == [1,7,2,4]
    validate_native_onnx(paths)
    # A chunk returning the wrong rows must fail the populated-cache gate.
    model = _read_only_family_graph(7)
    model.graph.node.insert(len(model.graph.node) - 1, h.make_node("Neg", ["rows"], ["negated"]))
    model.graph.node[-1].input[1] = "negated"
    onnx.save(model, paths[7])
    with pytest.raises(AssertionError):
        validate_native_onnx(paths)


def test_int4_family_fingerprints_all_dql_parameters(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    from torq.lab.quantization.onnx.weights import QuantizationConfig, WeightQuantizer

    paths = {b: tmp_path / f"decoder_{b}.onnx" for b in (1,4)}
    for size, path in paths.items():
        model = _read_only_family_graph(size)
        model.graph.node[0].output[0] = "raw"
        model.graph.initializer.extend([
            numpy_helper.from_array(np.array(1., np.float32), "scale"),
            numpy_helper.from_array(np.zeros((1,size,4), np.float32), "kv_new_mask"),
        ])
        model.graph.node.insert(1, h.make_node("Mul", ["raw", "scale"], ["scaled"]))
        model.graph.node.insert(2, h.make_node("Add", ["scaled", "kv_new_mask"], ["projected"]))
        quantizer = WeightQuantizer()
        quantizer._quantize_to_dql(model, quantizer._find_matmul_weights(model), QuantizationConfig.uniform(4, 2))
        # Conversion drops these unused inputs in the synthetic row-only graph.
        model.graph.input.extend([
            h.make_tensor_value_info("position_ids", T.INT32, [1,1]),
            h.make_tensor_value_info("past_key_values.0.key_value", T.BFLOAT16, [1,16,2,4]),
        ])
        onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=f"{path.name}.data", size_threshold=0)
    from torq.model_export.prefill import share_native_weights
    share_native_weights(paths.values())
    references = []
    for path in paths.values():
        stored = onnx.load(path, load_external_data=False)
        references.append({tensor.name: tuple((entry.key, entry.value) for entry in tensor.external_data)
                           for tensor in stored.graph.initializer if tensor.name.startswith("weight_")})
    assert references[0] == references[1]
    assert all(references[0].values())
    metadata = import_native_family(paths, tmp_path / "family.mlirbc")
    assert metadata["entries"]["main"]["inputs"][0]["dtype"] == T.BFLOAT16
    assert metadata["entries"]["main_4"]["inputs"][1]["dtype"] == T.INT32
    from iree.compiler import ir

    with ir.Context():
        assembly = str(ir.Module.parse((tmp_path / "family.mlirbc").read_bytes()))
    resource = re.search(r'weight_quantized: "0x08000000([0-9a-fA-F]+)"', assembly)
    assert resource and len(bytes.fromhex(resource[1])) == 16  # One byte per i4 element.
    for suffix in ("quantized", "scales", "zero_points"):
        model = onnx.load(paths[4])
        tensor = next(i for i in model.graph.initializer if i.name == "weight_" + suffix)
        original = tensor.raw_data
        tensor.raw_data = bytes([original[0] ^ 1]) + original[1:]
        onnx.save(model, paths[4])
        with pytest.raises(ValueError, match="parameters differ"):
            import_native_family(paths, tmp_path / "family.mlirbc")
        # Restore for the next parameter mutation.
        tensor.raw_data = original
        onnx.save(model, paths[4])


@pytest.mark.parametrize("sizes", [None, (4, 7)])
def test_gemma_native_cli_keeps_default_int4_gptq_and_read_only_cache(tmp_path, monkeypatch, sizes):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from torq.models.gemma3 import export as gemma

    monkeypatch.setattr(gemma.AutoConfig, "from_pretrained", Mock(return_value=SimpleNamespace(
        hidden_size=8, vocab_size=16, num_attention_heads=2,
    )))
    checkpoint = Mock(return_value=tmp_path / "gptq")
    monkeypatch.setattr(gemma, "_resolve_gptq_checkpoint", checkpoint)
    exported = []
    export_options = []
    def record_export(self, **kwargs):
        exported.append(self)
        export_options.append(kwargs)
    monkeypatch.setattr(gemma.Gemma3ModelExporter, "export_onnx", record_export)
    quantize = Mock()
    monkeypatch.setattr(gemma.Gemma3ModelExporter, "weight_quantize_models", quantize)
    parser = argparse.ArgumentParser()
    add_gemma3_export_args(parser)
    from torq.model_export import prefill
    check_compiler = Mock()
    monkeypatch.setattr(prefill, "check_native_compiler", check_compiler)
    arguments = ["--instruct-model", "--native-batch-prefill", "--skip-torq", "--compiler-path", "/selected/compiler"]
    if sizes is not None:
        arguments += ["--batch-prefill-sizes", ",".join(map(str, sizes))]
    arguments += ["--compile-flags", "--torq-max-nss-programs-size=33554432"]
    args = parser.parse_args(arguments)
    gemma.export_gemma3_from_args(args)
    checkpoint.assert_called_once_with(args, "int4")
    quantize.assert_called_once()
    assert exported[0]._read_only_kv_cache is True
    assert exported[0]._gptq_checkpoint == tmp_path / "gptq"
    assert export_options[0]["compiler_path"] == args.compiler_path
    assert export_options[0]["torq_compile_args"] == args.compile_flags
    assert exported[0]._native_prefill_enabled is True
    assert exported[0]._native_batch_prefill == (sizes or ())
    if sizes is None:
        check_compiler.assert_called_once_with(args.compiler_path)
    else:
        check_compiler.assert_not_called()


@pytest.mark.parametrize("native,quantization,read_only,expected", [
    (True,None,False,"constant-loads"),
    (True,None,True,"stream-schedule"),
    (True,"int4",True,"stream-schedule"),
    (False,None,False,"stream-schedule"),
])
def test_gemma_prefetch_defaults_follow_native_cache_configuration(monkeypatch, native, quantization, read_only, expected):
    from torq.model_export.onnx import OnnxModelExporterBase
    from torq.models.gemma3.export import Gemma3ModelExporter

    monkeypatch.setattr(OnnxModelExporterBase, "_model_compile_flags", lambda self, **kwargs: [
        "--torq-prefetch-algorithm=stream-schedule", "--torq-convert-dtypes",
    ])
    exporter = Gemma3ModelExporter.__new__(Gemma3ModelExporter)
    exporter._native_prefill_enabled = native
    exporter._native_batch_prefill = (4,7) if native else ()
    exporter._weight_quantization = quantization
    exporter._read_only_kv_cache = read_only
    flags = exporter._model_compile_flags()
    assert "--torq-convert-dtypes" in flags
    assert [f for f in flags if f.startswith("--torq-prefetch-algorithm=")] == [f"--torq-prefetch-algorithm={expected}"]


@pytest.mark.parametrize("size", [4,7])
def test_liquid_native_conv_matches_every_token_and_nonzero_history(size):
    import onnx_graphsurgeon as gs
    import onnxruntime as ort
    from torq.models.liquid.export import LiquidModelExporter

    rng = np.random.default_rng(3)
    channels = 5
    state = gs.Variable("past_conv.0",np.float32,[1,channels,3])
    x = gs.Variable("x",np.float32,[1,size,3*channels])
    graph = gs.Graph(inputs=[x,state],opset=18)
    def op(kind,name,inputs,shape,**attrs):
        out = gs.Variable(name,np.float32,shape)
        graph.nodes.append(gs.Node(op=kind,name=name,inputs=inputs,outputs=[out],attrs=attrs))
        return out
    def integer(name,value):
        return gs.Constant(name,np.array(value,np.int64))
    layout = op("Transpose","layout",[x],[1,3*channels,size],perm=[0,2,1])
    parts = [gs.Variable(f"part_{i}",np.float32,[1,channels,size]) for i in range(3)]
    graph.nodes.append(gs.Node(op="Split",name="split",inputs=[layout,integer("parts",[channels]*3)],outputs=parts,attrs={"axis":1}))
    gated = op("Mul","gated",[parts[0],parts[2]],[1,channels,size])
    concat = op("Concat","concat",[state,gated],[1,channels,size+3],axis=2)
    conv = op("Conv","conv",[concat,gs.Constant("kernel",rng.normal(size=(channels,1,3)).astype(np.float32))],[1,channels,size+1],group=channels)
    result = op("Slice","result",[conv,integer("start",[1]),integer("end",[size+1]),integer("axis",[2])],[1,channels,size])
    gate = op("Mul","gate",[parts[1],result],[1,channels,size])
    output = op("Transpose","output",[gate],[1,size,channels],perm=[0,2,1])
    present = op("Slice","present",[concat,integer("hs",[size]),integer("he",[size+3]),integer("ha",[2])],[1,channels,3])
    graph.outputs=[output,present]
    original=gs.export_onnx(graph); original.ir_version=10
    rewritten,count=LiquidModelExporter._replace_native_conv1d(original)
    assert count==1
    assert sum(n.op_type=="MatMul" for n in rewritten.graph.node)==1
    opts=ort.SessionOptions(); opts.intra_op_num_threads=1
    reference=ort.InferenceSession(original.SerializeToString(),opts)
    actual=ort.InferenceSession(rewritten.SerializeToString(),opts)
    for history in (np.zeros((1,channels,3),np.float32),rng.normal(size=(1,channels,3)).astype(np.float32)):
        inputs={"x":rng.normal(size=(1,size,3*channels)).astype(np.float32),"past_conv.0":history}
        for result,expected in zip(actual.run(None,inputs),reference.run(None,inputs)):
            np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
    before = actual.run(None,inputs)[0]
    inputs["x"][:,-1] += 100
    after = actual.run(None,inputs)[0]
    np.testing.assert_array_equal(before[:,:-1],after[:,:-1])
    next(n for n in original.graph.node if n.op_type == "Conv").attribute.append(h.make_attribute("auto_pad","SAME_UPPER"))
    with pytest.raises(ValueError,match="depthwise"):
        LiquidModelExporter._replace_native_conv1d(original)


def test_rope_position_vector_control_and_left_hand_frequency_parameter():
    model = h.make_model(h.make_graph([
        h.make_node("Cast",["position_ids"],["position_float"],to=T.FLOAT),
        h.make_node("Add",["position_float","offsets"],["positions"]),
        h.make_node("MatMul",["frequency","positions"],["angles"]),
    ],"main",[],[],[
        numpy_helper.from_array(np.arange(7,dtype=np.float32).reshape(1,7),"offsets"),
        numpy_helper.from_array(np.ones((4,1),np.float32),"frequency"),
    ]))
    assert set(_constant_roles(model)) == {"frequency"}
    assert not model.graph.initializer[0].raw_data
    assert not model.graph.initializer[1].raw_data
    model.graph.node[0].input[0] = "unknown_input"
    with pytest.raises(ValueError,match="unclassified"):
        _constant_roles(model)


def test_native_compile_options_are_explicit_and_family_overrides_rejected():
    from torq.model_export.prefill import native_compile_flags

    flags = native_compile_flags(["--torq-max-nss-programs-size=33554432"])
    assert flags.count("--torq-enable-multi-graph") == 1
    assert "--torq-convert-io-dtype" in flags
    for invalid in ("--torq-enable-multi-graph", "--torq-enable-multi-graph=false", "--torq-convert-io-dtype=false"):
        with pytest.raises(ValueError):
            native_compile_flags([invalid])


def test_compiler_failure_preserves_existing_output(tmp_path, monkeypatch):
    from support.model_export import StubExporter
    import torq.model_export.prefill as prefill

    exporter = StubExporter(tmp_path, {})
    exporter._native_batch_prefill = (4,7)
    old_output = tmp_path/"compiled"
    old_output.mkdir()
    sentinel = old_output/"transformer.vmfb"
    sentinel.write_bytes(b"old")
    def incompatible(path):
        raise RuntimeError("incompatible compiler")
    monkeypatch.setattr(prefill,"check_native_compiler",incompatible)
    with pytest.raises(RuntimeError,match="incompatible"):
        exporter.export_torq(torq_export_dir=old_output)
    assert sentinel.read_bytes() == b"old"
    assert exporter.setup_calls == 0


def test_export_compiles_family_once_and_keeps_family_flags_off_head(tmp_path, monkeypatch):
    import torq.model_export.onnx as me
    import torq.model_export.prefill as prefill
    from support.model_export import StubExporter

    exporter = StubExporter(tmp_path, {"model":_family_graph(1),"model_prefill_4":_family_graph(4),
                                       "model_prefill_7":_family_graph(7),"lm_head":_family_graph(1)})
    exporter._native_batch_prefill = (4,7)
    monkeypatch.setattr(exporter,"_model_compile_flags",lambda **kwargs:["--torq-convert-io-dtype"])
    exporter.export_onnx(validate=False,cleanup=False)
    calls=[]
    monkeypatch.setattr(prefill,"check_native_compiler",lambda path:None)
    monkeypatch.setattr(prefill,"compile_native_family",lambda paths,folder,**kwargs:calls.append(("family",paths,kwargs)))
    monkeypatch.setattr(me,"export_torq",lambda path,folder,**kwargs:calls.append(("head",path,kwargs)))
    exporter.export_torq(compiler_path=tmp_path/"compiler", torq_compile_args=["--torq-max-nss-programs-size=33554432"])
    assert [c[0] for c in calls] == ["family","head"]
    assert list(calls[0][1]) == [1,4,7]
    assert "--torq-enable-multi-graph" in calls[0][2]["compiler_args"]
    assert "--torq-enable-multi-graph" not in calls[1][2]["compiler_args"]
    assert calls[1][2]["use_binary"] is True
    assert "--torq-max-nss-programs-size" not in calls[0][2]["compiler_args"]
    assert "--torq-max-nss-programs-size=33554432" in calls[0][2]["compiler_args"]
    assert "--torq-convert-dtypes" in calls[1][2]["compiler_args"]
    assert calls[0][2]["compiler_path"] == calls[1][2]["compiler_path"] == tmp_path/"compiler"


def test_integer_shape_controls_stay_literals():
    shape=numpy_helper.from_array(np.array([1,4,8],np.int64),"shape")
    model=h.make_model(h.make_graph([h.make_node("Reshape",["x","shape"],["y"])],"main",[],[],[shape]))
    assert _constant_roles(model) == {}
    assert not model.graph.initializer[0].raw_data
    assert list(model.graph.initializer[0].int64_data) == [1,4,8]


def test_gemma_native_rope_keeps_token_layout_after_head_major_concat():
    import onnx_graphsurgeon as gs
    import onnxruntime as ort
    from torq.models.gemma3.export import Gemma3ModelExporter

    frequency=gs.Variable("frequency",np.float32,[1,2,7])
    tokens=gs.Variable("tokens",np.float32,[1,7,2])
    output=gs.Variable("output",np.float32,[1,7,4])
    graph=gs.Graph(opset=18,inputs=[frequency],outputs=[output],nodes=[
        gs.Node(op="Transpose",name="transpose",inputs=[frequency],outputs=[tokens],attrs={"perm":[0,2,1]}),
        gs.Node(op="Concat",name="/model/rotary_emb/Concat",inputs=[tokens,tokens],outputs=[output],attrs={"axis":-1}),
    ])
    original=gs.export_onnx(graph);original.ir_version=10
    Gemma3ModelExporter._native_rope_layout(graph)
    graph.cleanup().toposort()
    rewritten=gs.export_onnx(graph);rewritten.ir_version=10
    assert next(n for n in rewritten.graph.node if n.op_type=="Concat").attribute[0].i==1
    inputs={"frequency":np.arange(14,dtype=np.float32).reshape(1,2,7)}
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    expected=ort.InferenceSession(original.SerializeToString(),opts).run(None,inputs)[0]
    actual=ort.InferenceSession(rewritten.SerializeToString(),opts).run(None,inputs)[0]
    np.testing.assert_array_equal(actual,expected)


def test_liquid_vision_inherits_ordinary_gqa_configuration(tmp_path):
    import json
    from torq.models.liquid.export_vl import LiquidVLModelExporter

    config=dict(hidden_size=8,vocab_size=16,num_attention_heads=2,num_key_value_heads=1,
                head_dim=4,conv_dim=8,conv_L_cache=3,num_hidden_layers=2,layer_types=["conv","full_attention"])
    (tmp_path/"config.json").write_text(json.dumps(config))
    exporter=LiquidVLModelExporter(onnx_source_dir=tmp_path,models_dir=tmp_path,batch_prefill=None)
    assert exporter._native_batch_prefill == ()
    attention=exporter.graph_edit_blocks()["source.convert (attention)"][0]
    assert attention.args[-1] is True
    with pytest.raises(ValueError,match="vision"):
        LiquidVLModelExporter(native_batch_prefill=True, batch_prefill_sizes=(4,7))


@pytest.mark.parametrize("flag", ["--torq-convert-dtypes", "--torq-convert-io-dtype",
    "--torq-disable-slicing", "--torq-enable-split-constants-optimization",
    "--torq-enable-annotate-tied-operands"])
def test_native_flags_check_overrides_after_model_defaults(flag):
    from torq.model_export.prefill import native_compile_flags

    for flags in ([flag, flag + "=false"], [flag + "=0", flag]):
        with pytest.raises(ValueError, match="incompatible option"):
            native_compile_flags(flags)
    flags = native_compile_flags([flag, flag + "=true", flag + "=1"])
    assert flags.count(flag) == 1
    assert flag + "=true" not in flags
    assert flag + "=1" not in flags


@pytest.mark.parametrize("split_weights", [False, True])
def test_native_export_shares_weights_and_keeps_controls_inline(tmp_path, split_weights):
    from support.model_export import StubExporter
    from torq.utils.onnx import load_onnx_lazy, save_onnx

    components = {}
    for size in (1, 4, 7):
        model = _family_graph(size, width=16)
        model.graph.initializer.append(numpy_helper.from_array(np.arange(128 + size, dtype=np.int64), "offsets"))
        model.graph.node.append(h.make_node("Add", ["position_ids", "offsets"], ["positions"]))
        components["model" if size == 1 else f"model_prefill_{size}"] = model
    # The independent head follows the ordinary --split-weights policy.
    components["lm_head"] = _family_graph(1, width=16)
    exporter = StubExporter(tmp_path, components, split_weights=split_weights)
    exporter._native_batch_prefill = (4, 7)
    exporter.export_onnx(validate=True, cleanup=False)

    directory = exporter.export_dir
    data_path = directory / "transformer.weights.data"
    assert data_path.stat().st_size == 16 * 16 * 4
    paths = {size: exporter._export_paths["model" if size == 1 else f"model_prefill_{size}"] for size in (1, 4, 7)}
    references = []
    for size, path in paths.items():
        model = onnx.load(path, load_external_data=False)
        tensors = {tensor.name: tensor for tensor in model.graph.initializer}
        references.append({entry.key: entry.value for entry in tensors["weight"].external_data})
        assert not tensors["weight"].raw_data
        assert not tensors["offsets"].external_data
        np.testing.assert_array_equal(numpy_helper.to_array(tensors["offsets"]), np.arange(128 + size))
        onnx.checker.check_model(str(path), full_check=True)
        assert not path.with_name(path.name + ".data").exists()
    assert references == [dict(location=data_path.name, offset="0", length="1024")] * 3
    assert (directory / "lm_head.onnx.data").exists() == split_weights
    # Re-saving one member must not damage the shared file or its siblings.
    save_onnx(load_onnx_lazy(paths[4]), paths[4])
    validate_native_onnx(paths)
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    assert set(import_native_family(paths, directory / "family.mlirbc")["entries"]) == {"main", "main_4", "main_7"}


def test_shared_weights_preserve_distinct_parameters(tmp_path):
    from torq.model_export.prefill import share_native_weights
    from torq.utils.onnx import save_onnx

    paths = {size: tmp_path / f"decoder_{size}.onnx" for size in (1, 4)}
    for size, path in paths.items():
        save_onnx(_family_graph(size, weight=float(size), width=16), path, split=True)
    share_native_weights(paths.values())
    assert (tmp_path / "transformer.weights.data").stat().st_size == 2 * 1024
    for size, path in paths.items():
        np.testing.assert_array_equal(numpy_helper.to_array(onnx.load(path).graph.initializer[0]), np.eye(16) * size)
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    with pytest.raises(ValueError, match="parameters differ"):
        import_native_family(paths, tmp_path / "family.mlirbc")


def test_shared_weights_failed_write_preserves_original_graphs(tmp_path, monkeypatch):
    from torq.model_export.prefill import share_native_weights
    from torq.utils.onnx import save_onnx

    paths = [tmp_path / f"decoder_{size}.onnx" for size in (1, 4)]
    for size, path in zip((1, 4), paths):
        save_onnx(_family_graph(size, width=16), path, split=True)
    original = {path: path.read_bytes() for path in paths}
    save = onnx.save
    def fail_second(model, path):
        if path.name == "1.onnx":
            raise OSError("disk full")
        save(model, path)
    monkeypatch.setattr(onnx, "save", fail_second)
    with pytest.raises(OSError, match="disk full"):
        share_native_weights(paths)
    assert not (tmp_path / "transformer.weights.data").exists()
    for path in paths:
        assert path.read_bytes() == original[path]
        np.testing.assert_array_equal(numpy_helper.to_array(onnx.load(path).graph.initializer[0]), np.eye(16))


def test_native_gemma_quantization_shares_the_quantized_weights(tmp_path, monkeypatch):
    from support.model_export import StubExporter
    from torq.models.gemma3.export import Gemma3ModelExporter

    exporter = StubExporter(tmp_path, {
        "model": _family_graph(1, width=64),
        "model_prefill_4": _family_graph(4, width=64),
    })
    exporter._native_batch_prefill = (4,)
    exporter._weight_quantization = "int4"
    exporter._gptq_checkpoint = None
    monkeypatch.setattr(exporter, "_summarize_weight_quantization", lambda *args: dict(
        kl=0., cosine=1., max_abs_error=0., classification="test"), raising=False)
    exporter.export_onnx(validate=True, cleanup=False)
    Gemma3ModelExporter.weight_quantize_models(exporter)

    assert (exporter._quantize_dir / "transformer.weights.data").exists()
    references = []
    for path in exporter._export_paths.values():
        assert path.parent == exporter._quantize_dir
        stored = onnx.load(path, load_external_data=False)
        references.append({tensor.name: tuple((entry.key, entry.value) for entry in tensor.external_data)
                           for tensor in stored.graph.initializer if tensor.external_data})
        onnx.checker.check_model(str(path), full_check=True)
        assert not path.with_name(path.name + ".data").exists()
    assert references[0] == references[1]
    assert references[0]


@pytest.mark.parametrize("failure", ["publish", "install"])
def test_shared_weights_publish_failures_leave_all_models_readable(tmp_path, monkeypatch, failure):
    import torq.model_export.prefill as prefill
    from torq.utils.onnx import save_onnx

    paths = [tmp_path / f"decoder_{size}.onnx" for size in (1, 4)]
    for size, path in zip((1, 4), paths):
        save_onnx(_family_graph(size, width=16), path, split=True)
    data_path = tmp_path / "transformer.weights.data"
    if failure == "publish":
        data_path.write_bytes(b"existing weights")
    else:
        replace = prefill.os.replace
        def fail_second(source, destination):
            if destination == paths[1]:
                raise OSError("cannot install graph")
            replace(source, destination)
        monkeypatch.setattr(prefill.os, "replace", fail_second)
    with pytest.raises(OSError):
        prefill.share_native_weights(paths)
    if failure == "publish":
        assert data_path.read_bytes() == b"existing weights"
    for path in paths:
        assert path.with_name(path.name + ".data").exists()
        np.testing.assert_array_equal(numpy_helper.to_array(onnx.load(path).graph.initializer[0]), np.eye(16))


@pytest.mark.parametrize("family", ["gemma3", "gemma3-int4", "liquid"])
@pytest.mark.parametrize("mode", ["ordinary", "explicit", "automatic"])
def test_export_native_size_selection(tmp_path, monkeypatch, family, mode):
    import json
    import sys
    from types import SimpleNamespace
    from torq.model_export import prefill_sizes as probe
    from torq.models.gemma3.export import Gemma3ModelExporter
    from torq.models.liquid.export import LiquidModelExporter

    source = tmp_path / "source"
    source.mkdir()
    config = dict(hidden_size=32, vocab_size=16, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=4, conv_dim=4, conv_L_cache=3, num_hidden_layers=2,
                  layer_types=["conv", "full_attention"], sliding_window=16)
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer.json").write_text("{}")
    onnx.save(_family_graph(1, width=32), source / "model.onnx")
    options = dict(models_dir=tmp_path / "models", onnx_source_dir=source, max_gen_tokens=16,
                   extract_embeddings=True, split_lm_head=True, read_only_kv_cache=False,
                   native_batch_prefill=mode != "ordinary",
                   batch_prefill_sizes=(4, 7) if mode == "explicit" else None)
    if family.startswith("gemma3"):
        monkeypatch.setattr("torq.models.gemma3.export.AutoConfig.from_pretrained",
                            lambda *args, **kwargs: SimpleNamespace(**config))
        exporter = Gemma3ModelExporter(**options, weight_quantization="int4" if family == "gemma3-int4" else None)
    else:
        exporter = LiquidModelExporter("230m", **options)
    cls = type(exporter)
    monkeypatch.setattr(cls, "_model_compile_flags", lambda self, **kwargs: ["--torq-prefetch-algorithm=constant-loads"])
    monkeypatch.setattr(cls, "_load_onnx", lambda self: {"model": _family_graph(1, width=32)})
    def make_chunks(self, source_model=None):
        return {f"model_prefill_{size}": _family_graph(size, width=32)
                for size in self._native_batch_prefill}
    monkeypatch.setattr(cls, "_make_native_prefill", make_chunks)
    def make_static(self):
        self._components = {"model": _family_graph(1, width=32), **{
            f"model_prefill_{size}": _family_graph(size, width=32) for size in self._native_batch_prefill
        }}
    monkeypatch.setattr(cls, "make_static", make_static)
    monkeypatch.setattr(cls, "optimize_model", lambda *args: None)
    exports = []
    quantized_components = []
    def patch_decoder(self, path, component):
        exports.append(component)
        model = onnx.load(path)
        weight = next(i for i in model.graph.initializer if i.name == "weight")
        weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight) * 3, "weight"))
        onnx.save(model, path)
    monkeypatch.setattr(cls, "apply_post_static_patches", patch_decoder)
    if family == "gemma3-int4":
        def summarize(self, path, *args):
            quantized_components.append(path.name)
            return dict(kl=0, cosine=1, max_abs_error=0, classification="test")
        monkeypatch.setattr(cls, "_summarize_weight_quantization", summarize)
    probed_weights = []
    recommend = probe.recommend
    def inspect_final(path, *args, **kwargs):
        assert path == exporter._export_paths["model"]
        assert path != source / "model.onnx"
        model = onnx.load(path)
        assert ("quantized" in path.parts) == (family == "gemma3-int4")
        if family == "gemma3-int4":
            assert any(i.data_type == T.INT4 for i in model.graph.initializer)
            assert any(n.op_type == "DequantizeLinear" for n in model.graph.node)
        else:
            weight = next(i for i in model.graph.initializer if i.name == "weight")
            np.testing.assert_array_equal(numpy_helper.to_array(weight), np.eye(32) * 3)
        probed_weights.append([(i.name, i.data_type, tuple(i.dims), i.raw_data)
                               for i in model.graph.initializer])
        return recommend(path, *args, **kwargs)
    monkeypatch.setattr(probe, "recommend", inspect_final)
    monkeypatch.setattr(cls, "validate_onnx", lambda self: None)
    calls = []
    def run_probe(tool, path, chip, flags):
        assert tool == sys.executable
        assert chip == "SL2610"
        assert "--torq-prefetch-algorithm=constant-loads" in flags
        assert "--torq-enable-split-constants-optimization" in flags
        assert "--torq-enable-multi-graph" not in flags
        assert "--torq-max-nss-programs-size=33554432" in flags
        calls.append(path.name)
        return ""
    monkeypatch.setattr(probe, "run_probe", run_probe)
    monkeypatch.setattr(probe, "_parse_tile_and_fuse",
                        lambda _, m, k, n: probe.ProbeResult(m, 4 if m == 7 else 8, False, False))
    exporter.describe_graph_edits()
    assert calls == []  # Read-only plans never trigger compilation.
    exporter.export_onnx(validate=True, cleanup=False, compiler_path=sys.executable,
                         torq_compile_args=["--torq-max-nss-programs-size=33554432"])
    if family == "gemma3-int4":
        if mode == "automatic":
            assert not calls  # Wait for the normal decoder quantization.
            assert exporter._native_batch_prefill == ()
        exporter.weight_quantize_models()
    if mode == "automatic":
        final = onnx.load(exporter._export_paths["model"])
        assert probed_weights == [[(i.name, i.data_type, tuple(i.dims), i.raw_data)
                                   for i in final.graph.initializer]]
    expected = (4, 7, 16) if mode == "automatic" else (4, 7) if mode == "explicit" else ()
    assert exporter._native_batch_prefill == expected
    assert set(exporter._export_paths) == {"model", *(f"model_prefill_{size}" for size in expected)}
    assert bool(calls) == (mode == "automatic")
    assert exporter.export_dir.name == "static"
    assert exporter._quantize_dir.name == "static"
    assert exporter._convert_dir.name == "static"
    assert sorted(exports) == sorted(exporter._export_paths)  # Each graph exported once.
    if family == "gemma3-int4":
        assert len(quantized_components) == len(exporter._export_paths)
        assert len(set(quantized_components)) == len(quantized_components)
    assert exporter._torq_dir.parent == (exporter._quantize_dir if family == "gemma3-int4"
                                        else exporter.export_dir)
    if mode == "automatic":
        assert len(calls) == 14  # Decode plus B=4..16, bounded by the context.
        assert json.loads((exporter._models_dir / "prefill-probes/prefill_sizes.json").read_text())["chosen"] == list(expected)
        if family.startswith("gemma3"):
            assert all(f"model_prefill_{size}" in exporter._opt_configs for size in expected)


@pytest.mark.parametrize("chosen", [None, [], [1, 4], [4, 4], [7, 4], [17]])
def test_automatic_sizes_reject_no_recommendation_or_invalid_sizes(tmp_path, monkeypatch, chosen):
    from types import SimpleNamespace
    from torq.model_export import prefill_sizes
    from torq.model_export.prefill import probe_native_sizes

    monkeypatch.setattr(prefill_sizes, "recommend", lambda *args, **kwargs:
                        None if chosen is None else SimpleNamespace(chosen=chosen))
    with pytest.raises(RuntimeError, match="--batch-prefill-sizes"):
        probe_native_sizes(tmp_path / "source.onnx", context=16, minimum=4,
                           compiler_args=[], compiler_path=None, work_dir=tmp_path)


def test_automatic_probe_failure_keeps_exported_decoder(tmp_path, monkeypatch):
    from support.model_export import StubExporter
    from torq.model_export import prefill_sizes

    exporter = StubExporter(tmp_path, {"model": _family_graph(1)})
    exporter._native_prefill_enabled = True
    exporter._max_gen_tokens = 16
    monkeypatch.setattr(exporter, "_model_compile_flags", lambda **kwargs: [])
    def fail(path, *args, **kwargs):
        assert path == exporter._export_paths["model"]
        onnx.checker.check_model(str(path))
        raise prefill_sizes.ProbeError("compile failed")
    monkeypatch.setattr(prefill_sizes, "recommend", fail)
    with pytest.raises(RuntimeError, match="--batch-prefill-sizes.*compile failed"):
        exporter.export_onnx(validate=False)
    assert exporter._native_batch_prefill == ()
    assert set(exporter._export_paths) == {"model"}
    onnx.checker.check_model(str(exporter._export_paths["model"]))
    assert exporter.load_calls == 1


@pytest.mark.parametrize("enabled,sizes,static,error", [
    (False, (4, 7), True, "requires --native-batch-prefill"),
    (True, None, False, "requires static exports"),
    ((4, 7), None, True, "must be a boolean"),
])
def test_native_mode_validation_before_source_loading(enabled, sizes, static, error):
    from torq.models.gemma3.export import Gemma3ModelExporter
    from torq.models.liquid.export import LiquidModelExporter

    for cls in (Gemma3ModelExporter, LiquidModelExporter):
        with pytest.raises(ValueError, match=error):
            cls(native_batch_prefill=enabled, batch_prefill_sizes=sizes, static_models=static,
                extract_embeddings=True, split_lm_head=True, read_only_kv_cache=False)
