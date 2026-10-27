# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse

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
    assert parser.parse_args([]).batch_prefill == 64
    assert parser.parse_args(["--native-batch-prefill", "4,7"]).native_batch_prefill == (4, 7)
    with pytest.raises(SystemExit):
        parser.parse_args(["--batch-prefill", "0", "--native-batch-prefill", "4,7"])


@pytest.mark.parametrize("override", [dict(sizes=(4,4)), dict(sizes=(7,4)), dict(sizes=(1,4)),
    dict(sizes=(4,17)), dict(sizes=(2,), minimum=4), dict(read_only=True), dict(static=False),
    dict(embeddings=False), dict(split_head=False), dict(individual=True), dict(unsupported=True), dict(batch_prefill=64)])
def test_reject_unsupported_options(override):
    options = dict(sizes=(4,7), batch_prefill=None, context=16, minimum=2, static=True,
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


def _family_graph(size, weight=1.):
    # Full-cache identity state and a real final-token hidden output.
    inputs = [h.make_tensor_value_info("token_embedding",T.FLOAT,[1,size,4]),
              h.make_tensor_value_info("position_ids",T.INT64,[1,1]),
              h.make_tensor_value_info("past_key_values.0.key_value",T.FLOAT,[1,2,16,4])]
    outputs = [h.make_tensor_value_info("last_hidden_states",T.FLOAT,[1,1,4]),
               h.make_tensor_value_info("present.0.key_value",T.FLOAT,[1,2,16,4])]
    initializers = [numpy_helper.from_array(np.eye(4,dtype=np.float32)*weight,"weight")]
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
    validate_native_onnx(paths)
    onnx.save(_family_graph(7,weight=2.),paths[7])
    with pytest.raises(ValueError,match="parameters differ"):
        import_native_family(paths,output)


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


def test_native_compile_options_are_explicit_and_generation_options_rejected():
    from torq.model_export.prefill import native_compile_flags

    flags = native_compile_flags(["--torq-max-nss-programs-size=33554432"])
    assert flags.count("--torq-enable-batch-prefill") == 1
    assert "--torq-convert-io-dtype" in flags
    assert not any(f.startswith("--torq-prefill-") for f in flags)
    for invalid in ("--torq-prefill-batch-sizes=4", "--torq-enable-batch-prefill", "--torq-convert-io-dtype=false"):
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
    assert "--torq-enable-batch-prefill" in calls[0][2]["compiler_args"]
    assert "--torq-enable-batch-prefill" not in calls[1][2]["compiler_args"]
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
        LiquidVLModelExporter(native_batch_prefill=(4,7))
