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


@pytest.mark.parametrize("value", ["", "1", "4,4", "7,4", "4,", "4, 7", "04,7", "-4", "4.0", "４"])
def test_invalid_sizes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_native_sizes(value)


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
