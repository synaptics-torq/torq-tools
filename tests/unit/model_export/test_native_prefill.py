# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import re

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper as h, numpy_helper

from torq.model_export.prefill import (
    _constant_roles, FamilyContract, State, decoder_states, import_native_family,
    parse_native_sizes, validate_batch_prefill_options, validate_native_onnx,
)
from torq.models.gemma3 import add_gemma3_export_args
from torq.models.liquid import add_liquid_export_args
from torq.models.qwen import add_qwen_export_args


@pytest.mark.parametrize("value", ["", "1", "4,4", "7,4", "4,", "4, 7", "04,7", "-4", "4.0", "４"])
def test_invalid_sizes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_native_sizes(value)


@pytest.mark.parametrize("add_args", [add_gemma3_export_args, add_liquid_export_args, add_qwen_export_args])
def test_cli_batch_prefill_flag_and_sizes(add_args):
    parser = argparse.ArgumentParser()
    add_args(parser)
    # argparse default is None: the shared CLI check resolves it (on for
    # compile runs, off for --skip-torq) and the base constructor resolves it
    # for unsupported configurations.
    defaults = parser.parse_args([])
    assert defaults.batch_prefill is None
    assert defaults.batch_prefill_sizes is None
    explicit_on = parser.parse_args(["--batch-prefill"])
    assert explicit_on.batch_prefill is True
    explicit_off = parser.parse_args(["--no-batch-prefill"])
    assert explicit_off.batch_prefill is False
    automatic = parser.parse_args(["--batch-prefill"])
    assert automatic.batch_prefill_sizes is None
    sized = parser.parse_args(["--batch-prefill", "--batch-prefill-sizes", "4,7"])
    assert sized.batch_prefill is True
    assert sized.batch_prefill_sizes == (4, 7)
    # The legacy integer form is gone.
    for size in ("0", "64"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--batch-prefill", size])


@pytest.mark.parametrize("override", [dict(sizes=()), dict(sizes=(4,4)), dict(sizes=(7,4)), dict(sizes=(1,4)),
    dict(sizes=(4,17)), dict(sizes=(2,), minimum=4), dict(read_only=True), dict(static=False),
    dict(embeddings=False), dict(split_head=False), dict(individual=True), dict(unsupported=True)])
def test_reject_unsupported_options(override):
    options = dict(sizes=(4,7), enabled=True, context=16, minimum=2, static=True,
                   embeddings=True, split_head=True, read_only=False, individual=False, unsupported=False)
    options.update(override)
    with pytest.raises(ValueError):
        validate_batch_prefill_options(**options)


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


def _contract(path, *, read_only):
    return FamilyContract(states=decoder_states(onnx.load(path), read_only=read_only))


def test_family_import_and_numerical_gate(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    paths = {b: tmp_path/f"decoder_{b}.onnx" for b in (1,4,7)}
    for b,path in paths.items():
        onnx.save(_family_graph(b),path)
    output = tmp_path/"family.mlirbc"
    metadata = import_native_family(paths,output,_contract(paths[1],read_only=False))
    assert output.read_bytes().startswith(b"ML\xefR")
    assert set(metadata["entries"]) == {"main","main_4","main_7"}
    from iree.compiler import ir

    with ir.Context():
        assembly = str(ir.Module.parse(output.read_bytes()))
    assert set(re.findall(r"dense_resource<(weight[^>]*)>", assembly)) == {"weight"}
    validate_native_onnx(paths,contract=_contract(paths[1],read_only=False))
    onnx.save(_family_graph(7,weight=2.),paths[7])
    with pytest.raises(ValueError,match="parameters differ"):
        import_native_family(paths,output,_contract(paths[1],read_only=False))


def _read_only_family_graph(size, width=4):
    model = _family_graph(size, width=width)
    model.graph.input[2].CopyFrom(h.make_tensor_value_info("past_key_values.0.key_value", T.FLOAT, [1,16,2,width]))
    model.graph.output[1].CopyFrom(h.make_tensor_value_info("present.0.key_value", T.FLOAT, [1,size,2,width]))
    del model.graph.node[-1]
    model.graph.initializer.append(numpy_helper.from_array(np.array([2], np.int64), "rows_axis"))
    model.graph.node.extend([
        h.make_node("Unsqueeze", ["token_embedding", "rows_axis"], ["rows"]),
        h.make_node("Concat", ["rows", "rows"], ["present.0.key_value"], axis=2),
    ])
    return model


def test_read_only_family_with_convolution_state_first(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    paths = {size: tmp_path / f"decoder_{size}.onnx" for size in (1, 4, 7)}
    for size, path in paths.items():
        model = _read_only_family_graph(size)
        model.graph.input.insert(2, h.make_tensor_value_info("past_conv.0", T.FLOAT, [1, 4, 3]))
        model.graph.output.insert(1, h.make_tensor_value_info("present_conv.0", T.FLOAT, [1, 4, 3]))
        model.graph.node[1].output[0] = "last_token"
        for name, values in (("history_start", [-3]), ("history_end", [2**63 - 1]), ("history_axis", [2])):
            model.graph.initializer.append(numpy_helper.from_array(np.array(values, np.int64), name))
        model.graph.node.extend([
            h.make_node("Transpose", ["token_embedding"], ["conv_tokens"], perm=[0, 2, 1]),
            h.make_node("Concat", ["past_conv.0", "conv_tokens"], ["history"], axis=2),
            h.make_node("Slice", ["history", "history_start", "history_end", "history_axis"], ["present_conv.0"]),
            h.make_node("ReduceSum", ["present_conv.0", "history_axis"], ["history_sum"], keepdims=1),
            h.make_node("Transpose", ["history_sum"], ["history_hidden"], perm=[0, 2, 1]),
            h.make_node("Add", ["last_token", "history_hidden"], ["last_hidden_states"]),
        ])
        onnx.save(model, path)
    contract = _contract(paths[1], read_only=True)
    assert contract.states[0].seq_axis is None
    assert contract.states[1].rows_only
    validate_native_onnx(paths, contract=contract)
    metadata = import_native_family(paths, tmp_path / "family.mlirbc", contract)
    assert metadata["kv_cache_sequence_axis"] == 1
    outputs = metadata["entries"]["main_7"]["outputs"]
    assert outputs[1]["shape"] == [1, 4, 3]
    assert outputs[2]["shape"] == [1, 7, 2, 4]


def test_read_only_family_import_and_sequential_rows(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    paths = {b: tmp_path / f"decoder_{b}.onnx" for b in (1,4,7)}
    for size, path in paths.items():
        onnx.save(_read_only_family_graph(size), path)
    metadata = import_native_family(paths, tmp_path / "family.mlirbc", _contract(paths[1], read_only=True))
    assert metadata["kv_cache_sequence_axis"] == 1
    assert metadata["entries"]["main_7"]["outputs"][1]["shape"] == [1,7,2,4]
    validate_native_onnx(paths, contract=_contract(paths[1], read_only=True))
    # A chunk returning the wrong rows must fail the populated-cache gate.
    model = _read_only_family_graph(7)
    model.graph.node.insert(len(model.graph.node) - 1, h.make_node("Neg", ["rows"], ["negated"]))
    model.graph.node[-1].input[1] = "negated"
    onnx.save(model, paths[7])
    with pytest.raises(AssertionError):
        validate_native_onnx(paths, contract=_contract(paths[1], read_only=True))


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
    metadata = import_native_family(paths, tmp_path / "family.mlirbc", _contract(paths[1], read_only=True))
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
            import_native_family(paths, tmp_path / "family.mlirbc", _contract(paths[1], read_only=True))
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
    monkeypatch.setattr(gemma.Gemma3ModelExporter, "export_torq", Mock())
    quantize = Mock()
    monkeypatch.setattr(gemma.Gemma3ModelExporter, "weight_quantize_models", quantize)
    parser = argparse.ArgumentParser()
    add_gemma3_export_args(parser)
    from torq.model_export import prefill
    check_compiler = Mock()
    monkeypatch.setattr(prefill, "check_batch_prefill_compiler", check_compiler)
    arguments = ["--instruct-model", "--batch-prefill", "--compiler-path", "/selected/compiler"]
    if sizes is not None:
        arguments += ["--batch-prefill-sizes", ",".join(map(str, sizes))]
    arguments += ["--compile-flags", "--torq-max-nss-programs-size=33554432"]
    args = parser.parse_args(arguments)
    gemma.export_gemma3_from_args(args)
    checkpoint.assert_called_once_with(args, "int4")
    quantize.assert_called_once()
    assert quantize.call_args.kwargs == dict(
        validate=not args.skip_validation, cleanup=not args.no_onnx_cleanup,
        compiler_path=args.compiler_path, torq_compile_args=args.compile_flags or [])
    assert exported[0]._read_only_kv_cache is True
    assert exported[0]._gptq_checkpoint == tmp_path / "gptq"
    assert export_options[0]["compiler_path"] == args.compiler_path
    assert export_options[0]["torq_compile_args"] == args.compile_flags
    assert exported[0]._batch_prefill_enabled is True
    assert exported[0]._batch_prefill_sizes == (sizes or ())
    # Compile runs always need the selected compiler: explicit sizes compile
    # the family, omitted sizes probe first.
    check_compiler.assert_called_once_with(args.compiler_path)


@pytest.mark.parametrize("arguments,raise_match,enabled,sizes,compiler_calls", [
    # Rule 1: explicit --batch-prefill (or sizes, which imply it) + --skip-torq fails.
    (["--batch-prefill", "--skip-torq"], "requires compilation", None, None, 0),
    (["--batch-prefill-sizes", "4,12", "--skip-torq"], "requires compilation", None, None, 0),
    # Rule 2: the default with --skip-torq is a decode-only export (no compiler).
    (["--skip-torq"], None, False, (), 0),
    # Rule 3: the default on an unsupported configuration warns and exports decode-only.
    (["--no-split-lm-head"], None, False, (), 0),
    # Rule 3 (explicit): an explicit --batch-prefill on an unsupported configuration fails.
    (["--batch-prefill", "--no-split-lm-head"], "requires static exports", None, None, 0),
    # Explicit off on a supported configuration: off, no warning, no compiler.
    (["--no-batch-prefill"], None, False, (), 0),
    # Rule 4: otherwise on; omitted sizes probe automatically.
    ([], None, True, (), 1),
])
def test_batch_prefill_cli_resolution_rules(tmp_path, monkeypatch, caplog,
                                             arguments, raise_match, enabled, sizes, compiler_calls):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from torq.models.gemma3 import export as gemma

    monkeypatch.setattr(gemma.AutoConfig, "from_pretrained", Mock(return_value=SimpleNamespace(
        hidden_size=8, vocab_size=16, num_attention_heads=2,
    )))
    monkeypatch.setattr(gemma, "_resolve_gptq_checkpoint", Mock(return_value=None))
    # configure_logging clears the root handlers, which would drop caplog.
    monkeypatch.setattr(gemma, "configure_logging", lambda *a, **k: None)
    exported = []
    monkeypatch.setattr(gemma.Gemma3ModelExporter, "export_onnx",
                        lambda self, **kwargs: exported.append(self))
    monkeypatch.setattr(gemma.Gemma3ModelExporter, "export_torq", Mock())
    monkeypatch.setattr(gemma.Gemma3ModelExporter, "weight_quantize_models", Mock())
    parser = argparse.ArgumentParser()
    add_gemma3_export_args(parser)
    from torq.model_export import prefill
    check_compiler = Mock()
    monkeypatch.setattr(prefill, "check_batch_prefill_compiler", check_compiler)

    args = parser.parse_args(arguments)
    with caplog.at_level("WARNING"):
        if raise_match:
            with pytest.raises(ValueError, match=raise_match):
                gemma.export_gemma3_from_args(args)
        else:
            gemma.export_gemma3_from_args(args)
    if raise_match:
        assert not exported
        return
    assert exported[0]._batch_prefill_enabled is enabled
    assert exported[0]._batch_prefill_sizes == sizes
    assert check_compiler.call_count == compiler_calls
    warned = "off for this export" in caplog.text
    if arguments == ["--no-split-lm-head"]:
        assert warned  # rule 3: the default on an unsupported configuration
    else:
        assert not warned


def test_non_batch_prefill_exporters_default_silently(tmp_path, caplog):
    # Exporters without the batch-prefill option keep the base default:
    # off, with no resolution warning.
    from support.model_export import StubExporter

    with caplog.at_level("WARNING"):
        exporter = StubExporter(tmp_path, {"model": _family_graph(1)})
    assert exporter._batch_prefill_enabled is False
    assert exporter._batch_prefill_sizes == ()
    assert "batch-prefill" not in caplog.text


def test_api_default_unsupported_warns_and_explicit_sizes_fail(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from torq.models.gemma3 import export as gemma

    monkeypatch.setattr(gemma.AutoConfig, "from_pretrained", Mock(return_value=SimpleNamespace(
        hidden_size=8, vocab_size=16, num_attention_heads=2,
    )))
    kwargs = dict(static_models=True, extract_embeddings=True, split_lm_head=True,
                  read_only_kv_cache=False, keep_individual_kv_io=True, models_dir=tmp_path)
    # Python-API default on an unsupported configuration: warn, export decode-only.
    with caplog.at_level("WARNING"):
        exporter = gemma.Gemma3ModelExporter(**kwargs)
    assert exporter._batch_prefill_enabled is False
    assert exporter._batch_prefill_sizes == ()
    assert "off for this export" in caplog.text
    # Sizes imply an explicit request: invalid or off+sizes fails rather than
    # silently dropping them.
    for bad in (dict(batch_prefill_sizes=(1, 4)),
                dict(batch_prefill=True, batch_prefill_sizes=(4, 4)),
                dict(batch_prefill=False, batch_prefill_sizes=(4, 7))):
        options = {key: value for key, value in kwargs.items()
                   if key not in ("keep_individual_kv_io", "read_only_kv_cache")}
        options.update(bad)
        with pytest.raises(ValueError):
            gemma.Gemma3ModelExporter(**options)


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
    exporter._batch_prefill_enabled = native
    exporter._batch_prefill_sizes = (4,7) if native else ()
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
    from torq.model_export.prefill import batch_prefill_compile_flags

    flags = batch_prefill_compile_flags(["--torq-max-nss-programs-size=33554432"])
    assert flags.count("--torq-enable-multi-graph") == 1
    assert "--torq-convert-io-dtype" in flags
    for invalid in ("--torq-enable-multi-graph", "--torq-enable-multi-graph=false", "--torq-convert-io-dtype=false"):
        with pytest.raises(ValueError):
            batch_prefill_compile_flags([invalid])


def test_compiler_failure_preserves_existing_output(tmp_path, monkeypatch):
    from support.model_export import StubExporter
    import torq.model_export.prefill as prefill

    exporter = StubExporter(tmp_path, {})
    exporter._batch_prefill_sizes = (4,7)
    old_output = tmp_path/"compiled"
    old_output.mkdir()
    sentinel = old_output/"transformer.vmfb"
    sentinel.write_bytes(b"old")
    def incompatible(path):
        raise RuntimeError("incompatible compiler")
    monkeypatch.setattr(prefill,"check_batch_prefill_compiler",incompatible)
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
    exporter._batch_prefill_sizes = (4,7)
    monkeypatch.setattr(exporter,"_model_compile_flags",lambda **kwargs:["--torq-convert-io-dtype"])
    exporter.export_onnx(validate=False,cleanup=False)
    calls=[]
    monkeypatch.setattr(prefill,"check_batch_prefill_compiler",lambda path:None)
    monkeypatch.setattr(prefill,"compile_native_family",lambda paths,folder,contract=None,**kwargs:calls.append(("family",paths,kwargs,contract)))
    monkeypatch.setattr(me,"export_torq",lambda path,folder,**kwargs:calls.append(("head",path,kwargs)))
    exporter.export_torq(compiler_path=tmp_path/"compiler", torq_compile_args=["--torq-max-nss-programs-size=33554432"])
    assert [c[0] for c in calls] == ["family","head"]
    assert isinstance(calls[0][3], prefill.FamilyContract)
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
    assert exporter._batch_prefill_sizes == ()
    attention=exporter.graph_edit_blocks()["source.convert (attention)"][0]
    assert attention.args[-1] is True
    # Without --split-lm-head the base gate names the missing prerequisite.
    with pytest.raises(ValueError,match="--split-lm-head"):
        LiquidVLModelExporter(batch_prefill=True, batch_prefill_sizes=(4,7))
    # With split-lm-head on, a vision build option is rejected for the vision
    # reason (not the split reason).
    with pytest.raises(ValueError,match="vision"):
        LiquidVLModelExporter(batch_prefill=True, batch_prefill_sizes=(4,7),
                              split_lm_head=True, compile_vision=True)


def test_liquid_vision_constructor_preserves_pre_super_options(tmp_path):
    import json
    from torq.models.liquid.export_vl import LiquidVLModelExporter

    config=dict(hidden_size=8,vocab_size=16,num_attention_heads=2,num_key_value_heads=1,
                head_dim=4,conv_dim=8,conv_L_cache=3,num_hidden_layers=2,layer_types=["conv","full_attention"])
    (tmp_path/"config.json").write_text(json.dumps(config))
    exporter=LiquidVLModelExporter(onnx_source_dir=tmp_path,models_dir=tmp_path,
                                   max_gen_tokens=32,keep_individual_kv_io=True,split_lm_head=True)
    # These attributes are assigned before the base __init__; the base must preserve them.
    assert exporter._max_gen_tokens == 32
    assert exporter._keep_individual_kv_io is True
    assert exporter._split_lm_head is True
    assert exporter._read_only_kv_cache is False


@pytest.mark.parametrize("flag", ["--torq-convert-dtypes", "--torq-convert-io-dtype",
    "--torq-disable-slicing", "--torq-enable-split-constants-optimization",
    "--torq-enable-annotate-tied-operands"])
def test_native_flags_check_overrides_after_model_defaults(flag):
    from torq.model_export.prefill import batch_prefill_compile_flags

    for flags in ([flag, flag + "=false"], [flag + "=0", flag]):
        with pytest.raises(ValueError, match="incompatible option"):
            batch_prefill_compile_flags(flags)
    flags = batch_prefill_compile_flags([flag, flag + "=true", flag + "=1"])
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
    exporter._batch_prefill_sizes = (4, 7)
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
    validate_native_onnx(paths, contract=_contract(paths[1], read_only=False))
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    assert set(import_native_family(paths, directory / "family.mlirbc", _contract(paths[1], read_only=False))["entries"]) == {"main", "main_4", "main_7"}


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
        import_native_family(paths, tmp_path / "family.mlirbc", _contract(paths[1], read_only=False))


def test_decoder_states_rejects_unknown_state_names():
    model = _family_graph(4)
    model.graph.input[2].name = "cache.0"
    with pytest.raises(ValueError, match="unknown native decoder state"):
        decoder_states(model, read_only=False)


def test_fold_static_attention_mask_folds_input_to_ones_constant():
    import onnx_graphsurgeon as gs
    from torq.models.liquid.export import LiquidModelExporter

    mask = gs.Variable("attention_mask", np.int64, [1, 4])
    sliced = gs.Variable("sliced", np.int64, [1, 3])
    casted = gs.Variable("casted", np.float32, [1, 3])
    graph = gs.Graph(
        inputs=[mask],
        outputs=[casted],
        nodes=[
            gs.Node(op="Slice", name="Slice_Mask",
                    inputs=[mask,
                            gs.Constant("s", np.array([1], dtype=np.int64)),
                            gs.Constant("e", np.array([4], dtype=np.int64)),
                            gs.Constant("a", np.array([1], dtype=np.int64))],
                    outputs=[sliced]),
            gs.Node(op="Cast", name="Mask_Cast", inputs=[sliced], outputs=[casted],
                    attrs={"to": 1}),
        ],
        opset=18,
    )
    assert LiquidModelExporter._fold_static_attention_mask(graph) == 1
    assert "attention_mask" not in [t.name for t in graph.inputs]
    folded = next(n for n in graph.nodes if n.name == "Slice_Mask").inputs[0]
    assert isinstance(folded, gs.Constant)
    assert folded.values.dtype == np.int64
    assert folded.values.shape == (1, 4) and (folded.values == 1).all()
    exported = gs.export_onnx(graph)
    assert not any(i.name == "attention_mask" for i in exported.graph.input)

    x = gs.Variable("x", np.float32, [1])
    y = gs.Variable("y", np.float32, [1])
    plain = gs.Graph(inputs=[x], outputs=[y],
                     nodes=[gs.Node(op="Identity", name="id", inputs=[x], outputs=[y])],
                     opset=18)
    assert LiquidModelExporter._fold_static_attention_mask(plain) == 0
    assert [t.name for t in plain.inputs] == ["x"]


def test_validate_rejects_state_count_mismatch(tmp_path):
    paths = {b: tmp_path / f"decoder_{b}.onnx" for b in (1, 4)}
    for b, path in paths.items():
        onnx.save(_family_graph(b), path)
    contract = FamilyContract(states=(
        State(input="past_key_values.0.key_value", output="present.0.key_value", seq_axis=2),
        State(input="past_key_values.1.key_value", output="present.1.key_value", seq_axis=2),
    ))
    with pytest.raises(ValueError, match="state count mismatch"):
        validate_native_onnx(paths, contract=contract)


def test_constant_role_override_controls_unclassified_constant(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    paths = {b: tmp_path / f"decoder_{b}.onnx" for b in (1, 4)}
    for b, path in paths.items():
        model = _family_graph(b)
        model.graph.initializer.append(numpy_helper.from_array(np.ones((1, 1, 4), np.float32), "fill"))
        model.graph.node.append(h.make_node("Concat", ["fill", "projected"], ["fill_out"], axis=1))
        onnx.save(model, path)
    contract = _contract(paths[1], read_only=False)
    with pytest.raises(ValueError, match="unclassified native float constant"):
        import_native_family(paths, tmp_path / "family.mlirbc", contract)
    overridden = FamilyContract(states=contract.states, constant_roles={"fill": "control"})
    metadata = import_native_family(paths, tmp_path / "family2.mlirbc", overridden)
    assert set(metadata["entries"]) == {"main", "main_4"}


def test_constant_roles_reject_invalid_overrides():
    with pytest.raises(ValueError, match="invalid constant role"):
        FamilyContract(states=(), constant_roles={"fill": "weight"})


def test_renamed_state_contract_imports_and_validates(tmp_path):
    pytest.importorskip("iree.compiler.extras.onnx_importer")
    paths = {b: tmp_path / f"decoder_{b}.onnx" for b in (1, 4)}
    for b, path in paths.items():
        model = _family_graph(b)
        model.graph.input[2].name = "cache.0"
        model.graph.output[1].name = "cache_out.0"
        model.graph.node[-1].input[0] = "cache.0"
        model.graph.node[-1].output[0] = "cache_out.0"
        onnx.save(model, path)
    contract = FamilyContract(states=(State(input="cache.0", output="cache_out.0", seq_axis=2),))
    metadata = import_native_family(paths, tmp_path / "family.mlirbc", contract)
    assert metadata["kv_cache_sequence_axis"] == 2
    assert metadata["entries"]["main_4"]["inputs"][2]["name"] == "cache.0"
    validate_native_onnx(paths, contract=contract)


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
    exporter._batch_prefill_sizes = (4,)
    exporter._weight_quantization = "int4"
    exporter._gptq_checkpoint = None
    monkeypatch.setattr(exporter, "_summarize_weight_quantization", lambda *args: dict(
        kl=0., cosine=1., max_abs_error=0., classification="test"), raising=False)
    exporter.export_onnx(validate=True, cleanup=False)
    monkeypatch.setattr(exporter, "_quantize_components",
                        Gemma3ModelExporter._quantize_components.__get__(exporter), raising=False)
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


@pytest.mark.parametrize("family,read_only", [("gemma3", False), ("gemma3-int4", False),
                                             ("liquid", False), ("liquid", True), ("gemma3", True), ("gemma3-int4", True)])
@pytest.mark.parametrize("mode", ["ordinary", "explicit", "automatic"])
def test_export_native_size_selection(tmp_path, monkeypatch, family, mode, read_only):
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
    graph = _read_only_family_graph if read_only else _family_graph
    options = dict(models_dir=tmp_path / "models", onnx_source_dir=source, max_gen_tokens=16,
                   extract_embeddings=True, split_lm_head=True, read_only_kv_cache=read_only,
                   batch_prefill=mode != "ordinary",
                   batch_prefill_sizes=(4, 7) if mode == "explicit" else None)
    if family.startswith("gemma3"):
        monkeypatch.setattr("torq.models.gemma3.export.AutoConfig.from_pretrained",
                            lambda *args, **kwargs: SimpleNamespace(**config))
        exporter = Gemma3ModelExporter(**options, weight_quantization="int4" if family == "gemma3-int4" else None)
    else:
        exporter = LiquidModelExporter("230m", **options)
    monkeypatch.setattr("torq.model_export.prefill.check_batch_prefill_compiler", lambda *a: None)
    cls = type(exporter)
    monkeypatch.setattr(cls, "_model_compile_flags", lambda self, **kwargs: ["--torq-prefetch-algorithm=constant-loads"])
    monkeypatch.setattr(cls, "_load_onnx", lambda self: {"model": graph(1, width=32)})
    def make_chunks(self, source_model=None):
        return {f"model_prefill_{size}": graph(size, width=32)
                for size in self._batch_prefill_sizes}
    monkeypatch.setattr(cls, "_make_native_prefill", make_chunks)
    def make_static(self):
        self._components = {"model": graph(1, width=32)}
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
            assert exporter._batch_prefill_sizes == ()
        exporter.weight_quantize_models(validate=True, cleanup=False, compiler_path=sys.executable,
                                        torq_compile_args=["--torq-max-nss-programs-size=33554432"])
    if mode == "automatic":
        final = onnx.load(exporter._export_paths["model"])
        assert probed_weights == [[(i.name, i.data_type, tuple(i.dims), i.raw_data)
                                   for i in final.graph.initializer]]
    expected = (4, 7, 16) if mode == "automatic" else (4, 7) if mode == "explicit" else ()
    assert exporter._batch_prefill_sizes == expected
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
    exporter._batch_prefill_enabled = True
    exporter._max_gen_tokens = 16
    monkeypatch.setattr(exporter, "_model_compile_flags", lambda **kwargs: [])
    monkeypatch.setattr("torq.model_export.prefill.check_batch_prefill_compiler", lambda *a: None)
    def fail(path, *args, **kwargs):
        assert path == exporter._export_paths["model"]
        onnx.checker.check_model(str(path))
        raise prefill_sizes.ProbeError("compile failed")
    monkeypatch.setattr(prefill_sizes, "recommend", fail)
    with pytest.raises(RuntimeError, match="--batch-prefill-sizes.*compile failed"):
        exporter.export_onnx(validate=False)
    assert exporter._batch_prefill_sizes == ()
    assert set(exporter._export_paths) == {"model"}
    onnx.checker.check_model(str(exporter._export_paths["model"]))
    assert exporter.load_calls == 1


@pytest.mark.parametrize("enabled,sizes,static,error", [
    (False, (4, 7), True, "requires --batch-prefill"),
    (True, None, False, "requires static exports"),
    ((4, 7), None, True, "must be a boolean"),
])
def test_native_mode_validation_before_source_loading(enabled, sizes, static, error):
    from torq.models.gemma3.export import Gemma3ModelExporter
    from torq.models.liquid.export import LiquidModelExporter

    for cls in (Gemma3ModelExporter, LiquidModelExporter):
        with pytest.raises(ValueError, match=error):
            cls(batch_prefill=enabled, batch_prefill_sizes=sizes, static_models=static,
                extract_embeddings=True, split_lm_head=True, read_only_kv_cache=False)


def test_automatic_compiler_check_preserves_previous_export(tmp_path, monkeypatch):
    from support.model_export import StubExporter

    exporter = StubExporter(tmp_path, {"model": _family_graph(1)})
    exporter._batch_prefill_enabled = True
    previous_dir = tmp_path / "export"
    previous_dir.mkdir(parents=True)
    previous = previous_dir / "previous.onnx"
    previous.write_bytes(b"previous export")
    def reject(*args):
        raise RuntimeError("compiler unavailable")
    monkeypatch.setattr("torq.model_export.prefill.check_batch_prefill_compiler", reject)
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        exporter.export_onnx()
    assert previous.read_bytes() == b"previous export"


# --- Qwen onboarding ---------------------------------------------------------

@pytest.mark.parametrize("size", [4, 7])
def test_qwen_native_rotary_positions_match_sequential_tokens(size):
    import onnx_graphsurgeon as gs
    import onnxruntime as ort
    from torq.models.qwen._graph import QwenOnnxGraphEditor

    heads, width = 2, 4

    def rotary_graph(query_shape):
        freq_raw = numpy_helper.from_array(np.random.default_rng(1).normal(size=(1, width // 2, 1)).astype(np.float32), "freq_raw")
        freq_shape = numpy_helper.from_array(np.array([1, width // 2, 1], np.int64), "freq_shape")
        axes = numpy_helper.from_array(np.array([1], np.int64), "axis")
        return h.make_model(h.make_graph([
            h.make_node("Unsqueeze", ["position_ids", "axis"], ["position_3d"]),
            h.make_node("Cast", ["position_3d"], ["position"], to=T.FLOAT),
            h.make_node("Expand", ["freq_raw", "freq_shape"], ["freq_expanded"]),
            h.make_node("Cast", ["freq_expanded"], ["freq"], to=T.FLOAT),
            h.make_node("MatMul", ["freq", "position"], ["angles"], name="rotary_emb/MatMul"),
            h.make_node("Transpose", ["angles"], ["angles_t"], perm=[0, 2, 1]),
            h.make_node("Concat", ["angles_t", "angles_t"], ["both"], axis=-1),
            h.make_node("Cos", ["both"], ["cos"]),
            h.make_node("Unsqueeze", ["cos", "axis"], ["cos_b"]),
            h.make_node("Mul", ["q", "cos_b"], ["rot"]),
        ], "/model", [
            h.make_tensor_value_info("q", T.FLOAT, query_shape),
            h.make_tensor_value_info("position_ids", T.INT64, [1, 1]),
        ], [h.make_tensor_value_info("rot", T.FLOAT, query_shape)], [freq_raw, freq_shape, axes]))

    rng = np.random.default_rng(3)
    reference = ort.InferenceSession(rotary_graph([1, heads, 1, width]).SerializeToString())
    chunk_model = rotary_graph([1, heads, size, width])
    editor = QwenOnnxGraphEditor(gs.import_onnx(chunk_model))
    assert editor.fix_native_rotary_positions(size)
    assert "Add" in {node.op for node in gs.import_onnx(editor.to_onnx()).nodes}
    chunk = ort.InferenceSession(editor.to_onnx().SerializeToString())

    query = rng.normal(size=(1, heads, size, width)).astype(np.float32)
    for start in (0, 5):
        expected = np.concatenate(
            [reference.run(["rot"], {"q": query[:, :, offset:offset + 1],
                                     "position_ids": np.array([[start + offset]], dtype=np.int64)})[0]
             for offset in range(size)],
            axis=2,
        )
        actual = chunk.run(["rot"], {"q": query, "position_ids": np.array([[start]], dtype=np.int64)})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    decode = QwenOnnxGraphEditor(gs.import_onnx(rotary_graph([1, heads, 1, width])))
    assert not decode.fix_native_rotary_positions(1)


def test_qwen_native_rotary_positions_is_a_noop_without_the_qwen3_subgraph():
    import onnx_graphsurgeon as gs
    from torq.models.qwen._graph import QwenOnnxGraphEditor

    model = h.make_model(h.make_graph([
        h.make_node("Identity", ["x"], ["y"]),
    ], "main", [h.make_tensor_value_info("x", T.FLOAT, [1, 1])], [h.make_tensor_value_info("y", T.FLOAT, [1, 1])]))
    assert not QwenOnnxGraphEditor(gs.import_onnx(model)).fix_native_rotary_positions(4)


@pytest.mark.parametrize("size", [4, 7])
def test_qwen_legacy_rotary_one_hot_matches_sequential_gathers(size):
    import onnx_graphsurgeon as gs
    import onnxruntime as ort
    from torq.models.qwen._graph import QwenOnnxGraphEditor

    positions, width = 32, 4
    tables = [
        numpy_helper.from_array(np.random.default_rng(seed).normal(size=(positions, width)).astype(np.float32),
                                f"rotary_emb_cache_{index}")
        for index, seed in enumerate((11, 12))
    ]

    def legacy_graph():
        nodes = [
            h.make_node("Gather", ["rotary_emb_cache_0", "position_ids"], ["selected_0"], axis=0),
            h.make_node("Gather", ["rotary_emb_cache_1", "position_ids"], ["selected_1"], axis=0),
            h.make_node("Identity", ["selected_0"], ["out_0"]),
            h.make_node("Identity", ["selected_1"], ["out_1"]),
        ]
        return h.make_model(h.make_graph(
            nodes,
            "main",
            [h.make_tensor_value_info("position_ids", T.INT64, [1, 1])],
            [h.make_tensor_value_info(name, T.FLOAT, [1, 1, width]) for name in ("out_0", "out_1")],
            tables,
        ))

    reference = ort.InferenceSession(legacy_graph().SerializeToString())
    editor = QwenOnnxGraphEditor(gs.import_onnx(legacy_graph()))
    assert editor.rotary_gather_to_onehot_matmul(size) == 2
    chunk = ort.InferenceSession(editor.to_onnx().SerializeToString())

    for start in (0, 7):
        expected = [
            np.concatenate(
                [reference.run(None, {"position_ids": np.array([[start + offset]], dtype=np.int64)})[index]
                 for offset in range(size)],
                axis=1,
            )
            for index in range(2)
        ]
        actual = chunk.run(None, {"position_ids": np.array([[start]], dtype=np.int64)})
        for result, want in zip(actual, expected):
            np.testing.assert_allclose(result, want, rtol=1e-6, atol=1e-6)

    # Decode keeps the existing one-hot path unchanged.
    decode = QwenOnnxGraphEditor(gs.import_onnx(legacy_graph()))
    assert decode.rotary_gather_to_onehot_matmul() == 2
    single = ort.InferenceSession(decode.to_onnx().SerializeToString())
    for start in (0, 7):
        np.testing.assert_allclose(
            single.run(None, {"position_ids": np.array([[start]], dtype=np.int64)}),
            reference.run(None, {"position_ids": np.array([[start]], dtype=np.int64)}),
            rtol=1e-6, atol=1e-6,
        )


@pytest.mark.parametrize("size", [1, 4])
def test_qwen_take_last_token_hidden_slices_the_chunk_row(size):
    import onnx_graphsurgeon as gs
    from torq.models.qwen._graph import QwenOnnxGraphEditor

    model = h.make_model(h.make_graph([
        h.make_node("Identity", ["x"], ["last_hidden_states"]),
    ], "main", [h.make_tensor_value_info("x", T.FLOAT, [1, size, 4])],
               [h.make_tensor_value_info("last_hidden_states", T.FLOAT, [1, size, 4])]))
    editor = QwenOnnxGraphEditor(gs.import_onnx(model))
    assert editor.take_last_token_hidden(size) is (size > 1)
    import onnxruntime as ort
    if size > 1:
        session = ort.InferenceSession(editor.to_onnx().SerializeToString())
        values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        np.testing.assert_array_equal(session.run(None, {"x": values})[0], values[:, -1:])


@pytest.mark.parametrize("quantization,convert,split_model,individual,read_only,expected", [
    ("none", False, False, False, True, (4, 12)),
    ("int4", True, False, False, True, (4, 12)),
    ("none", True, False, False, True, ValueError),
    ("int8", True, False, False, True, ValueError),
    ("none", False, True, False, True, ValueError),
    ("none", False, False, True, False, ValueError),
    ("none", False, False, False, False, ValueError),
])
def test_qwen_native_gates_supported_configs(quantization, convert, split_model, individual, read_only, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from torq.models.qwen import export as qwen_export
    from torq.models.qwen.export import QwenModelExporter

    config = SimpleNamespace(hidden_size=1024, vocab_size=16384, num_attention_heads=16, num_key_value_heads=8)
    with patch.object(qwen_export.AutoConfig, "from_pretrained", return_value=config):
        options = dict(
            static_models=True,
            max_gen_tokens=256,
            convert_dtypes=convert,
            keep_individual_kv_io=individual,
            split_lm_head=True,
            split_model=split_model,
            read_only_kv_cache=read_only,
            weight_quantization=None if quantization == "none" else quantization,
            batch_prefill=True,
            batch_prefill_sizes=(4, 12),
        )
        if expected is ValueError:
            with pytest.raises(ValueError, match="--batch-prefill requires"):
                QwenModelExporter(**options)
        else:
            exporter = QwenModelExporter(**options)
            assert exporter._batch_prefill_sizes == expected
            # Embedding extraction is implicit in --split-lm-head for Qwen.
            assert exporter._extract_embeddings is True


def test_qwen_native_contract_declares_undecidable_constants():
    from torq.models.qwen.export import QwenModelExporter

    # Rotary MatMul fed by Cast(Cast(Expand(table))) on the frequency side and
    # Cast(Unsqueeze(position_ids)) on the position side, like both Qwen
    # sources; plus a Qwen2.5-style projection bias Add.
    model = h.make_model(h.make_graph([
        h.make_node("Unsqueeze", ["position_ids", "axes"], ["position_3d"]),
        h.make_node("Cast", ["position_3d"], ["position"], to=T.FLOAT),
        h.make_node("Expand", ["freq_table", "freq_shape"], ["freq_expanded"]),
        h.make_node("Cast", ["freq_expanded"], ["freq_cast"], to=T.FLOAT),
        h.make_node("MatMul", ["freq_cast", "position"], ["angles"], name="rotary_emb/MatMul"),
        h.make_node("MatMul", ["x", "weight"], ["projected"]),
        h.make_node("Add", ["bias", "projected"], ["biased"]),
    ], "main", [
        h.make_tensor_value_info("position_ids", T.INT64, [1, 1]),
        h.make_tensor_value_info("x", T.FLOAT, [1, 1, 4]),
    ], [], [
        numpy_helper.from_array(np.ones((1, 4, 1), np.float32), "freq_table"),
        numpy_helper.from_array(np.array([1, 4, 1], np.int64), "freq_shape"),
        numpy_helper.from_array(np.array([1], np.int64), "axes"),
        numpy_helper.from_array(np.ones((4,), np.float32), "weight"),
        numpy_helper.from_array(np.ones((4,), np.float32), "bias"),
    ]))

    assert QwenModelExporter._native_contract_roles(model) == {
        "freq_table": "frequency", "bias": "parameter",
    }

    # A table feeding the MatMul directly is the classifier's own frequency
    # rule's job, and a scalar Add is its control rule's: no declarations.
    direct = h.make_model(h.make_graph([
        h.make_node("MatMul", ["freq_direct", "position"], ["angles"], name="rotary_emb/MatMul"),
        h.make_node("MatMul", ["x", "weight"], ["projected"]),
        h.make_node("Add", ["scale", "projected"], ["scaled"]),
    ], "main", [], [], [
        numpy_helper.from_array(np.ones((1, 4, 1), np.float32), "freq_direct"),
        numpy_helper.from_array(np.ones((4,), np.float32), "weight"),
        numpy_helper.from_array(np.array(0.5, np.float32), "scale"),
    ]))
    assert QwenModelExporter._native_contract_roles(direct) == {}


def test_qwen_native_int4_auto_sizes_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch
    from torq.models.qwen import export as qwen_export
    from torq.models.qwen.export import QwenModelExporter

    config = SimpleNamespace(hidden_size=1024, vocab_size=16384, num_attention_heads=16, num_key_value_heads=8)
    with patch.object(qwen_export.AutoConfig, "from_pretrained", return_value=config):
        with pytest.raises(ValueError, match="requires explicit --batch-prefill-sizes"):
            QwenModelExporter(
                static_models=True,
                max_gen_tokens=256,
                split_lm_head=True,
                read_only_kv_cache=True,
                weight_quantization="int4",
                batch_prefill=True,
                batch_prefill_sizes=None,
            )


def test_qwen_native_quantized_chunk_names_must_match_decode():
    from types import SimpleNamespace
    from torq.models.qwen.export import QwenModelExporter

    def matmuls(*names):
        return [(SimpleNamespace(name=name), name + ".w", None) for name in names]

    exporter = QwenModelExporter.__new__(QwenModelExporter)
    exporter._check_native_quantized_component("model.onnx", matmuls("a", "b"))
    exporter._check_native_quantized_component("model_prefill_4.onnx", matmuls("a", "b"))
    with pytest.raises(ValueError, match="different MatMul set"):
        exporter._check_native_quantized_component("model_prefill_4.onnx", matmuls("a", "renamed_b"))


@pytest.mark.parametrize("sizes,weight,convert_expected", [
    ((4, 12), "none", False),
    ((4, 12), "int4", True),
    (None, "none", False),
])
def test_qwen_native_cli_forwards_options_and_compiler_check(tmp_path, monkeypatch, sizes, weight, convert_expected):
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    from torq.models.qwen import export as qwen

    monkeypatch.setattr(qwen.AutoConfig, "from_pretrained", Mock(return_value=SimpleNamespace(
        hidden_size=1024, vocab_size=16384, num_attention_heads=16, num_key_value_heads=8)))
    exported = []
    def record_export(self, **kwargs):
        exported.append((self, kwargs))
    monkeypatch.setattr(qwen.QwenModelExporter, "export_onnx", record_export)
    monkeypatch.setattr(qwen.QwenModelExporter, "export_torq", Mock())
    converted = []
    def record_convert(self, **kwargs):
        converted.append((self, kwargs))
    monkeypatch.setattr(qwen.QwenModelExporter, "convert_models", record_convert)

    parser = argparse.ArgumentParser()
    add_qwen_export_args(parser)
    from torq.model_export import prefill
    check_compiler = Mock()
    monkeypatch.setattr(prefill, "check_batch_prefill_compiler", check_compiler)

    arguments = [
        "--batch-prefill", "--compiler-path", "/selected/compiler",
        "--no-gptq", "--weight-quantization", weight,
    ]
    if sizes is not None:
        arguments += ["--batch-prefill-sizes", ",".join(map(str, sizes))]
    args = parser.parse_args(arguments)
    qwen.export_qwen_from_args(args)

    exporter, options = exported[0]
    assert exporter._batch_prefill_enabled is True
    assert exporter._batch_prefill_sizes == (sizes or ())
    assert exporter._read_only_kv_cache is True
    assert exporter._split_lm_head is True
    assert exporter._convert_dtypes is convert_expected
    assert options["compiler_path"] == Path("/selected/compiler")
    assert options["torq_compile_args"] == (args.compile_flags or [])
    assert len(converted) == int(convert_expected)
    # Compile runs always need the selected compiler: explicit sizes compile
    # the family, omitted sizes probe first.
    check_compiler.assert_called_once_with(Path("/selected/compiler"))


# --- SmolLM2 onboarding ------------------------------------------------------

@pytest.mark.parametrize("size", [4, 7])
def test_smollm2_native_rotary_positions_match_sequential_tokens(size):
    import onnx_graphsurgeon as gs
    import onnxruntime as ort
    from torq.models.smollm2._graph import SmolLM2OnnxGraphEditor

    heads, width = 3, 4  # D/2 = 2

    def rotary_graph(seq):
        freq = numpy_helper.from_array(np.random.default_rng(1).normal(size=(1, width // 2, 1)).astype(np.float32), "freq")
        axes = numpy_helper.from_array(np.array([1], np.int64), "axes")
        return h.make_model(h.make_graph([
            h.make_node("Unsqueeze", ["position_ids", "axes"], ["position_3d"]),
            h.make_node("Cast", ["position_3d"], ["position"], to=T.FLOAT),
            h.make_node("MatMul", ["freq", "position"], ["angles"], name="rotary_emb/MatMul"),
            h.make_node("Concat", ["angles", "angles"], ["angles_cat"], axis=1),
            h.make_node("Transpose", ["angles_cat"], ["angles_t"], perm=[0, 2, 1]),
            h.make_node("Cos", ["angles_t"], ["cos"]),
            h.make_node("Unsqueeze", ["cos", "axes"], ["cos_b"]),
            h.make_node("Mul", ["q", "cos_b"], ["rot"]),
        ], "/model", [
            h.make_tensor_value_info("q", T.FLOAT, [1, heads, seq, width]),
            h.make_tensor_value_info("position_ids", T.INT64, [1, 1]),
        ], [h.make_tensor_value_info("rot", T.FLOAT, [1, heads, seq, width])], [freq, axes]))

    rng = np.random.default_rng(3)
    reference = ort.InferenceSession(rotary_graph(1).SerializeToString())
    chunk_model = rotary_graph(size)
    # fix_io(chunk_len) sets the sequence_length dim of position_ids to B;
    # the edit must restore the ABI start position [1, 1].
    dims = chunk_model.graph.input[1].type.tensor_type.shape.dim
    dims[1].dim_value = size
    editor = SmolLM2OnnxGraphEditor(gs.import_onnx(chunk_model))
    assert editor.fix_native_rotary_positions(size)
    assert list(editor.graph.inputs[1].shape) == [1, 1]
    chunk = ort.InferenceSession(editor.to_onnx().SerializeToString())

    query = rng.normal(size=(1, heads, size, width)).astype(np.float32)
    for start in (0, 5):
        expected = np.concatenate(
            [reference.run(["rot"], {"q": query[:, :, offset:offset + 1],
                                     "position_ids": np.array([[start + offset]], dtype=np.int64)})[0]
             for offset in range(size)],
            axis=2,
        )
        actual = chunk.run(["rot"], {"q": query, "position_ids": np.array([[start]], dtype=np.int64)})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    decode = SmolLM2OnnxGraphEditor(gs.import_onnx(rotary_graph(1)))
    assert not decode.fix_native_rotary_positions(1)


def test_smollm2_native_gates_supported_configs():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from torq.models.smollm2 import export as smollm2_export
    from torq.models.smollm2.export import SmolLM2ModelExporter

    config = SimpleNamespace(hidden_size=576, vocab_size=49152, num_attention_heads=9, num_key_value_heads=3)
    def construct(**overrides):
        options = dict(
            static_models=True,
            max_gen_tokens=256,
            extract_embeddings=True,
            split_lm_head=True,
            read_only_kv_cache=True,
            batch_prefill=True,
            batch_prefill_sizes=(4, 12),
        )
        options.update(overrides)
        return SmolLM2ModelExporter(**options)

    with patch.object(smollm2_export, "_repo_id_for_size", return_value="repo"), \
         patch.object(smollm2_export.AutoConfig, "from_pretrained", return_value=config):
        # OD4: 135M and 360M in; 1.7B out.
        for model_size in ("135M", "360M"):
            exporter = construct(model_size=model_size)
            assert exporter._batch_prefill_sizes == (4, 12)
        for override in (dict(model_size="1.7B"), dict(dynamic_quantize=True),
                         dict(convert_dtypes=True), dict(split_weights=True),
                         dict(read_only_kv_cache=False)):
            with pytest.raises(ValueError, match="--batch-prefill requires"):
                construct(**override)
        # Sizes without the native flag fail before any source loading.
        with pytest.raises(ValueError, match="requires --batch-prefill"):
            construct(batch_prefill=False, batch_prefill_sizes=(4, 12))


def test_smollm2_split_lm_head_defaults_off():
    import argparse
    from torq.models.smollm2 import add_smollm2_export_args

    parser = argparse.ArgumentParser()
    add_smollm2_export_args(parser)
    defaults = parser.parse_args([])
    assert defaults.split_lm_head is False
    assert defaults.batch_prefill is None
    assert defaults.batch_prefill_sizes is None
    gate = parser.parse_args(["--batch-prefill", "--batch-prefill-sizes", "4,12", "--split-lm-head"])
    assert gate.split_lm_head is True and gate.batch_prefill_sizes == (4, 12)


def test_smollm2_static_runner_wires_the_split_lm_head(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock
    import torq.models.smollm2._inference as smollm2_inf

    config = {"num_hidden_layers": 2, "num_key_value_heads": 1, "hidden_size": 8,
              "num_attention_heads": 2, "bos_token_id": 1, "eos_token_id": 2,
              "pad_token_id": 3}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    model_runner = Mock()
    model_runner.model_path = tmp_path / "model.onnx"
    model_runner.input_shapes = {}
    head_runner = Mock()
    monkeypatch.setattr(smollm2_inf, "ORTInferenceRunner",
                        Mock(side_effect=[model_runner, head_runner]))

    class _FakeTokenizer:
        def encode(self, text, **kwargs):
            class _Encoded:
                ids = [1]
            return _Encoded()

        def decode(self, ids, **kwargs):
            return "<tok>"

    monkeypatch.setattr(smollm2_inf, "_load_tokenizer", Mock(return_value=_FakeTokenizer()))
    monkeypatch.setattr(smollm2_inf, "_hf_hub_download", Mock(return_value=str(config_path)))
    monkeypatch.setattr(smollm2_inf.SmolLM2Static, "_find_token_embeddings", Mock(return_value=None))

    split = smollm2_inf.SmolLM2Static.from_onnx(
        tmp_path / "model.onnx", 16, lm_head_path=tmp_path / "lm_head.onnx")
    assert split._lm_head is head_runner
    assert split._lm_head.infer.called is False  # wired, not invoked here

    smollm2_inf.ORTInferenceRunner.reset_mock()
    smollm2_inf.ORTInferenceRunner.side_effect = [model_runner]
    plain = smollm2_inf.SmolLM2Static.from_onnx(tmp_path / "model.onnx", 16)
    assert plain._lm_head is None
    assert smollm2_inf.ORTInferenceRunner.call_count == 1



def test_native_validation_tolerance_defaults_and_override():
    import inspect
    import torq.model_export.prefill as prefill
    from unittest.mock import Mock, patch
    from torq.model_export.onnx import OnnxModelExporterBase
    from torq.models.smollm2.export import SmolLM2ModelExporter

    # The shared default is unchanged (1e-4/1e-4).
    signature = inspect.signature(OnnxModelExporterBase.__init__)
    assert signature.parameters["native_validation_rtol"].default == 1e-4
    assert signature.parameters["native_validation_atol"].default == 1e-4

    calls = []
    def record_validate(paths, *, contract, rtol=1e-4, atol=1e-4):
        calls.append((rtol, atol))
    with patch.object(prefill, "validate_native_onnx", record_validate), \
         patch.object(prefill, "share_native_weights", lambda *a: None):
        stub = SmolLM2ModelExporter.__new__(SmolLM2ModelExporter)
        stub._batch_prefill_enabled = True
        stub._batch_prefill_sizes = (4,)
        stub._native_validation_rtol = 1e-3
        stub._native_validation_atol = 1e-4
        stub._export_dir = None
        stub._export_paths = {"model": "m"}
        stub._export_path_for_component = lambda key: f"{key}.onnx"
        stub._components = {}
        stub._opt_configs = {}
        stub._make_native_prefill = lambda: {"model_prefill_4": None}
        stub._export_components = lambda comps, **kw: stub._export_paths.update(
            {k: f"{k}.onnx" for k in comps})
        stub._native_family_paths = lambda paths=None: {1: "m", 4: "model_prefill_4.onnx"}
        stub._native_family_contract = lambda path=None: None
        stub._complete_native_prefill(validate=True)
        calls.clear()
        stub._settle_native_family(validate=True)
    assert calls == [(1e-3, 1e-4)]


# --- LFM2-VL text decoder (patch-25) ----------------------------------------

def _vl_config_dir(tmp_path):
    import json
    config = dict(hidden_size=8, vocab_size=16, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=4, conv_dim=8, conv_L_cache=3, num_hidden_layers=2,
                  layer_types=["conv", "full_attention"])
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_liquid_vl_native_gating_and_family_keying(tmp_path):
    from torq.models.liquid.export_vl import DECODER, LiquidVLModelExporter

    _vl_config_dir(tmp_path)

    def construct(**overrides):
        options = dict(onnx_source_dir=tmp_path, models_dir=tmp_path,
                       split_lm_head=True, batch_prefill=True,
                       batch_prefill_sizes=(4, 12))
        options.update(overrides)
        return LiquidVLModelExporter(**options)

    exporter = construct()
    assert exporter._batch_prefill_sizes == (4, 12)
    assert exporter._batch_prefill_enabled is True
    # The attribute stays False for the VL LUT flow; only the base validation
    # saw the parameter.
    assert exporter._extract_embeddings is False

    for override in (dict(compile_vision=True), dict(vision_res=256),
                     dict(image_decoder_parts=3), dict(keep_conv1d=True),
                     dict(simulate_bf16=True), dict(chunk_lm_head=True),
                     dict(dynamic_quantize=True), dict(convert_dtypes=True),
                     dict(split_lm_head=False), dict(keep_individual_kv_io=True),
                     dict(batch_prefill_sizes=None)):
        with pytest.raises(ValueError, match="--batch-prefill requires"):
            construct(**override)

    exporter._export_paths = {DECODER: "d", "model_prefill_4": "c4", "model_prefill_12": "c12"}
    assert exporter._native_family_paths() == {1: "d", 4: "c4", 12: "c12"}


def test_liquid_vl_native_contract_sources_the_decoder_member(tmp_path):
    import onnx as onnx_mod
    from torq.models.liquid.export_vl import DECODER, LiquidVLModelExporter

    _vl_config_dir(tmp_path)
    exporter = LiquidVLModelExporter(onnx_source_dir=tmp_path, models_dir=tmp_path,
                                     split_lm_head=True, batch_prefill=True,
                                     batch_prefill_sizes=(4,))
    decode_path = tmp_path / f"{DECODER}.onnx"
    onnx_mod.save(_family_graph(1), str(decode_path))
    exporter._export_paths = {DECODER: decode_path}
    contract = exporter._native_family_contract()
    assert len(contract.states) == 1
    assert contract.states[0].input == "past_key_values.0.key_value"
    assert contract.states[0].rows_only is False
    assert contract.states[0].seq_axis == 2
    # The inherited liquid 350M mask role is present but inert on this graph.
    assert "control" in set(contract.constant_roles.values())
    explicit = exporter._native_family_contract(path=decode_path)
    assert explicit.states == contract.states


def test_liquid_vl_cli_forwards_native_options(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import torq.models.liquid.export_vl as vl
    from torq.models.liquid import add_liquid_vl_export_args

    constructed = []
    def fake_ctor(**kwargs):
        constructed.append(kwargs)
        return Mock()
    monkeypatch.setattr(vl, "LiquidVLModelExporter", fake_ctor)
    from torq.model_export import prefill
    monkeypatch.setattr(prefill, "check_batch_prefill_compiler", Mock())

    parser = argparse.ArgumentParser()
    add_liquid_vl_export_args(parser)
    args = parser.parse_args(["--batch-prefill", "--batch-prefill-sizes", "4,12",
                              "--split-lm-head"])
    vl.export_liquid_vl_from_args(args)
    kwargs = constructed[0]
    assert kwargs["batch_prefill"] is True
    assert kwargs["batch_prefill_sizes"] == (4, 12)
    assert kwargs["split_lm_head"] is True
    constructed.clear()
    # Rule 1: explicit --batch-prefill with --skip-torq fails.
    args = parser.parse_args(["--batch-prefill", "--skip-torq"])
    with pytest.raises(ValueError, match="requires compilation"):
        vl.export_liquid_vl_from_args(args)
    # Rule 2: the default with --skip-torq resolves to decode-only.
    constructed.clear()
    args = parser.parse_args(["--skip-torq"])
    vl.export_liquid_vl_from_args(args)
    assert constructed[0]["batch_prefill"] is False
