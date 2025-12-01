# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import version_converter
from torq.utils.logging import add_logging_args, configure_logging

from ...utils.onnx import get_model_opset, is_same_dtype

logger = logging.getLogger("FP32-Converter")

_ONNX_DTYPE_MAPPING: Final[dict[str, onnx.TensorProto.DataType]] = {
    "fp32": onnx.TensorProto.FLOAT,
    "bf16": onnx.TensorProto.BFLOAT16,
    "fp16": onnx.TensorProto.FLOAT16,
    "int32": onnx.TensorProto.INT32,
    "uint32": onnx.TensorProto.UINT32,
    "int16": onnx.TensorProto.INT16,
    "uint16": onnx.TensorProto.UINT16,
    "int8": onnx.TensorProto.INT8,
    "uint8": onnx.TensorProto.UINT8,
}


def upgrade_model(model: onnx.ModelProto, target_opset: int) -> onnx.ModelProto:
    if (curr_opset := get_model_opset(model)) >= target_opset:
        logger.info("Model already at opset %d >= %d, skipping upgrade", curr_opset, target_opset)
        return model
    upgraded = version_converter.convert_version(model, target_opset)
    logger.info("Upgraded model opset to %d", target_opset)
    return upgraded


def convert_model(
    input_model: str | os.PathLike,
    output_model: str | os.PathLike,
    export_dtype: str,
    use_modelopt: bool = False,
    convert_io: bool = False,
    max_float: float = 1e9,
    target_opset: int = 22,
):

    if use_modelopt:
        try:
            from modelopt.onnx import autocast
        except ImportError:
            logger.warning("Cannot import TensorRT modelopt, falling back to manual conversion")
            use_modelopt = False

    if use_modelopt:
        converted_model = autocast.convert_to_mixed_precision(
            str(input_model),
            export_dtype,
            data_max=max_float,
            init_max=max_float,
            keep_io_types=not convert_io
        )
    else:
        input_model = onnx.load(input_model)
        input_model = upgrade_model(input_model, target_opset)
        converted_model = FP32Converter(
            export_dtype,
            convert_io=convert_io
        ).convert_model(input_model)

    export_dir = Path(output_model).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(converted_model, output_model)
    logger.info("Saved converted model to '%s'", str(output_model))


class OnnxDtypeConverterBase(ABC):

    def __init__(
        self,
        original_dtype: str,
        export_dtype: str,
        convert_io: bool = False
    ):
        self._original_dtype_str = original_dtype
        self._export_dtype_str = export_dtype
        self._convert_io = convert_io

        self._original_onnx_dtype = self._validate_dtype(original_dtype)
        self._export_onnx_dtype = self._validate_dtype(export_dtype)
        self._convert_exceptions: dict[str, str] = {}

    @staticmethod
    def _validate_dtype(dtype: str) -> onnx.TensorProto.DataType:
        onnx_dtype = _ONNX_DTYPE_MAPPING.get(dtype)
        if onnx_dtype is None:
            raise ValueError(
                f"Invalid dtype '{dtype}', select from {list(_ONNX_DTYPE_MAPPING.keys())}"
            )
        return onnx_dtype

    @staticmethod
    def _collect_all_graphs(root: gs.Graph) -> list[gs.Graph]:
        graphs: list[gs.Graph] = []
        queue: list[gs.Graph] = [root]

        while queue:
            g = queue.pop()
            graphs.append(g)
            for node in g.nodes:
                for attr_val in node.attrs.values():
                    if isinstance(attr_val, gs.Graph):
                        queue.append(attr_val)
                    elif isinstance(attr_val, list):
                        for v in attr_val:
                            if isinstance(v, gs.Graph):
                                queue.append(v)

        return graphs

    @staticmethod
    def _add_cast_node(consumers: list[gs.Node] | None = None):
        consumers = consumers or []

    @abstractmethod
    def _convert_graph(
        self,
        graph: gs.Graph,
    ): ...

    def _convert_tensor(
        self,
        tensor: gs.Variable | gs.Constant,
        node: gs.Node,
        graph: gs.Graph,
        idx: int | str,
        is_attr: bool = False
    ):
        if not is_same_dtype(tensor.dtype, self._original_onnx_dtype):
            logger.debug("Skipping non-%s tensor '%s'", self._original_dtype_str, tensor.name)
            return
        if tensor.name in self._convert_exceptions:
            logger.debug("Skipping dtype conversion of explicitly marked tensor '%s' (%s)",
                         tensor.name, self._convert_exceptions[tensor.name])
            return

        if isinstance(tensor, gs.Variable):
            # special handling not needed for runtime tensors
            tensor.dtype = self._export_onnx_dtype
            logger.debug("Set dtype of tensor '%s' to %s", tensor.name, self._export_dtype_str)
        elif isinstance(tensor, gs.Constant):
            new_const_name: str = tensor.name + f"_{self._export_dtype_str}"
            if not (new_const := graph.tensors().get(new_const_name)):
                if self._original_onnx_dtype == onnx.TensorProto.FLOAT and self._export_onnx_dtype == onnx.TensorProto.BFLOAT16:
                    new_const = gs.Constant(
                        new_const_name,
                        tensor.values,
                        export_dtype=self._export_onnx_dtype
                    )
                else:
                    try:
                        np_type = onnx.helper.tensor_dtype_to_np_dtype(self._export_onnx_dtype)
                    except (TypeError, ValueError, KeyError):
                        raise RuntimeError(f"Unsupported tensor datatype {self._export_dtype_str}")
                    new_const = gs.Constant(
                        new_const_name,
                        tensor.values.astype(np_type)
                    )
            logger.debug("Add %s initializer '%s'", self._export_dtype_str, new_const.name)
            try:
                if is_attr:
                    assert isinstance(idx, str), "Node attribute index must be a string"
                    node.attrs[idx] = new_const
                    logger.debug("Set attr '%s' of node '%s' to '%s'", str(idx), node.name, new_const.name)
                else:
                    assert isinstance(idx, int), "Node input index must be an integer"
                    node.inputs[idx] = new_const
                    logger.debug("Set input %d of node '%s' to '%s'", int(idx), node.name, new_const.name)
            except (IndexError, ValueError, KeyError) as e:
                typ = "attribute" if is_attr else "input"
                logger.error("Failed to update %s %s ('%s') of node '%s'", typ, str(idx), tensor.name, node.name)
        else:
            logger.warning("Skipping conversion due to invalid tensor type '%s'", str(type(tensor)))

    def _update_inputs(self, graph: gs.Graph):
        for graph_inp in list(graph.inputs):
            if not is_same_dtype(graph_inp.dtype, self._original_onnx_dtype):
                logger.debug("Skipping non-%s input '%s'", self._original_dtype_str, graph_inp.name)
                continue
            if self._convert_io:
                graph_inp.dtype = self._export_onnx_dtype
                logger.debug("Set dtype to %s for input '%s'", self._export_dtype_str, graph_inp.name)
            else:
                # add cast nodes at model inputs
                consumers = list(graph_inp.outputs)
                if not consumers:
                    continue
                inp_new = graph.layer(
                    name=graph_inp.name + f"_to_{self._export_dtype_str}",
                    op="Cast",
                    inputs=[graph_inp],
                    outputs=[gs.Variable(graph_inp.name + f"_{self._export_dtype_str}", dtype=self._export_onnx_dtype, shape=graph_inp.shape)],
                    attrs={"to": self._export_onnx_dtype}
                )[0]
                for node in consumers:
                    for i, val in enumerate(node.inputs):
                        if val is graph_inp:
                            node.inputs[i] = inp_new
                            logger.debug("Update node '%s' to accept %s input '%s'", node.name, self._export_dtype_str, inp_new.name)
                self._convert_exceptions[graph_inp.name] = "graph input and convert_io=False"

    def _update_outputs(self, graph: gs.Graph):
        for i, graph_out in enumerate(list(graph.outputs)):
            if not is_same_dtype(graph_out.dtype, self._original_onnx_dtype):
                logger.debug("Skipping non-%s output '%s'", self._original_dtype_str, graph_out.name)
                continue
            graph_out.dtype = self._export_onnx_dtype
            logger.debug("Set dtype to %s for output '%s'", self._export_dtype_str, graph_out.name)
            if not self._convert_io:
                # add cast nodes at model outputs
                out_name = graph_out.name
                graph_out.name = graph_out.name + f"_{self._export_dtype_str}"
                out_new = graph.layer(
                    name=graph_out.name + f"_to_{self._original_dtype_str}",
                    op="Cast",
                    inputs=[graph_out],
                    outputs=[gs.Variable(out_name, dtype=self._original_onnx_dtype, shape=graph_out.shape)],
                    attrs={"to": self._original_onnx_dtype}
                )[0]
                graph.outputs[i] = out_new
                logger.debug("Add %s cast node for output '%s'", self._original_dtype_str, out_name)
                self._convert_exceptions[out_name] = "graph output and convert_io=False"

    def _check_tensor_dtypes(self, graph: gs.Graph) -> tuple[list[str], list[str]]:
        all_tensors: list[str] = []
        not_converted: list[str] = []
        for tensor_name, tensor in graph.tensors().items():
            tensor_dtype = getattr(tensor, "export_dtype", None) or tensor.dtype
            if is_same_dtype(tensor_dtype, self._original_onnx_dtype):
                if (exc_reason := self._convert_exceptions.get(tensor_name)) is None:
                    logger.warning(
                        "Graph '%s': tensor '%s' not converted to %s (unhandled)",
                        graph.name, tensor_name, self._export_dtype_str
                    )
                else:
                    logger.info(
                        "Graph '%s': tensor '%s' not converted to %s (%s)",
                        graph.name, tensor_name, self._export_dtype_str, exc_reason
                    )
                not_converted.append(tensor_name)
            all_tensors.append(tensor_name)
        return all_tensors, not_converted

    def _check_conversion(self, graph: gs.Graph) -> tuple[list[str], list[str]]:
        graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True
        ).toposort()
        return self._check_tensor_dtypes(graph)

    def convert_model(
        self,
        input_model: onnx.ModelProto
    ) -> onnx.ModelProto:
        root, *subgraphs = self._collect_all_graphs(gs.import_onnx(input_model))
        all_tensors: list[str] = []
        not_converted: list[str] = []

        # convert subgraphs (if any)
        for g in subgraphs:
            self._convert_graph(g)
            self._update_inputs(g)
            self._update_outputs(g)
            conv_info = self._check_conversion(g)
            all_tensors += conv_info[0]
            not_converted += conv_info[1]

        # convert root graph
        self._convert_graph(root)
        self._update_inputs(root)
        self._update_outputs(root)
        conv_info = self._check_conversion(root)
        all_tensors += conv_info[0]
        not_converted += conv_info[1]

        onnx.save_model(gs.export_onnx(root), "temp.onnx")

        new_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(root), check_type=True, strict_mode=True, data_prop=len(subgraphs) == 0
        )
        new_model.ir_version = input_model.ir_version
        onnx.checker.check_model(new_model, full_check=True)

        all_tensors = set(all_tensors)
        not_converted = set(not_converted)
        total = len(all_tensors)
        failed = len(not_converted)
        converted = total - failed

        if total == 0:
            logger.info(
                "Conversion results: graph has no %s tensors to convert to %s",
                self._original_dtype_str, self._export_dtype_str
            )
        else:
            pct = (converted / total) * 100.0
            logger.info(
                "Conversion results: converted %d/%d (%.2f%%) %s tensors to %s",
                converted, total, pct, self._original_dtype_str, self._export_dtype_str
            )

        return new_model


class FP32Converter(OnnxDtypeConverterBase):

    def __init__(
        self,
        export_dtype: str,
        convert_io: bool = False,
        direct_cast: bool = True
    ):
        super().__init__("fp32", export_dtype, convert_io)

        self._direct_cast = direct_cast

    @staticmethod
    def _is_fp32(tensor: gs.Variable | gs.Constant) -> bool:
        dtype = getattr(tensor, "export_dtype", None) or tensor.dtype
        return is_same_dtype(dtype, np.float32)

    def _convert_graph(
        self,
        graph: gs.Graph
    ):
        skip_names = {i.name for i in graph.inputs} | {o.name for o in graph.outputs}
        for node in list(graph.nodes):
            # special case: Cast -> properly handle casts to fp32
            if node.op == "Cast" and self._is_fp32(cast_out := node.outputs[0]):
                if cast_out.name in self._convert_exceptions:
                    logger.debug("Skipping dtype conversion of explicitly marked fp32 cast output '%s'", cast_out.name)
                    continue
                if self._direct_cast:
                    cast_out.dtype = self._export_onnx_dtype
                    node.attrs["to"] = self._export_onnx_dtype
                    logger.debug("Update Cast op '%s' to directly cast to %s", node.name, self._export_dtype_str)
                else:
                    self._convert_exceptions[cast_out.name] = "Cast output and direct_cast=False"
                    consumers: list[gs.Node] = list(cast_out.outputs)
                    out_f16: gs.Variable = graph.layer(
                        name=cast_out.name + f"_cast_fp32_to_{self._export_dtype_str}",
                        op="Cast",
                        inputs=[cast_out],
                        outputs=[gs.Variable(cast_out.name + f"_{self._export_dtype_str}", dtype=self._export_onnx_dtype, shape=cast_out.shape)],
                        attrs={"to": self._export_onnx_dtype},
                    )[0]
                    for consumer in consumers:
                        for i, inp in enumerate(consumer.inputs):
                            if inp is cast_out:
                                consumer.inputs[i] = out_f16
                                logger.debug("Add fp32 -> %s Cast node to feed '%s'", self._export_dtype_str, consumer.name)
                continue

            # special case: DQL -> input and scale output must be fp32
            if node.op == "DynamicQuantizeLinear":
                inp_f32: gs.Variable = graph.layer(
                    name=node.name + "_inp_cast_f32",
                    op="Cast",
                    inputs=[node.inputs[0]],
                    outputs=[gs.Variable(node.name + "_inp_f32", dtype=np.float32, shape=node.inputs[0].shape)],
                    attrs={"to": onnx.TensorProto.FLOAT},
                )[0]
                node.inputs[0] = inp_f32
                self._convert_exceptions[inp_f32.name] = "DynamicQuantizeLinear input"
                logger.debug("Add %s -> fp32 Cast node to input of DQL node '%s'", self._export_dtype_str, node.name)

                scale_out: gs.Variable = node.outputs[1]
                self._convert_exceptions[scale_out.name] = "DynamicQuantizeLinear output"
                consumers: list[gs.Node] = list(scale_out.outputs)
                scale_out_f16: gs.Variable = graph.layer(
                    name=node.name + f"_scale_cast_{self._export_dtype_str}",
                    op="Cast",
                    inputs=[scale_out],
                    outputs=[gs.Variable(node.name + f"_scale_{self._export_dtype_str}", dtype=self._export_onnx_dtype, shape=scale_out.shape)],
                    attrs={"to": self._export_onnx_dtype},
                )[0]
                for consumer in consumers:
                    for i, inp in enumerate(consumer.inputs):
                        if inp is scale_out:
                            consumer.inputs[i] = scale_out_f16
                            logger.debug("Add fp32 -> %s Cast node to feed '%s'", self._export_dtype_str, consumer.name)
                continue

            # special case: Constant -> constant value stored as an attribute
            if node.op == "Constant" and (val := node.attrs.get("value")) is not None:
                self._convert_tensor(val, node, graph, "value", is_attr=True)

            # special case: ConstantOfShape -> constant value stored as an attribute
            if node.op == "ConstantOfShape" and (val := node.attrs.get("value")) is not None:
                self._convert_tensor(val, node, graph, "value", is_attr=True)

            # special case: Resize -> only input and output can be cast to bf16
            if node.op == "Resize" and node.inputs[0].name not in skip_names:
                self._convert_tensor(node.inputs[0], node, graph, 0)
                if self._is_fp32(out := node.outputs[0]) and out.name not in skip_names:
                    out.dtype = self._export_onnx_dtype
                continue

            for i, inp in enumerate(list(node.inputs)):
                if inp.name in skip_names:
                    logger.debug("Skipping dtype conversion of model input '%s'", inp.name)
                    continue
                self._convert_tensor(inp, node, graph, i)

            for out in node.outputs:
                assert isinstance(out, gs.Variable), f"Non gs.Variable output '{out.name}' ({type(out)}) for node '{node.name}'"
                if out.name in skip_names:
                    logger.debug("Skipping dtype conversion of model output '%s'", out.name)
                    continue
                if self._is_fp32(out):
                    out.dtype = self._export_onnx_dtype
                    if node.op == "Cast":
                        node.attrs["to"] = self._export_onnx_dtype

        logger.info("Updated graph '%s' dtypes to %s", graph.name, self._export_dtype_str)
    

def add_onnx_fp32_convert_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input ONNX model path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output ONNX model path"
    )
    parser.add_argument(
        "-e", "--export-dtype",
        type=str,
        metavar="DTYPE",
        choices=["fp16", "bf16"],
        required=True,
        help="Export data type (choices: %(choices)s)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=22,
        help="ONNX opset to use; note that a relatively new opset is required for bf16 support in some ops (default: %(default)s)"
    )
    parser.add_argument(
        "--max-float",
        type=float,
        default=1e9,
        help="Maximum FP32 value in model initializers and data, tensors with data > `max_float` will be left in fp32"
    )
    parser.add_argument(
        "--convert-io",
        action="store_true",
        default=False,
        help="Convert model I/O to export dtype"
    )
    parser.add_argument(
        "--modelopt",
        action="store_true",
        default=False,
        help="Use TensorRT modelopt for dtype conversion"
    )
    add_logging_args(parser)


def onnx_fp32_convert_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    convert_model(
        args.input,
        args.output,
        args.export_dtype,
        args.modelopt,
        args.convert_io,
        args.max_float,
        args.opset
    )


def main():
    parser = argparse.ArgumentParser()
    add_onnx_fp32_convert_args(parser)
    onnx_fp32_convert_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
