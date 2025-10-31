import os
import argparse
import logging
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import version_converter
from torq.utils.logging import add_logging_args, configure_logging

from ...utils.onnx import get_model_opset

logger = logging.getLogger("FP32-Converter")

_EXPORT_DTYPE_MAPPING: Final[dict[str, onnx.TensorProto.DataType]] = {
    "bf16": onnx.TensorProto.BFLOAT16,
    "fp16": onnx.TensorProto.FLOAT16
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
    if export_dtype == "fp16" and not use_modelopt:
        logger.warning("FP16 conversion is only available via TensorRT modelopt")
        use_modelopt = True

    if use_modelopt:
        try:
            from modelopt.onnx import autocast
        except ImportError:
            logger.warning("Cannot import TensorRT modelopt, falling back to manual conversion")
            use_modelopt = False
            if export_dtype == "fp16":
                raise RuntimeError("No available path for converting FP32 to FP16, please install TensorRT modelopt")

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


class FP32Converter:

    def __init__(
        self,
        export_dtype: str,
        convert_io: bool = False,
        direct_cast: bool = True
    ):
        self._export_dtype = export_dtype
        self._export_onnx_dtype = self._validate_export_dtype(export_dtype)
        self._convert_io = convert_io
        self._direct_cast = direct_cast

        self._graph: gs.Graph | None = None
        self._has_subgraphs: bool = False

    @staticmethod
    def _validate_export_dtype(export_dtype: str) -> onnx.TensorProto.DataType:
        if not (export_onnx_dtype := _EXPORT_DTYPE_MAPPING.get(export_dtype)):
            raise ValueError(f"Invalid export dtype '{export_dtype}', select from {_EXPORT_DTYPE_MAPPING.keys()}")
        return export_onnx_dtype

    @property
    def graph(self) -> gs.Graph:
        if not self._graph:
            raise ValueError("No loaded graph")
        return self._graph

    @graph.setter
    def graph(self, graph: gs.Graph):
        if not isinstance(graph, gs.Graph):
            raise TypeError(f"Invalid graph type {type(graph)}")
        self._graph = graph

    def _convert_tensor(
        self,
        tensor: gs.Variable | gs.Constant,
        node: gs.Node,
        idx: int | str,
        is_attr: bool = False
    ):
        if tensor.dtype != np.float32:
            logger.debug("Skipping non-fp32 tensor '%s'", tensor.name)
            return

        if isinstance(tensor, gs.Variable):
            tensor.dtype = self._export_onnx_dtype
            logger.debug("Set dtype of tensor '%s' to %s", tensor.name, self._export_dtype)
        elif isinstance(tensor, gs.Constant):
            if not (const_f16 := self.graph.tensors().get(tensor.name + f"_{self._export_dtype}")):
                const_f16 = gs.Constant(
                    tensor.name + f"_{self._export_dtype}",
                    tensor.values,
                    export_dtype=self._export_onnx_dtype
                )
            logger.debug("Add %s initialzier '%s'", self._export_dtype, const_f16.name)
            if is_attr:
                assert isinstance(idx, str), "Index must be a string for attribute update"
                node.attrs[idx] = const_f16
                logger.debug("Set attr '%s' of node '%s' to '%s'", str(idx), node.name, const_f16.name)
            else:
                assert isinstance(idx, int), "Index must be an integer for input update"
                node.inputs[idx] = const_f16
                logger.debug("Set input %d of node '%s' to '%s'", int(idx), node.name, const_f16.name)
        else:
            logger.warning("Skipping conversion due to invalid tensor type '%s'", str(type(tensor)))

    def _convert_graph(
        self,
        graph: gs.Graph,
        is_subgraph: bool
    ):
        for node in list(graph.nodes):
            for attr, attr_val in node.attrs.items():
                if isinstance(attr_val, gs.Graph):
                    self._has_subgraphs = True
                    logger.info(
                        "Converting subgraph from attr '%s' of %s node '%s'",
                        str(attr),
                        node.op,
                        node.name
                    )
                    self._convert_graph(attr_val, is_subgraph=True)

        self.graph = graph
        skip_names = {i.name for i in self.graph.inputs} if (not is_subgraph and not self._convert_io) else set()
        for node in list(self.graph.nodes):
            if node.op == "Cast":
                if (cast_out := node.outputs[0]).dtype == np.float32:
                    if self._direct_cast:
                        cast_out.dtype = self._export_onnx_dtype
                        node.attrs["to"] = self._export_onnx_dtype
                        logger.debug("Update Cast op '%s' to directly cast to %s", node.name, self._export_dtype)
                    else:
                        consumers: list[gs.Node] = list(cast_out.outputs)
                        out_f16: gs.Variable = self.graph.layer(
                            name=cast_out.name + f"_cast_fp32_to_{self._export_dtype}",
                            op="Cast",
                            inputs=[cast_out],
                            outputs=[gs.Variable(cast_out.name + f"_{self._export_dtype}", dtype=self._export_onnx_dtype, shape=cast_out.shape)],
                            attrs={"to": self._export_onnx_dtype},
                        )[0]
                        for consumer in consumers:
                            for i, inp in enumerate(consumer.inputs):
                                if inp is cast_out:
                                    consumer.inputs[i] = out_f16
                                    logger.debug("Add fp32 -> %s Cast node to feed '%s'", self._export_dtype, consumer.name)
                continue

            if node.op == "DynamicQuantizeLinear":
                inp_f32: gs.Variable = graph.layer(
                    name=node.name + "_inp_cast_f32",
                    op="Cast",
                    inputs=[node.inputs[0]],
                    outputs=[gs.Variable(node.name + "_inp_f32", dtype=np.float32, shape=node.inputs[0].shape)],
                    attrs={"to": onnx.TensorProto.FLOAT},
                )[0]
                node.inputs[0] = inp_f32
                logger.debug("Add %s -> fp32 Cast node to input of DQL node '%s'", self._export_dtype, node.name)

                scale_out: gs.Variable = node.outputs[1]
                consumers: list[gs.Node] = list(scale_out.outputs)
                scale_out_f16: gs.Variable = graph.layer(
                    name=node.name + f"_scale_cast_{self._export_dtype}",
                    op="Cast",
                    inputs=[scale_out],
                    outputs=[gs.Variable(node.name + f"_scale_{self._export_dtype}", dtype=self._export_onnx_dtype, shape=scale_out.shape)],
                    attrs={"to": self._export_onnx_dtype},
                )[0]
                for consumer in consumers:
                    for i, inp in enumerate(consumer.inputs):
                        if inp is scale_out:
                            consumer.inputs[i] = scale_out_f16
                            logger.debug("Add fp32 -> %s Cast node to feed '%s'", self._export_dtype, consumer.name)
                continue

            if node.op == "Constant" and (val := node.attrs.get("value")) is not None:
                self._convert_tensor(val, node, "value", is_attr=True)

            if node.op == "ConstantOfShape" and (val := node.attrs.get("value")) is not None:
                self._convert_tensor(val, node, "value", is_attr=True)

            if node.op == "Resize":
                self._convert_tensor(node.inputs[0], node, 0)
                if (out := node.outputs[0]).dtype == np.float32:
                    out.dtype = self._export_onnx_dtype
                continue

            for i, inp in enumerate(list(node.inputs)):
                if inp.name in skip_names:
                    logger.debug("Skipping dtype conversion of model input '%s'", inp.name)
                    continue
                self._convert_tensor(inp, node, i)

            for out in node.outputs:
                assert isinstance(out, gs.Variable), f"Non gs.Variable output '{out.name}' ({type(out)}) for node '{node.name}'"
                if out.dtype == np.float32:
                    out.dtype = self._export_onnx_dtype
        logger.info("Updated graph dtypes to %s", self._export_dtype)

    def _update_inputs(self):
        for graph_inp in list(self.graph.inputs):
            if graph_inp.dtype != np.float32:
                logger.debug("Skipping non-fp32 input '%s'", graph_inp.name)
                continue
            if self._convert_io:
                graph_inp.dtype = self._export_onnx_dtype
                logger.debug("Set dtype to %s for input '%s'", self._export_dtype, graph_inp.name)
            else:
                # add cast nodes at model inputs
                consumers = list(graph_inp.outputs)
                if not consumers:
                    continue
                inp_f16 = self.graph.layer(
                    name=graph_inp.name + f"_to_{self._export_dtype}",
                    op="Cast",
                    inputs=[graph_inp],
                    outputs=[gs.Variable(graph_inp.name + f"_{self._export_dtype}", dtype=self._export_onnx_dtype, shape=graph_inp.shape)],
                    attrs={"to": self._export_onnx_dtype}
                )[0]
                for node in consumers:
                    for i, val in enumerate(node.inputs):
                        if val is graph_inp:
                            node.inputs[i] = inp_f16
                            logger.debug("Update node '%s' to accept %s input '%s'", node.name, self._export_dtype, inp_f16.name)

    def _update_outputs(self):
        for i, graph_out in enumerate(list(self.graph.outputs)):
            if graph_out.dtype != self._export_onnx_dtype:
                logger.debug("Skipping non-%s output '%s'", self._export_dtype, graph_out.name)
                continue
            if self._convert_io:
                graph_out.dtype = self._export_onnx_dtype
                logger.debug("Set dtype to %s for output '%s'", self._export_dtype, graph_out.name)
            else:
                # add cast nodes at model outputs
                out_name = graph_out.name
                graph_out.name = graph_out.name + f"_{self._export_dtype}"
                out_fp32 = self.graph.layer(
                    name=graph_out.name + "_to_fp32",
                    op="Cast",
                    inputs=[graph_out],
                    outputs=[gs.Variable(out_name, dtype=onnx.TensorProto.FLOAT, shape=graph_out.shape)],
                    attrs={"to": onnx.TensorProto.FLOAT}
                )[0]
                self.graph.outputs[i] = out_fp32
                logger.debug("Add fp32 cast node for output '%s'", out_name)

    def convert_model(
        self,
        input_model: onnx.ModelProto
    ) -> onnx.ModelProto:
        self._convert_graph(gs.import_onnx(input_model), is_subgraph=False)
        self._update_inputs()
        self._update_outputs()

        self.graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True
        ).toposort()
        new_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(self.graph), check_type=True, strict_mode=True, data_prop=not self._has_subgraphs
        )
        new_model.ir_version = input_model.ir_version
        onnx.checker.check_model(new_model, full_check=True)
        return new_model
    

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
