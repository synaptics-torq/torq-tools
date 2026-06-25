#!/usr/bin/env python3
"""Merge ReduceMean(keepdims=0) + Unsqueeze back into ReduceMean(keepdims=1).

decompose_norm (torq.tools) converts ReduceMean(keepdims=1) to
ReduceMean(keepdims=0) + Unsqueeze. The golden model doesn't have this
conversion. This script reverses it to match golden's structure.

Usage:
    python3 scripts/merge_reducemean_unsqueeze.py -i model.onnx -o model_merged.onnx
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph
    nodes = list(graph.node)

    output_to_node = {}
    for n in nodes:
        for o in n.output:
            output_to_node[o] = n

    input_to_consumers = defaultdict(list)
    for n in nodes:
        for inp in n.input:
            input_to_consumers[inp].append(n)

    init_names = {i.name for i in graph.initializer}

    to_remove = set()
    merged = 0

    for unsq in nodes:
        if unsq.op_type != "Unsqueeze":
            continue
        if unsq.name in to_remove:
            continue

        unsq_input = unsq.input[0]
        rm_node = output_to_node.get(unsq_input)
        if rm_node is None or rm_node.op_type != "ReduceMean":
            continue

        keepdims = 1
        for attr in rm_node.attribute:
            if attr.name == "keepdims":
                keepdims = attr.i
                break
        if keepdims != 0:
            continue

        consumers = input_to_consumers.get(unsq_input, [])
        if len(consumers) != 1:
            continue

        axes_attr = None
        for attr in rm_node.attribute:
            if attr.name == "axes":
                axes_attr = list(attr.ints)
                break

        if axes_attr is None and len(rm_node.input) > 1:
            axes_input = rm_node.input[1]
            if axes_input in init_names:
                for init in graph.initializer:
                    if init.name == axes_input:
                        axes_attr = list(numpy_helper.to_array(init).flatten())
                        break

        unsq_axes = None
        if len(unsq.input) > 1:
            axes_name = unsq.input[1]
            if axes_name in init_names:
                for init in graph.initializer:
                    if init.name == axes_name:
                        unsq_axes = list(numpy_helper.to_array(init).flatten())
                        break

        if axes_attr is not None and unsq_axes is not None:
            if sorted(axes_attr) != sorted(unsq_axes):
                continue

        new_attrs = {}
        for attr in rm_node.attribute:
            if attr.name == "keepdims":
                continue
            if attr.name == "axes":
                new_attrs["axes"] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                new_attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                new_attrs[attr.name] = list(attr.ints)
        new_attrs["keepdims"] = 1

        new_rm = helper.make_node(
            "ReduceMean",
            inputs=rm_node.input,
            outputs=unsq.output,
            name=rm_node.name,
            **new_attrs
        )

        idx = nodes.index(rm_node)
        nodes[idx] = new_rm
        to_remove.add(unsq.name)
        merged += 1

    if merged > 0:
        nodes = [n for n in nodes if n.name not in to_remove]
        del graph.node[:]
        graph.node.extend(nodes)

    print(f"Merged {merged} ReduceMean+Unsqueeze pairs into ReduceMean(keepdims=1)")

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape inference: {e}")

    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
