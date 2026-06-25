#!/usr/bin/env python3
"""Deduplicate ONNX node names by appending _N suffixes to collisions."""
import argparse
import onnx
from collections import Counter


def dedup_node_names(model):
    seen = {}
    fixed = 0
    for node in model.graph.node:
        if node.name:
            if node.name in seen:
                seen[node.name] += 1
                node.name = f"{node.name}_{seen[node.name]}"
                fixed += 1
            else:
                seen[node.name] = 0
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    fixed = dedup_node_names(model)
    print(f"[dedup_node_names] fixed {fixed} duplicate node names")
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
