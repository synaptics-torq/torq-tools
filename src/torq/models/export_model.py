# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .moonshine import add_moonshine_export_args
from .moonshine_streaming import add_moonshine_streaming_export_args
from .smollm2 import add_smollm2_export_args
from .gemma3 import add_gemma3_export_args
from .liquid import add_liquid_export_args, add_liquid_vl_export_args
from .yolo26 import add_yolo26_export_args, add_yolo26_compile_args


def main():
    parser = argparse.ArgumentParser(description="Export models to Torq")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Export moonshine to Torq")
    add_moonshine_export_args(moonshine)

    moonshine_streaming = model.add_parser("moonshine_streaming", help="Export Moonshine Streaming to Torq")
    add_moonshine_streaming_export_args(moonshine_streaming)

    smollm2 = model.add_parser("smollm2", help="Export SmolLM2 to Torq")
    add_smollm2_export_args(smollm2)

    gemma3 = model.add_parser("gemma3", help="Export Gemma3 to Torq")
    add_gemma3_export_args(gemma3)

    liquid = model.add_parser("liquid", help="Export LFM2.5 (Liquid) to Torq")
    add_liquid_export_args(liquid)

    liquid_vl = model.add_parser("liquid-vl", help="Export LFM2-VL-450M (Liquid) to Torq")
    add_liquid_vl_export_args(liquid_vl)

    yolo26 = model.add_parser("yolo26", help="Export YOLO26 detection model to Torq")
    add_yolo26_export_args(yolo26)

    yolo26_compile = model.add_parser("yolo26-compile", help="Compile the HF-hosted YOLO26 int8 TFLite to a Torq vmfb")
    add_yolo26_compile_args(yolo26_compile)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.export import export_moonshine_from_args
        export_moonshine_from_args(args)
    elif args.model_name == "moonshine_streaming":
        from .moonshine_streaming.export import export_moonshine_streaming_from_args
        export_moonshine_streaming_from_args(args)
    elif args.model_name == "smollm2":
        from .smollm2.export import export_smollm2_from_args
        export_smollm2_from_args(args)
    elif args.model_name == "gemma3":
        from .gemma3.export import export_gemma3_from_args
        export_gemma3_from_args(args)
    elif args.model_name == "liquid":
        from .liquid.export import export_liquid_from_args
        export_liquid_from_args(args)
    elif args.model_name == "liquid-vl":
        from .liquid.export_vl import export_liquid_vl_from_args
        export_liquid_vl_from_args(args)
    elif args.model_name == "yolo26":
        from .yolo26.export import export_yolo26_from_args
        export_yolo26_from_args(args)
    elif args.model_name == "yolo26-compile":
        from .yolo26.compile import compile_yolo26_from_args
        compile_yolo26_from_args(args)


if __name__ == "__main__":
    main()
