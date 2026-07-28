# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .moonshine import add_moonshine_infer_args
from .moonshine_streaming import add_moonshine_streaming_infer_args
from .smollm2 import add_smollm2_infer_args
from .gemma3 import add_gemma3_infer_args
from .liquid import add_liquid_infer_args
from .rtmo import add_rtmo_infer_args


def main():
    parser = argparse.ArgumentParser(description="Infer models")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Run Moonshine inference")
    add_moonshine_infer_args(moonshine)

    moonshine_streaming = model.add_parser("moonshine_streaming", help="Run Moonshine Streaming inference")
    add_moonshine_streaming_infer_args(moonshine_streaming)

    smollm2 = model.add_parser("smollm2", help="Run SmolLM2 inference")
    add_smollm2_infer_args(smollm2)

    gemma3 = model.add_parser("gemma3", help="Run Gemma3 inference")
    add_gemma3_infer_args(gemma3)

    liquid = model.add_parser("liquid", help="Run LFM2.5 (Liquid) inference")
    add_liquid_infer_args(liquid)

    rtmo = model.add_parser("rtmo", help="Run RTMO tiny pose inference")
    add_rtmo_infer_args(rtmo)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.infer import infer_moonshine
        infer_moonshine(args)
    elif args.model_name == "moonshine_streaming":
        from .moonshine_streaming.infer import infer_moonshine_streaming
        infer_moonshine_streaming(args)
    elif args.model_name == "smollm2":
        from .smollm2.infer import infer_smollm2
        infer_smollm2(args)
    elif args.model_name == "gemma3":
        from .gemma3.infer import infer_gemma3
        infer_gemma3(args)
    elif args.model_name == "liquid":
        from .liquid.infer import infer_liquid
        infer_liquid(args)
    elif args.model_name == "rtmo":
        from .rtmo.infer import infer_rtmo
        infer_rtmo(args)


if __name__ == "__main__":
    main()
