# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .moonshine import add_moonshine_infer_args
from .smollm2 import add_smollm2_infer_args


def main():
    parser = argparse.ArgumentParser(description="Infer models")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Run Moonshine inference")
    add_moonshine_infer_args(moonshine)

    smollm2 = model.add_parser("smollm2", help="Run SmolLM2 inference")
    add_smollm2_infer_args(smollm2)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.infer import infer_moonshine
        infer_moonshine(args)
    elif args.model_name == "smollm2":
        from .smollm2.infer import infer_smollm2
        infer_smollm2(args)


if __name__ == "__main__":
    main()
