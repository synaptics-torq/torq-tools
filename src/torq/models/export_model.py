# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .moonshine import add_moonshine_export_args
from .smollm2 import add_smollm2_export_args


def main():
    parser = argparse.ArgumentParser(description="Export models to Torq")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Export moonshine to Torq")
    add_moonshine_export_args(moonshine)

    smollm2 = model.add_parser("smollm2", help="Export SmolLM2 to Torq")
    add_smollm2_export_args(smollm2)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.export import export_moonshine_from_args
        export_moonshine_from_args(args)
    elif args.model_name == "smollm2":
        from .smollm2.export import export_smollm2_from_args
        export_smollm2_from_args(args)


if __name__ == "__main__":
    main()
