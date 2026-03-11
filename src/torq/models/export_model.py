# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .moonshine import add_moonshine_export_args
from .customer_b import add_customer_b_export_args


def main():
    parser = argparse.ArgumentParser(description="Export models to Torq")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Export moonshine to Torq")
    add_moonshine_export_args(moonshine)

    customer_b = model.add_parser("customer_b", help="Export Customer B models to Torq")
    add_customer_b_export_args(customer_b)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.export import export_moonshine_from_args
        export_moonshine_from_args(args)
    elif args.model_name == "customer_b":
        from .customer_b.export import export_customer_b_from_args
        export_customer_b_from_args(args)


if __name__ == "__main__":
    main()
