# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .moonshine import add_moonshine_infer_args
from .customer_b import add_customer_b_infer_args


def main():
    parser = argparse.ArgumentParser(description="Infer models")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Run Moonshine inference")
    add_moonshine_infer_args(moonshine)

    customer_b = model.add_parser("customer_b", help="Run Customer B inference")
    add_customer_b_infer_args(customer_b)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.infer import infer_moonshine
        infer_moonshine(args)
    elif args.model_name == "customer_b":
        from .customer_b.infer import infer_customer_b
        infer_customer_b(args)


if __name__ == "__main__":
    main()
