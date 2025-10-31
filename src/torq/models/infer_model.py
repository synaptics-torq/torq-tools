import argparse

from .moonshine import add_moonshine_infer_args


def main():
    parser = argparse.ArgumentParser(description="Infer models")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Run Moonshine inference")
    add_moonshine_infer_args(moonshine)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.infer import infer_moonshine
        infer_moonshine(args)


if __name__ == "__main__":
    main()
