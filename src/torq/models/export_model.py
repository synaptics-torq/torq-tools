import argparse

from .moonshine import add_moonshine_export_args


def main():
    parser = argparse.ArgumentParser(description="Export models to Torq")
    model = parser.add_subparsers(dest="model_name", required=True)

    moonshine = model.add_parser("moonshine", help="Export moonshine to Torq")
    add_moonshine_export_args(moonshine)

    args = parser.parse_args()

    if args.model_name == "moonshine":
        from .moonshine.export import export_moonshine_from_args
        export_moonshine_from_args(args)


if __name__ == "__main__":
    main()
