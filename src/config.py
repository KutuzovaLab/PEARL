import argparse
import yaml

def parse_config_arg(description="PEARL script"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file for this script.",
    )
    args = parser.parse_args()
    return load_config(args.config)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
