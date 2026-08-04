"""
Usage:  python mona.py --config configs/config_mona.yaml
"""

from src.config import parse_config_arg
from src.mona import load_mona_json, parse_mona

def main(cfg):
    data = load_mona_json(cfg["input_json"])
    df = parse_mona(data)
    df.to_csv(cfg["output_csv"], index=False)
    print(len(df))

if __name__ == "__main__":
    config = parse_config_arg(description="Parse MoNA JSON export into a CSV.")
    main(config)
