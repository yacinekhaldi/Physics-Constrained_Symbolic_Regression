from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcsr.data import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    args = parser.parse_args()
    csv_path = download_dataset(force=args.force)
    print(f"Dataset CSV: {csv_path}")


if __name__ == "__main__":
    main()
