from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcsr.experiment import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["Nuavg"], help="Targets to model.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gp-population", type=int, default=250)
    parser.add_argument("--gp-generations", type=int, default=6)
    parser.add_argument(
        "--full-gp",
        action="store_true",
        help="Also run GP-SR for sparse/noise ablations. Slower.",
    )
    parser.add_argument("--xgb-estimators", type=int, default=700)
    args = parser.parse_args()
    manifest = run_experiments(
        targets=args.targets,
        random_state=args.random_state,
        gp_population=args.gp_population,
        gp_generations=args.gp_generations,
        full_gp=args.full_gp,
        xgb_estimators=args.xgb_estimators,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
