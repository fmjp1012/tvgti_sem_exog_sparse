from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

# Local imports
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running as a script
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.io.plotting import apply_style
from utils.io.results import create_result_dir, backup_script, save_json
from utils.configs.types import ExperimentConfig


def load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML is required. Please install with: pip install pyyaml") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified TVGTI simulation runner")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario override")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    parser.add_argument("--out_root", type=str, default=None, help="Output root override")
    args = parser.parse_args()

    # Load config via dataclass
    if args.config is None:
        ec = ExperimentConfig()
    else:
        ec = ExperimentConfig.from_yaml(Path(args.config))
    if args.scenario:
        ec.scenario = args.scenario
    if args.seed is not None:
        ec.seed = args.seed
    if args.out_root:
        ec.output.root = args.out_root

    apply_style(
        use_latex=ec.plot.latex,
        font_family=ec.plot.font,
        base_font_size=ec.plot.size,
    )

    # Prepare output directory (preserve current date structure)
    result_dir = create_result_dir(Path(ec.output.root), ec.scenario)
    backup_script(Path(__file__), result_dir)

    # Dispatch to existing run scripts by scenario name
    dispatch = {
        "linear": ("code.run_linear", "main"),
        "brownian": ("code.run_brownian", "main"),
        "piecewise": ("code.run_piecewise", "main"),
        "linear_mean": ("code.run_linear_mean", "main"),
        "brownian_mean": ("code.run_brownian_mean", "main"),
        "piecewise_mean": ("code.run_piecewise_mean", "main"),
        "linear_once": ("code.run_linear_once", "main"),
        "brownian_once": ("code.run_brownian_once", "main"),
        "piecewise_once": ("code.run_piecewise_once", "main"),
    }

    meta: Dict[str, Any] = {"scenario": ec.scenario, "seed": ec.seed, "config": {
        "plot": {"latex": ec.plot.latex, "font": ec.plot.font, "size": ec.plot.size},
        "output": {"root": ec.output.root},
        **ec.extra,
    }}
    save_json(meta, result_dir, name="meta.json")

    if ec.scenario in dispatch:
        module_name, func_name = dispatch[ec.scenario]
        mod = __import__(module_name, fromlist=[func_name])
        getattr(mod, func_name)()
    else:
        print(f"Unknown scenario '{ec.scenario}'. Available: {sorted(dispatch.keys())}")
        return

    print(f"Output: {result_dir}")


if __name__ == "__main__":
    main()
