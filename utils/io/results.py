from __future__ import annotations
import shutil
import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


def create_result_dir(root: Path, scenario: str, extra_tag: Optional[str] = None) -> Path:
    ts = dt.datetime.now().strftime("%y%m%d")
    base = root / ts
    base.mkdir(parents=True, exist_ok=True)
    # optional subdir per scenario
    scenario_dir = base / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    if extra_tag:
        tag_dir = scenario_dir / extra_tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        return tag_dir
    return scenario_dir


def backup_script(script_path: Path, dest_dir: Path) -> Path:
    dest = dest_dir / f"{Path(script_path).name}"
    # すでに同じ場所にある場合（例: scripts_dir 配下のファイルを scripts_dir にバックアップ）
    # は shutil.copy が SameFileError になるのでスキップする。
    try:
        if Path(script_path).resolve() == Path(dest).resolve():
            return dest
        shutil.copy(str(script_path), str(dest))
    except shutil.SameFileError:
        return dest
    return dest


def save_json(meta: Dict[str, Any], dest_dir: Path, name: str = "meta.json") -> Path:
    p = dest_dir / name
    with p.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return p


def make_result_filename(prefix: str, params: Dict[str, Any], suffix: str = ".png") -> str:
    """
    Build a standardized result filename with stable key ordering.
    Example: prefix=piecewise, params={"N":20,"T":1000,"seed":3} ->
    piecewise_N=20_T=1000_seed=3.png
    If params is large, include a short hash tail to avoid overlong names.
    """
    # Flatten and order keys for determinism
    items = [f"{k}={params[k]}" for k in sorted(params.keys())]
    core = "_".join(items)
    name = f"{prefix}_{core}"
    if len(name) > 200:
        h = hashlib.sha1(core.encode("utf-8")).hexdigest()[:8]
        name = f"{prefix}_{h}"
    return f"{name}{suffix}"
