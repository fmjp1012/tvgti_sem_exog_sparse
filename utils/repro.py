from __future__ import annotations

import platform
import sys
from dataclasses import asdict, is_dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def to_jsonable(obj: Any) -> Any:
    """JSONに落とせる形へ変換する（Path/ndarray/np scalar など対応）。"""
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    return obj


def collect_environment_info(
    package_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """再現性のための環境情報を収集する。"""
    pkg_names = list(package_names) if package_names is not None else []
    packages: Dict[str, Optional[str]] = {}
    for name in pkg_names:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
        except Exception:
            packages[name] = None

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }
