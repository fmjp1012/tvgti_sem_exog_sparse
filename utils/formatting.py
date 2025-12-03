"""
フォーマット・表示ユーティリティ

値の変換や表示用フォーマット関数を一元管理します。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def fmt_value(value: object) -> str:
    """
    値を表示用にフォーマットする。

    Parameters
    ----------
    value : object
        フォーマットする値

    Returns
    -------
    str
        フォーマット済み文字列
    """
    if value is None:
        return "<none>"
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def coerce_bool(value: object, default: bool = False) -> bool:
    """
    値をbool型に変換する。

    Parameters
    ----------
    value : object
        変換する値
    default : bool
        valueがNoneの場合のデフォルト値

    Returns
    -------
    bool
        変換後のbool値
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def print_block(title: str, items: Dict[str, object]) -> None:
    """
    ブロック形式で辞書を出力する。

    Parameters
    ----------
    title : str
        ブロックのタイトル
    items : Dict[str, object]
        出力する項目の辞書
    """
    if not items:
        return
    print(f"--- {title} ---")
    for key, value in items.items():
        print(f"{key:>24}: {fmt_value(value)}")


def to_python_value(value: Any) -> Any:
    """
    NumPy型をPython標準型に変換する。

    Parameters
    ----------
    value : Any
        変換する値

    Returns
    -------
    Any
        Python標準型に変換された値
    """
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, np.int_)):
        return int(value)
    return value


def clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    辞書内のNumPy型をPython標準型に変換する。

    Parameters
    ----------
    data : Dict[str, Any]
        変換する辞書

    Returns
    -------
    Dict[str, Any]
        変換後の辞書
    """
    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: _clean(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(val) for val in obj]
        return to_python_value(obj)
    
    return {key: _clean(val) for key, val in data.items()}


def to_list_of_ints(value: Any) -> list[int]:
    """
    値を整数リストに変換する。

    Parameters
    ----------
    value : Any
        変換する値（カンマ区切り文字列、リスト、または単一値）

    Returns
    -------
    list[int]
        整数のリスト
    """
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def to_list_of_bools(value: Any) -> list[bool]:
    """
    値をboolリストに変換する。

    Parameters
    ----------
    value : Any
        変換する値

    Returns
    -------
    list[bool]
        boolのリスト
    """
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [coerce_bool(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [coerce_bool(v) for v in value]
    return [coerce_bool(value)]

