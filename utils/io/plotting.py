from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Optional


def apply_style(
    use_latex: bool = True,
    font_family: str = "Times New Roman",
    base_font_size: int = 15,
) -> None:
    """Apply a consistent matplotlib style across simulations.

    Parameters
    ----------
    use_latex : bool
        Whether to enable LaTeX rendering.
    font_family : str
        Primary font family.
    base_font_size : int
        Base font size for all text.
    """
    plt.rc("text", usetex=use_latex)
    plt.rc("font", family="serif")
    plt.rcParams["font.family"] = font_family
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.minor.width"] = 1.0
    plt.rcParams["ytick.minor.width"] = 1.0
    plt.rcParams["xtick.major.size"] = 10
    plt.rcParams["ytick.major.size"] = 10
    plt.rcParams["xtick.minor.size"] = 5
    plt.rcParams["ytick.minor.size"] = 5
    plt.rcParams["font.size"] = base_font_size
