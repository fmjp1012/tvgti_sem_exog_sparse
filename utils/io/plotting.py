from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


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


def plot_heatmaps(
    matrices: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = "RdBu_r",
    show: bool = True,
) -> plt.Figure:
    """Plot heatmaps comparing multiple matrices (True vs estimated).

    Parameters
    ----------
    matrices : Dict[str, np.ndarray]
        Dictionary mapping method names to their estimated matrices.
        Should include 'True' key for the ground truth matrix.
    save_path : Optional[Union[str, Path]]
        Path to save the figure. If None, figure is not saved.
    title : Optional[str]
        Overall figure title (e.g., "t=999").
    figsize : Tuple[int, int]
        Figure size in inches.
    cmap : str
        Colormap name.
    show : bool
        Whether to display the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    n_matrices = len(matrices)
    # Determine grid layout
    if n_matrices <= 3:
        nrows, ncols = 1, n_matrices
    elif n_matrices <= 6:
        nrows, ncols = 2, (n_matrices + 1) // 2
    else:
        ncols = 4
        nrows = (n_matrices + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    if n_matrices == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    # Compute global max for consistent color scale
    max_abs = max(float(np.max(np.abs(m))) for m in matrices.values()) + 1e-12

    for idx, (name, mat) in enumerate(matrices.items()):
        ax = axes[idx]
        im = ax.imshow(
            mat,
            cmap=cmap,
            vmin=-max_abs,
            vmax=max_abs,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_matrices, len(axes)):
        axes[idx].axis("off")

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[:n_matrices], location="right", shrink=0.8, pad=0.02)
    cbar.set_label("")

    if title:
        fig.suptitle(title, fontsize=14)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
