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


def plot_heatmaps_suite(
    matrices: Dict[str, np.ndarray],
    base_save_path: Union[str, Path],
    title_suffix: str = "",
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = "RdBu_r",
    show: bool = True,
    use_offline_as_reference: bool = False,
) -> Dict[str, plt.Figure]:
    """Plot three types of heatmaps and save them.

    Generates:
    1. All matrices (estimates + reference)
    2. Estimates only (without reference)
    3. Difference from reference

    Parameters
    ----------
    matrices : Dict[str, np.ndarray]
        Dictionary mapping method names to their estimated matrices.
        Should include 'True' key (or 'Offline' key when use_offline_as_reference=True).
    base_save_path : Union[str, Path]
        Base path for saving figures. Suffix will be added before extension.
        e.g., "result/heatmap.png" -> "result/heatmap_all.png", etc.
    title_suffix : str
        Suffix to add to titles (e.g., "at t=49 (last trial)").
    figsize : Tuple[int, int]
        Figure size in inches.
    cmap : str
        Colormap name.
    show : bool
        Whether to display the figures.
    use_offline_as_reference : bool
        If True, use 'Offline' as the reference matrix instead of 'True'.

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with keys 'all', 'estimates_only', 'diff' mapping to figure objects.
    """
    base_path = Path(base_save_path)
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    # Determine reference key
    reference_key = "Offline" if use_offline_as_reference else "True"
    if reference_key not in matrices:
        raise ValueError(f"Reference key '{reference_key}' not found in matrices. "
                         f"Available keys: {list(matrices.keys())}")
    
    reference_matrix = matrices[reference_key]
    
    # Separate estimates and reference
    estimate_keys = [k for k in matrices.keys() if k not in ("True", "Offline")]
    estimates_only = {k: matrices[k] for k in estimate_keys}
    
    # Compute differences
    diff_matrices = {
        f"{k} - {reference_key}": matrices[k] - reference_matrix
        for k in estimate_keys
    }

    figures = {}

    # 1. All matrices (including reference)
    all_path = parent / f"{stem}_all{suffix}"
    figures["all"] = plot_heatmaps(
        matrices=matrices,
        save_path=all_path,
        title=f"All Estimates vs {reference_key} {title_suffix}".strip(),
        figsize=figsize,
        cmap=cmap,
        show=show,
    )

    # 2. Estimates only (excluding True/Offline)
    if estimates_only:
        estimates_path = parent / f"{stem}_estimates_only{suffix}"
        figures["estimates_only"] = plot_heatmaps(
            matrices=estimates_only,
            save_path=estimates_path,
            title=f"Estimates Only {title_suffix}".strip(),
            figsize=figsize,
            cmap=cmap,
            show=show,
        )

    # 3. Difference from reference
    if diff_matrices:
        diff_path = parent / f"{stem}_diff{suffix}"
        figures["diff"] = plot_heatmaps(
            matrices=diff_matrices,
            save_path=diff_path,
            title=f"Difference from {reference_key} {title_suffix}".strip(),
            figsize=figsize,
            cmap=cmap,
            show=show,
        )

    return figures
