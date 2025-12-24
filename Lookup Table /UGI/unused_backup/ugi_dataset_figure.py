#!/usr/bin/env python3
"""
Generate publication-ready figures for the UGI reaction dataset.

Figure 1: Stacked histograms of all parameters plus yield.
Figure 2: Feature-importance bars (Pearson |r| vs. yield).
Figure 3: Two-panel design-space overview used in the manuscript.
Figure 4: PCA-based 2-D map that compresses the 4-D stoichiometric grid.

Usage (inside IPython/Jupyter):
>>> from ugi_dataset_figure import load_dataset, show_figures, save_figures
>>> df = load_dataset()
>>> show_figures("figure1", df)  # or "figure2", "figure3", "figure4", "all"
>>> save_figures("all", df, directory="figures/ugi", prefix="ugi")
"""
#%%
from __future__ import annotations
import contextlib
from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).with_name("ugi_merged_dataset.csv")
FEATURE_COLUMNS = ["amine_mM", "aldehyde_mM", "isocyanide_mM", "ptsa"]
HISTOGRAM_COLUMNS = FEATURE_COLUMNS + ["yield"]
HISTOGRAM_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the merged UGI dataset."""
    return pd.read_csv(path)


def make_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot stoichiometric coverage with jittered scatter markers."""
    rng = np.random.default_rng(42)
    jitter = 3.0  # mM jitter so discrete grids are easier to see
    x = df["amine_mM"].to_numpy() + rng.normal(0, jitter, len(df))
    y = df["aldehyde_mM"].to_numpy() + rng.normal(0, jitter, len(df))

    ptsa_min, ptsa_max = df["ptsa"].min(), df["ptsa"].max()
    sizes = np.interp(df["ptsa"], (ptsa_min, ptsa_max), (15, 120))

    scatter = ax.scatter(
        x,
        y,
        c=df["isocyanide_mM"],
        s=sizes,
        cmap="viridis",
        alpha=0.55,
        linewidth=0,
    )
    cbar = plt.colorbar(scatter, ax=ax)

    ax.set_title("A  Design space coverage", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Amine (mM)")
    ax.set_ylabel("Aldehyde (mM)")
    ax.set_xlim(110, 310)
    ax.set_ylim(110, 310)
    ax.set_xticks(sorted(df["amine_mM"].unique()))
    ax.set_yticks(sorted(df["aldehyde_mM"].unique()))
    ax.grid(alpha=0.15)

    # Compact legend for p-TsOH loading encoded as marker size.
    quantiles = df["ptsa"].quantile([0.1, 0.5, 0.9]).to_numpy()
    legend_sizes = np.interp(quantiles, (ptsa_min, ptsa_max), (15, 120))
    handles = [
        ax.scatter([], [], s=size, color="tab:gray", alpha=0.55) for size in legend_sizes
    ]
    labels = [f"{value:.2f}" for value in quantiles]
    ax.legend(
        handles,
        labels,
        title="p-TsOH (mediated fraction)",
        scatterpoints=1,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
    )

    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Isocyanide (mM)", fontsize=10)


def make_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot mean yield surface projected on amine vs. aldehyde."""
    grid = (
        df.groupby(["aldehyde_mM", "amine_mM"])["yield"]
        .mean()
        .unstack()
        .sort_index()
        .sort_index(axis=1)
    )
    x_levels = grid.columns.to_numpy()
    y_levels = grid.index.to_numpy()

    im = ax.imshow(
        grid.to_numpy(),
        origin="lower",
        cmap="inferno",
        aspect="auto",
        vmin=df["yield"].min(),
        vmax=df["yield"].max(),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean yield (fraction)")

    ax.set_title("B  Yield landscape", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Amine (mM)")
    ax.set_ylabel("Aldehyde (mM)")
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_yticks(np.arange(len(y_levels)))
    ax.set_xticklabels([f"{int(x)}" for x in x_levels], rotation=45)
    ax.set_yticklabels([f"{int(y)}" for y in y_levels])

    # Annotate with quick stats so the panel reads standalone.
    median = df["yield"].median() * 100
    top_decile = df["yield"].quantile(0.9) * 100
    ax.text(
        1.02,
        0.03,
        f"Median: {median:.1f}%\nTop decile: {top_decile:.1f}%",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )


def make_overview_figure(df: pd.DataFrame) -> plt.Figure:
    """Create the two-panel overview figure and return the matplotlib figure."""
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True, sharey=False
    )
    make_panel_a(axes[0], df)
    make_panel_b(axes[1], df)
    fig.suptitle("UGI Reaction Dataset Overview", fontsize=14, fontweight="bold")
    return fig


def simple_histogram(values: np.ndarray, bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Manual histogram (avoids numpy's OpenMP-heavy implementation)."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        edges = np.linspace(0, 1, bins + 1)
        return np.zeros(bins, dtype=int), edges

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    edges = np.linspace(vmin, vmax, bins + 1)
    counts = np.zeros(bins, dtype=int)
    span = vmax - vmin
    for value in finite_values:
        position = (value - vmin) / span
        idx = int(position * bins)
        if idx < 0:
            idx = 0
        elif idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts, edges


def make_histogram_figure(df: pd.DataFrame) -> plt.Figure:
    """Create Figure 1 – stacked parameter and yield histograms."""
    fig, axes = plt.subplots(
        len(HISTOGRAM_COLUMNS),
        1,
        figsize=(8, 12),
        constrained_layout=True,
        sharex=False,
    )
    label_map = {
        "amine_mM": "Amine (mM)",
        "aldehyde_mM": "Aldehyde (mM)",
        "isocyanide_mM": "Isocyanide (mM)",
        "ptsa": "p-TsOH (mediated fraction)",
        "yield": "Yield (fraction)",
    }
    for ax, column, color in zip(axes, HISTOGRAM_COLUMNS, HISTOGRAM_COLORS):
        values = df[column].to_numpy(dtype=float)
        counts, edges = simple_histogram(values, bins=30)
        poly_x = []
        poly_y = []
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            poly_x.extend([left, left, right, right])
            poly_y.extend([0, count, count, 0])
        poly_x.append(edges[-1])
        poly_y.append(0)
        poly_x = np.asarray(poly_x)
        poly_y = np.asarray(poly_y, dtype=float)
        ax.fill_between(poly_x, poly_y, color=color, alpha=0.35)
        ax.plot(poly_x, poly_y, color=color, linewidth=1.0)
        ax.set_ylabel("Count")
        ax.set_title(label_map.get(column, column), loc="left", fontsize=11)
        ax.grid(alpha=0.15, axis="y")
        ax.set_xlim(edges[0], edges[-1])

    axes[-1].set_xlabel("Value")
    fig.suptitle(
        "Figure 1 – Parameter and yield distributions", fontsize=14, fontweight="bold"
    )
    return fig


def compute_feature_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Compute Pearson correlations between each feature and the yield."""
    y = df["yield"].to_numpy(dtype=float)
    y_centered = y - y.mean()
    y_norm = np.sqrt((y_centered**2).sum())
    results: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        x = df[column].to_numpy(dtype=float)
        x_centered = x - x.mean()
        denom = np.sqrt((x_centered**2).sum()) * y_norm
        corr = 0.0 if denom == 0 else float((x_centered * y_centered).sum() / denom)
        results[column] = corr
    return results


def make_importance_figure(df: pd.DataFrame) -> plt.Figure:
    """Create Figure 2 – feature importance bars based on |Pearson r|."""
    correlations = compute_feature_correlations(df)
    columns = list(correlations.keys())
    importance = np.array([abs(correlations[col]) for col in columns])
    order = np.argsort(importance)[::-1]
    ordered_cols = [columns[i] for i in order]
    ordered_importance = importance[order]

    col_labels = {
        "amine_mM": "Amine (mM)",
        "aldehyde_mM": "Aldehyde (mM)",
        "isocyanide_mM": "Isocyanide (mM)",
        "ptsa": "p-TsOH",
    }

    if ordered_importance.size:
        denom = ordered_importance.max()
        denom = 1.0 if denom == 0 else denom
        norm = ordered_importance / denom
    else:
        norm = ordered_importance
    cmap = plt.get_cmap("viridis")
    colors = [cmap(value) for value in norm]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    positions = np.arange(len(ordered_cols))
    ax.set_yticks(positions)
    ax.set_yticklabels([col_labels.get(col, col) for col in ordered_cols])
    ax.set_xlabel("|Pearson correlation with yield|")
    ax.grid(axis="x", alpha=0.2)
    for idx, value in enumerate(ordered_importance):
        y0 = idx - 0.35
        y1 = idx + 0.35
        ax.fill_betweenx(
            [y0, y1],
            np.zeros(2),
            np.full(2, value),
            color=colors[idx],
            alpha=0.85,
        )
        ax.plot([0, value], [idx, idx], color="#1f1f1f", linewidth=1.2)
        ax.text(
            value + 0.01,
            idx,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_ylim(-0.5, len(ordered_cols) - 0.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(ordered_importance.max() * 1.1, 0.05))
    ax.set_title(
        "Figure 2 – Feature importance (absolute Pearson r)",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.text(
        0.0,
        -0.15,
        "Method: absolute Pearson correlation between each parameter and yield.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    return fig


def standardize_features(df: pd.DataFrame) -> np.ndarray:
    """Return z-scored features for dimension reduction."""
    features = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std


def covariance_matrix(data: np.ndarray) -> np.ndarray:
    """Compute the covariance matrix using only elementwise operations."""
    n_samples, n_features = data.shape
    cov = np.zeros((n_features, n_features), dtype=float)
    denom = max(n_samples - 1, 1)
    for i in range(n_features):
        for j in range(i, n_features):
            prod = data[:, i] * data[:, j]
            value = float(prod.sum()) / denom
            cov[i, j] = cov[j, i] = value
    return cov


def matrix_vector_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Multiply matrix and vector without BLAS (avoids sandbox restrictions)."""
    size = len(vector)
    result = np.zeros(size, dtype=float)
    for i in range(size):
        total = 0.0
        for j in range(size):
            total += float(matrix[i, j]) * float(vector[j])
        result[i] = total
    return result


def vector_dot(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Dot product implemented via elementwise multiply/sum."""
    return float((vec_a * vec_b).sum())


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return a unit vector in the same direction."""
    norm = float(np.sqrt((vec**2).sum()))
    if norm == 0.0:
        return vec.copy()
    return vec / norm


def power_iteration(
    matrix: np.ndarray, iterations: int = 300, seed: int | None = None
) -> tuple[float, np.ndarray]:
    """Compute the dominant eigenpair via power iteration."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=matrix.shape[0]).astype(float)
    vec = normalize_vector(vec)
    for _ in range(iterations):
        vec = normalize_vector(matrix_vector_product(matrix, vec))
    eigenvalue = vector_dot(vec, matrix_vector_product(matrix, vec))
    return eigenvalue, vec


def deflate_matrix(
    matrix: np.ndarray, eigenvalue: float, eigenvector: np.ndarray
) -> np.ndarray:
    """Deflate the matrix to reveal the next eigenpair."""
    size = len(eigenvector)
    deflated = matrix.copy()
    for i in range(size):
        for j in range(size):
            deflated[i, j] -= eigenvalue * eigenvector[i] * eigenvector[j]
    return deflated


def project_component(data: np.ndarray, component: np.ndarray) -> np.ndarray:
    """Project standardized data onto a component vector."""
    projection = np.zeros(data.shape[0], dtype=float)
    for idx in range(component.shape[0]):
        projection += data[:, idx] * component[idx]
    return projection


def compute_pca_embedding(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 2-D PCA embedding and its explained variance ratios."""
    standardized = standardize_features(df)
    cov = covariance_matrix(standardized)
    eig1, vec1 = power_iteration(cov, seed=0)
    deflated = deflate_matrix(cov, eig1, vec1)
    eig2, vec2 = power_iteration(deflated, seed=1)
    # Re-orthogonalize to guard numerical drift.
    vec2 = vec2 - vector_dot(vec2, vec1) * vec1
    vec2 = normalize_vector(vec2)

    embedding = np.zeros((standardized.shape[0], 2), dtype=float)
    embedding[:, 0] = project_component(standardized, vec1)
    embedding[:, 1] = project_component(standardized, vec2)

    total_variance = float(np.trace(cov))
    explained = np.array([eig1, eig2]) / max(total_variance, 1e-12)
    return embedding, explained


def make_embedding_figure(df: pd.DataFrame) -> plt.Figure:
    """Create the PCA-based space map figure."""
    embedding, explained = compute_pca_embedding(df)
    pc1_label = f"PC1 ({explained[0] * 100:.1f}% var.)"
    pc2_label = f"PC2 ({explained[1] * 100:.1f}% var.)"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    scatter_yield = axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df["yield"],
        cmap="inferno",
        s=12,
        alpha=0.7,
        linewidth=0,
    )
    plt.colorbar(
        scatter_yield, ax=axes[0], fraction=0.046, pad=0.04, label="Yield (fraction)"
    )
    axes[0].set_title("A  PCA map colored by yield", loc="left", fontweight="bold")

    scatter_ptsa = axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df["ptsa"],
        cmap="viridis",
        s=12,
        alpha=0.7,
        linewidth=0,
    )
    plt.colorbar(
        scatter_ptsa,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="p-TsOH (mediated fraction)",
    )
    axes[1].set_title("B  PCA map colored by p-TsOH", loc="left", fontweight="bold")

    for ax in axes:
        ax.set_xlabel(pc1_label)
        ax.set_ylabel(pc2_label)
        ax.grid(alpha=0.2)

    fig.suptitle("UGI reaction space – PCA embedding", fontsize=14, fontweight="bold")
    return fig


FIGURE_REGISTRY: dict[str, tuple[callable, str]] = {
    "histograms": (make_histogram_figure, "Figure 1 – Parameter histograms"),
    "importance": (
        make_importance_figure,
        "Figure 2 – Feature importance vs. yield",
    ),
    "overview": (make_overview_figure, "Figure 3 – Design space overview"),
    "embedding": (make_embedding_figure, "Figure 4 – PCA space map"),
}
FIGURE_ALIASES = {
    "figure1": "histograms",
    "figure2": "importance",
    "figure3": "overview",
    "figure4": "embedding",
}
FIGURE_SEQUENCE = list(FIGURE_REGISTRY.keys())
VALID_MODES = set(FIGURE_SEQUENCE) | {"both", "all"} | set(FIGURE_ALIASES.keys())


def _select_modes(mode: str) -> list[str]:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    resolved = FIGURE_ALIASES.get(mode, mode)
    if resolved == "both":
        return ["overview", "embedding"]
    if resolved == "all":
        return FIGURE_SEQUENCE
    return [resolved]


def _build_figures(mode: str, df: pd.DataFrame | None) -> list[tuple[plt.Figure, str, str]]:
    if df is None:
        df = load_dataset()
    figures: list[tuple[plt.Figure, str, str]] = []
    for key in _select_modes(mode):
        builder, title = FIGURE_REGISTRY[key]
        figures.append((builder(df), title, key))
    return figures


def show_figures(
    mode: str = "overview", df: pd.DataFrame | None = None
) -> list[plt.Figure]:
    """Interactive-friendly helper to render figures inside IPython/Jupyter."""
    figures = _build_figures(mode, df)
    for fig, title, _ in figures:
        set_window_title(fig, title)
    plt.show()
    return [fig for fig, _, _ in figures]


def save_figures(
    mode: str = "overview",
    df: pd.DataFrame | None = None,
    directory: str | Path = "figures",
    prefix: str = "ugi",
    fmt: str = "png",
    dpi: int = 300,
) -> list[Path]:
    """Save the requested figures to disk instead of showing them."""
    figures = _build_figures(mode, df)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fig, _, key in figures:
        filename = f"{prefix}_{key}.{fmt}"
        path = directory / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
        plt.close(fig)
    return saved_paths


def set_window_title(fig: plt.Figure, title: str) -> None:
    """Set the GUI window title if supported by the backend."""
    manager = fig.canvas.manager
    if manager is not None:
        with contextlib.suppress(AttributeError):
            manager.set_window_title(title)


__all__ = [
    "load_dataset",
    "show_figures",
    "save_figures",
    "make_histogram_figure",
    "make_importance_figure",
    "make_overview_figure",
    "make_embedding_figure",
]
