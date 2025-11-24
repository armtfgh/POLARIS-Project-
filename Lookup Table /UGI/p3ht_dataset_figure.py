#!/usr/bin/env python3
"""
Generate publication-ready figures for the P3HT conductivity dataset.

Figure 1: Stacked histograms of every formulation parameter plus conductivity.
Figure 2: Feature-importance bars (absolute Pearson r vs. conductivity).
Figure 3: Two-panel formulation-space overview (scatter + conductivity heatmap).
Figure 4: PCA-based 2-D map compressing the 5-D formulation descriptors.

Usage (inside IPython/Jupyter):
>>> from p3ht_dataset_figure import load_dataset, show_figures, save_figures
>>> df = load_dataset()
>>> show_figures("figure1", df)   # or "figure2", "figure3", "figure4", "all"
>>> save_figures("all", df, directory="figures/p3ht", prefix="p3ht")
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).with_name("P3HT_dataset.csv")
FEATURE_COLUMNS = [
    "P3HT content (%)",
    "D1 content (%)",
    "D2 content (%)",
    "D6 content (%)",
    "D8 content (%)",
]
TARGET_COLUMN = "Conductivity"
HISTOGRAM_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
HISTOGRAM_COLORS = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
]
DESIGN_X = "D1 content (%)"
DESIGN_Y = "D2 content (%)"
COLOR_COLUMN = "P3HT content (%)"
SIZE_COLUMN = "D6 content (%)"
ALPHA_COLUMN = "D8 content (%)"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the curated P3HT dataset."""
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def scale_series(values: pd.Series, new_min: float, new_max: float) -> np.ndarray:
    """Scale a pandas Series to a new [min, max] interval."""
    array = values.to_numpy(dtype=float)
    vmin = float(array.min())
    vmax = float(array.max())
    if vmin == vmax:
        return np.full_like(array, (new_min + new_max) / 2, dtype=float)
    return new_min + (array - vmin) * (new_max - new_min) / (vmax - vmin)


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


# -----------------------------------------------------------------------------
# Figure 1 – Histograms
# -----------------------------------------------------------------------------
def make_histogram_figure(df: pd.DataFrame) -> plt.Figure:
    """Create Figure 1 – stacked parameter and conductivity histograms."""
    fig, axes = plt.subplots(
        len(HISTOGRAM_COLUMNS),
        1,
        figsize=(8, 14),
        constrained_layout=True,
        sharex=False,
    )
    label_map = {
        "P3HT content (%)": "P3HT content (%)",
        "D1 content (%)": "D1 content (%)",
        "D2 content (%)": "D2 content (%)",
        "D6 content (%)": "D6 content (%)",
        "D8 content (%)": "D8 content (%)",
        "Conductivity": "Conductivity (S/cm)",
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
        "Figure 1 – Parameter and conductivity distributions",
        fontsize=14,
        fontweight="bold",
    )
    return fig


# -----------------------------------------------------------------------------
# Figure 2 – Feature importances
# -----------------------------------------------------------------------------
def compute_feature_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Compute Pearson correlations between each feature and conductivity."""
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
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
    ax.set_yticklabels(ordered_cols)
    ax.set_xlabel("|Pearson correlation with conductivity|")
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
        -0.2,
        "Method: absolute Pearson correlation between each parameter and conductivity.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    return fig


# -----------------------------------------------------------------------------
# Figure 3 – Design-space overview
# -----------------------------------------------------------------------------
def make_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Scatter view of formulation coverage."""
    x = df[DESIGN_X].to_numpy(dtype=float)
    y = df[DESIGN_Y].to_numpy(dtype=float)
    colors = df[COLOR_COLUMN].to_numpy(dtype=float)
    sizes = scale_series(df[SIZE_COLUMN], 40.0, 240.0)
    alphas = scale_series(df[ALPHA_COLUMN], 0.35, 0.95)

    scatter = ax.scatter(
        x,
        y,
        c=colors,
        s=sizes,
        cmap="plasma",
        alpha=alphas,
        linewidth=0.2,
        edgecolor="#222222",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"{COLOR_COLUMN}")

    ax.set_xlabel(DESIGN_X)
    ax.set_ylabel(DESIGN_Y)
    ax.set_title("A  Formulation coverage", loc="left", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    # Marker-size legend for D6.
    d6_values = df[SIZE_COLUMN].quantile([0.1, 0.5, 0.9]).to_numpy()
    legend_sizes = scale_series(pd.Series(d6_values), 40.0, 240.0)
    handles = [
        ax.scatter([], [], s=size, color="none", edgecolor="#555555") for size in legend_sizes
    ]
    labels = [f"{val:.1f}%" for val in d6_values]
    ax.legend(
        handles,
        labels,
        title="D6 content (%)",
        scatterpoints=1,
        loc="upper right",
        frameon=False,
    )
    ax.text(
        0.02,
        0.06,
        "Opacity encodes D8 content (%).",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )


def compute_bin_means(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean values in a 2-D bin grid."""
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    counts = np.zeros((bins, bins), dtype=int)
    sums = np.zeros((bins, bins), dtype=float)
    for xi, yi, zi in zip(x, y, values):
        if np.isnan(xi) or np.isnan(yi) or np.isnan(zi):
            continue
        px = (xi - x_min) / (x_max - x_min + 1e-12)
        py = (yi - y_min) / (y_max - y_min + 1e-12)
        ix = min(max(int(px * bins), 0), bins - 1)
        iy = min(max(int(py * bins), 0), bins - 1)
        counts[iy, ix] += 1
        sums[iy, ix] += zi
    means = np.full_like(sums, np.nan)
    mask = counts > 0
    means[mask] = sums[mask] / counts[mask]
    return means, x_edges, y_edges


def make_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Heatmap of mean conductivity over the D1/D2 plane."""
    x = df[DESIGN_X].to_numpy(dtype=float)
    y = df[DESIGN_Y].to_numpy(dtype=float)
    values = df[TARGET_COLUMN].to_numpy(dtype=float)
    grid, x_edges, y_edges = compute_bin_means(x, y, values, bins=32)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean conductivity (S/cm)")
    ax.set_xlabel(DESIGN_X)
    ax.set_ylabel(DESIGN_Y)
    ax.set_title("B  Conductivity landscape", loc="left", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, linestyle=":")

    median = df[TARGET_COLUMN].median()
    top_decile = df[TARGET_COLUMN].quantile(0.9)
    ax.text(
        1.02,
        0.05,
        f"Median: {median:.1f} S/cm\nTop decile: {top_decile:.1f} S/cm",
        transform=ax.transAxes,
        fontsize=9,
        color="#222222",
        ha="left",
        va="bottom",
    )


def make_overview_figure(df: pd.DataFrame) -> plt.Figure:
    """Create Figure 3 – two-panel formulation overview."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
        sharey=False,
    )
    make_panel_a(axes[0], df)
    make_panel_b(axes[1], df)
    fig.suptitle("Figure 3 – P3HT formulation space overview", fontsize=14, fontweight="bold")
    return fig


# -----------------------------------------------------------------------------
# Figure 4 – PCA embedding
# -----------------------------------------------------------------------------
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
    vec2 = vec2 - vector_dot(vec2, vec1) * vec1
    vec2 = normalize_vector(vec2)

    embedding = np.zeros((standardized.shape[0], 2), dtype=float)
    embedding[:, 0] = project_component(standardized, vec1)
    embedding[:, 1] = project_component(standardized, vec2)

    total_variance = float(np.trace(cov))
    explained = np.array([eig1, eig2]) / max(total_variance, 1e-12)
    return embedding, explained


def make_embedding_figure(df: pd.DataFrame) -> plt.Figure:
    """Create Figure 4 – PCA-based space map."""
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

    scatter_cond = axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df[TARGET_COLUMN],
        cmap="inferno",
        s=20,
        alpha=0.75,
        linewidth=0,
    )
    plt.colorbar(
        scatter_cond, ax=axes[0], fraction=0.046, pad=0.04, label="Conductivity (S/cm)"
    )
    axes[0].set_title("A  PCA map colored by conductivity", loc="left", fontweight="bold")

    scatter_d8 = axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df[ALPHA_COLUMN],
        cmap="viridis",
        s=20,
        alpha=0.75,
        linewidth=0,
    )
    plt.colorbar(
        scatter_d8,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="D8 content (%)",
    )
    axes[1].set_title("B  PCA map colored by D8 content", loc="left", fontweight="bold")

    for ax in axes:
        ax.set_xlabel(pc1_label)
        ax.set_ylabel(pc2_label)
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 4 – P3HT formulation PCA embedding", fontsize=14, fontweight="bold")
    return fig


FIGURE_REGISTRY: dict[str, tuple[callable, str]] = {
    "histograms": (make_histogram_figure, "Figure 1 – Parameter histograms"),
    "importance": (
        make_importance_figure,
        "Figure 2 – Feature importance vs. conductivity",
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


# -----------------------------------------------------------------------------
# Interactive helper
# -----------------------------------------------------------------------------
def set_window_title(fig: plt.Figure, title: str) -> None:
    """Set the GUI window title if supported by the backend."""
    manager = fig.canvas.manager
    if manager is not None:
        with contextlib.suppress(AttributeError):
            manager.set_window_title(title)


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
    prefix: str = "p3ht",
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


__all__ = [
    "load_dataset",
    "show_figures",
    "save_figures",
    "make_histogram_figure",
    "make_importance_figure",
    "make_overview_figure",
    "make_embedding_figure",
]
