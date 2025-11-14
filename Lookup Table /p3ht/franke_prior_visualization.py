#!/usr/bin/env python3
"""
Nine-panel Franke-surface storyboard for manuscript Figure 2.

Grid (3 rows × 3 columns):
Row 1: Ground truth, Prior A (x₁↑/x₂↓), Prior B (x₁↓/x₂↑).
Row 2: Prior C (synergy-heavy), Prior D (peak/valley + bump), Prior E (reversed peak/valley + bump).
Row 3: Prior F (data-aligned hybrid), Prior G/H showing confidence-scale sensitivity (e.g., high vs low confidence).
"""
#%%
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import argparse

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize


@dataclass
class Stats:
    rho: float
    alpha: float


@dataclass
class PriorPanel:
    title: str
    description: str
    surface: np.ndarray
    stats: Stats


def franke_surface(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    term1 = 0.75 * np.exp(-(9 * x1 - 2) ** 2 / 4.0 - (9 * x2 - 2) ** 2 / 4.0)
    term2 = 0.75 * np.exp(-((9 * x1 + 1) ** 2) / 49.0 - (9 * x2 + 1) / 10.0)
    term3 = 0.5 * np.exp(-(9 * x1 - 7) ** 2 / 4.0 - (9 * x2 - 3) ** 2 / 4.0)
    term4 = -0.2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)
    return term1 + term2 + term3 + term4


def stats_against_truth(truth: np.ndarray, guess: np.ndarray) -> Stats:
    tf = truth.ravel()
    gf = guess.ravel()
    rho = float(np.corrcoef(tf, gf)[0, 1])
    denom = np.dot(gf, gf) + 1e-12
    alpha = float(np.dot(tf, gf) / denom)
    return Stats(rho=rho, alpha=alpha)


def evaluate_prior_surface(spec: Dict[str, Any], X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    d = grid.shape[1]
    out = np.zeros(grid.shape[0], dtype=np.float64)

    def parse_dim(name: Any) -> int | None:
        if isinstance(name, int):
            idx = name
        elif isinstance(name, str) and name.startswith("x"):
            try:
                idx = int(name[1:]) - 1
            except ValueError:
                return None
        else:
            return None
        return idx if 0 <= idx < d else None

    def sigmoid(z: np.ndarray, center: float = 0.5, k: float = 6.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-k * (z - center)))

    def gauss1d(z: np.ndarray, mu: float, s: float) -> np.ndarray:
        s = max(s, 1e-6)
        return np.exp(-0.5 * ((z - mu) / s) ** 2)

    for name, eff_spec in (spec.get("effects") or {}).items():
        idx = parse_dim(name)
        if idx is None:
            continue
        z = grid[:, idx]
        eff = str(eff_spec.get("effect", "flat")).lower()
        scale = float(eff_spec.get("scale", 0.0))
        conf = float(eff_spec.get("confidence", 0.0))
        amp = 0.6 * scale * conf
        if amp == 0.0:
            continue

        range_hint = eff_spec.get("range_hint")
        center = 0.5
        width = 0.18
        if isinstance(range_hint, (list, tuple)) and len(range_hint) == 2:
            lo, hi = float(range_hint[0]), float(range_hint[1])
            center = 0.5 * (lo + hi)
            width = max(abs(hi - lo) / 3.0, 0.05)

        if eff in {"increase", "increasing"}:
            out = out + amp * sigmoid(z, center=center)
        elif eff in {"decrease", "decreasing"}:
            out = out - amp * sigmoid(z, center=center)
        elif eff in {"peak", "nonmonotone-peak"}:
            out = out + amp * gauss1d(z, mu=center, s=width)
        elif eff in {"valley", "nonmonotone-valley"}:
            out = out - amp * gauss1d(z, mu=center, s=width)

    for inter in (spec.get("interactions") or []):
        pair = inter.get("vars") or inter.get("pair") or inter.get("indices")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        idx_a = parse_dim(pair[0])
        idx_b = parse_dim(pair[1])
        if idx_a is None or idx_b is None:
            continue

        itype = str(inter.get("type", "synergy")).lower()
        sign = -1.0 if itype in {"antagonism", "tradeoff", "negative"} else 1.0
        strength = max(float(inter.get("scale", inter.get("strength", 0.0))), 0.0)
        conf = max(float(inter.get("confidence", 0.0)), 0.0)
        if strength == 0.0 and conf == 0.0:
            conf = 0.5
        amp = 0.2 * (strength if strength > 0 else 1.0) * conf
        term = grid[:, idx_a] * grid[:, idx_b]
        out = out + sign * amp * term

    for bump in (spec.get("bumps") or []):
        if not bump:
            continue
        mu_vals = bump.get("mu")
        if mu_vals is None:
            continue
        mu = np.array(list(mu_vals)[:d], dtype=np.float64)
        if mu.size == 0:
            mu = np.full(d, 0.5, dtype=np.float64)
        elif mu.size < d:
            mu = np.concatenate([mu, np.full(d - mu.size, 0.5, dtype=np.float64)])

        sigma_vals = bump.get("sigma", 0.15)
        if isinstance(sigma_vals, (list, tuple)):
            sigma = np.array(list(sigma_vals)[:d], dtype=np.float64)
            if sigma.size == 0:
                sigma = np.full(d, 0.15, dtype=np.float64)
            elif sigma.size < d:
                sigma = np.concatenate([sigma, np.full(d - sigma.size, sigma[-1], dtype=np.float64)])
        else:
            sigma = np.full(d, float(sigma_vals), dtype=np.float64)
        sigma = np.clip(sigma, 1e-6, None)

        amp = float(bump.get("amp", 0.1))
        diff = (grid - mu) / sigma
        gauss = np.exp(-0.5 * np.sum(diff ** 2, axis=1))
        out = out + amp * gauss

    return out.reshape(X1.shape)


def build_panels_from_specs(spec_list: List[Dict[str, Any]], X1, X2, truth) -> List[PriorPanel]:
    panels = []
    for spec in spec_list:
        surface = evaluate_prior_surface(spec["prior_schema"], X1, X2)
        panels.append(
            PriorPanel(
                title=spec["title"],
                description=spec["description"],
                surface=surface,
                stats=stats_against_truth(truth, surface),
            )
        )
    return panels


def build_data_aligned_prior(X1: np.ndarray, X2: np.ndarray, truth: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
    xs = X1[0]
    mean_x1 = truth.mean(axis=0)
    mean_x2 = truth.mean(axis=1)

    idx_peak_x1 = int(np.argmax(mean_x1))
    center_x1 = float(xs[idx_peak_x1])
    hint_x1 = [max(center_x1 - 0.18, 0.0), min(center_x1 + 0.18, 1.0)]
    idx_valley_x2 = int(np.argmin(mean_x2))
    center_x2 = float(xs[idx_valley_x2])
    hint_x2 = [max(center_x2 - 0.2, 0.0), min(center_x2 + 0.2, 1.0)]

    base_spec = {
        "effects": {
            "x1": {"effect": "peak", "scale": 0.95, "confidence": 0.95, "range_hint": hint_x1},
            "x2": {"effect": "valley", "scale": 0.95, "confidence": 0.95, "range_hint": hint_x2},
        },
        "interactions": [],
        "bumps": [],
    }

    base_surface = evaluate_prior_surface(base_spec, X1, X2)
    residual = truth - base_surface

    term = X1 * X2
    denom = float(np.sum(term ** 2) + 1e-12)
    beta = float(np.sum(residual * term) / denom)
    inter_type = "synergy" if beta >= 0 else "antagonism"
    strength = float(np.clip(abs(beta) / 0.15, 0.4, 5.0))
    base_spec["interactions"] = [
        {"vars": ["x1", "x2"], "type": inter_type, "scale": strength, "confidence": 0.98}
    ]
    residual = residual - beta * term

    flat = residual.ravel()
    idx_pos = int(np.argmax(flat))
    idx_neg = int(np.argmin(flat))
    bumps = []
    for idx in (idx_pos, idx_neg):
        coords = np.unravel_index(idx, residual.shape)
        amp = float(residual[coords])
        if abs(amp) < 1e-3:
            continue
        bumps.append(
            {
                "mu": [float(X1[coords]), float(X2[coords])],
                "sigma": [0.05, 0.05],
                "amp": amp,
            }
        )
    base_spec["bumps"] = bumps

    meta = {"x1_center": center_x1, "x2_center": center_x2, "interaction": inter_type}
    return base_spec, meta


def annotate_panel(ax: plt.Axes, description: str, stats: Stats) -> None:
    text = (
        f"{description}\n"
        rf"$\rho={stats.rho:.2f}$, $\alpha={stats.alpha:.2f}$"
    )
    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )


def render_storyboard(panels: List[PriorPanel], truth, X1, X2, save_path: str | None = None) -> None:
    global_min = min(truth.min(), *(p.surface.min() for p in panels))
    global_max = max(truth.max(), *(p.surface.max() for p in panels))
    norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = colormaps["viridis"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes_flat = axes.ravel()

    truth_ax = axes_flat[0]
    contour_levels = np.linspace(truth.min(), truth.max(), 30)
    im_truth = truth_ax.contourf(X1, X2, truth, levels=contour_levels, cmap=cmap, norm=norm)
    truth_ax.contour(X1, X2, truth, levels=contour_levels[::3], colors="k", linewidths=0.4, alpha=0.6)
    truth_ax.set_title("Ground truth")
    truth_ax.set_xlabel("$x_1$")
    truth_ax.set_ylabel("$x_2$")
    truth_ax.set_aspect("equal")
    truth_ax.text(
        0.02,
        0.95,
        "Benchmark Franke surface.",
        transform=truth_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    used_axes = []
    for ax, panel in zip(axes_flat[1:], panels):
        im = ax.contourf(X1, X2, panel.surface, levels=30, cmap=cmap, norm=norm)
        ax.contour(X1, X2, panel.surface, levels=10, colors="k", linewidths=0.3, alpha=0.4)
        ax.set_title(panel.title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
        annotate_panel(ax, panel.description, panel.stats)
        used_axes.append(ax)

    for ax in axes_flat[len(panels) + 1:]:
        ax.axis("off")

    plt.subplots_adjust(left=0.05, right=0.92, bottom=0.06, top=0.94, wspace=0.25, hspace=0.3)
    cax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
    fig.colorbar(im_truth, cax=cax, label="Surface value / prior mean")
    fig.suptitle("Prior readouts vs Franke surface — contrasting effects & data-aligned hybrid", fontsize=18)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render prior storyboard for the Franke surface.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g., figure.png).",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def main() -> None:
    args = parse_args()
    xs = np.linspace(0.0, 1.0, 220)
    X1, X2 = np.meshgrid(xs, xs)
    truth = franke_surface(X1, X2)
    prior_f_spec, prior_f_meta = build_data_aligned_prior(X1, X2, truth)

    panel_specs = [
        {
            "title": "Prior A — x₁ ↑, x₂ ↓",
            "description": "Two monotone effects (x₁↑, x₂↓)",
            "prior_schema": {
                "effects": {
                    "x1": {"effect": "increase", "scale": 0.95, "confidence": 0.9, "range_hint": [0.1, 0.9]},
                    "x2": {"effect": "decrease", "scale": 0.95, "confidence": 0.9, "range_hint": [0.2, 0.95]},
                },
                "interactions": [],
                "bumps": [],
            },
        },
        {
            "title": "Prior B — x₁ ↓, x₂ ↑",
            "description": "Roles flipped (x₁↓, x₂↑)",
            "prior_schema": {
                "effects": {
                    "x1": {"effect": "decrease", "scale": 0.95, "confidence": 0.9, "range_hint": [0.2, 0.95]},
                    "x2": {"effect": "increase", "scale": 0.95, "confidence": 0.9, "range_hint": [0.1, 0.9]},
                },
                "interactions": [],
                "bumps": [],
            },
        },
        {
            "title": "Prior C — synergy",
            "description": "x₁↓/x₂↑ with strong synergy",
            "prior_schema": {
                "effects": {
                    "x1": {"effect": "decrease", "scale": 0.6, "confidence": 0.85, "range_hint": [0.25, 0.9]},
                    "x2": {"effect": "increase", "scale": 0.6, "confidence": 0.85, "range_hint": [0.1, 0.8]},
                },
                "interactions": [
                    {"vars": ["x1", "x2"], "type": "synergy", "scale": 3.0, "confidence": 0.98}
                ],
                "bumps": [],
            },
        },
        {
            "title": "Prior D — peak/valley + bump",
            "description": "x₁ peak, x₂ valley + bump",
            "prior_schema": {
                "effects": {
                    "x1": {"effect": "peak", "scale": 0.95, "confidence": 0.9, "range_hint": [0.45, 0.75]},
                    "x2": {"effect": "valley", "scale": 0.95, "confidence": 0.9, "range_hint": [0.55, 0.85]},
                },
                "interactions": [],
                "bumps": [
                    {"mu": [0.8, 0.2], "sigma": [0.07, 0.05], "amp": 0.6}
                ],
            },
        },
        {
            "title": "Prior E — reversed peak/valley",
            "description": "x₁ valley, x₂ peak + bump",
            "prior_schema": {
                "effects": {
                    "x1": {"effect": "valley", "scale": 0.9, "confidence": 0.9, "range_hint": [0.25, 0.6]},
                    "x2": {"effect": "peak", "scale": 0.9, "confidence": 0.9, "range_hint": [0.2, 0.5]},
                },
                "interactions": [],
                "bumps": [
                    {"mu": [0.2, 0.8], "sigma": [0.05, 0.07], "amp": 0.6}
                ],
            },
        },
        {
            "title": "Prior F — data-aligned hybrid",
            "description": (
                f"x₁ peak≈{prior_f_meta['x1_center']:.2f}, x₂ valley≈{prior_f_meta['x2_center']:.2f}, "
                f"{prior_f_meta['interaction']} + dual bumps"
            ),
            "prior_schema": prior_f_spec,
        },
        {
            "title": "Prior G — low confidence",
            "description": "Same as Prior A but confidence=0.4",
            "prior_schema": {
                "effects": {
                    "x1": {"effect": "increase", "scale": 0.95, "confidence": 0.4, "range_hint": [0.1, 0.9]},
                    "x2": {"effect": "decrease", "scale": 0.95, "confidence": 0.4, "range_hint": [0.2, 0.95]},
                },
                "interactions": [],
                "bumps": [],
            },
        },
        {
            "title": "Prior H — high scale",
            "description": "Prior B but scale boosted (1.3)",
            "prior_schema": {
                "effects": {
                    "x1": {"effect": "decrease", "scale": 1.3, "confidence": 0.9, "range_hint": [0.2, 0.95]},
                    "x2": {"effect": "increase", "scale": 1.3, "confidence": 0.9, "range_hint": [0.1, 0.9]},
                },
                "interactions": [],
                "bumps": [],
            },
        },
    ]

    panels = build_panels_from_specs(panel_specs, X1, X2, truth)
    render_storyboard(panels, truth, X1, X2, save_path=args.output)


if __name__ == "__main__":
    main()


#%%
