from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from readout_schema import readout_to_prior

if TYPE_CHECKING:
    from main_benchmark import LookupTable


def _format_method_label(method: str) -> str:
    if method.startswith("hybrid_rho_"):
        tail = method[len("hybrid_rho_"):]
        fade_part = None
        if "_fade_" in tail:
            tail, fade_part = tail.split("_fade_", 1)
        try:
            rho_val = float(tail)
            rho_label = rf"$\\rho = {rho_val:+.2f}$"
        except ValueError:
            rho_label = rf"$\\rho = {tail}$"
        if fade_part is not None:
            try:
                fade_val = float(fade_part)
                return f"{rho_label}, fade={fade_val:.2f}"
            except ValueError:
                return f"{rho_label}, fade={fade_part}"
        return rho_label
    return method


def plot_runs_mean_lookup(hist_df: pd.DataFrame, *, methods: Optional[List[str]] = None,
                          ci: str = "sd", ax: Optional[plt.Axes] = None,
                          title: Optional[str] = None,
                          xlabel: str = "Iteration",
                          ylabel: str = "Best so far",
                          legend_title: Optional[str] = None) -> plt.Axes:
    """Plot mean best-so-far vs iteration across seeds with a shaded uncertainty band."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))

    df = hist_df.copy()
    df = df[df["iter"] >= 0]

    if methods is None:
        methods = list(df["method"].unique())

    legend_handles = []
    legend_labels = []
    for m in methods:
        d = df[df["method"] == m]
        if d.empty:
            continue
        agg = d.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:
            err = agg["std"]
        x = agg["iter"].to_numpy()
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        line, = ax.plot(x, y, linewidth=2.0)
        ax.fill_between(x, y - e, y + e, alpha=0.18, color=line.get_color())
        legend_handles.append(line)
        legend_labels.append(_format_method_label(m))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_facecolor("white")
    if ax.figure:
        ax.figure.patch.set_facecolor("white")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, frameon=False, title=legend_title,
                  fontsize=11, title_fontsize=11, loc="best")
    plt.tight_layout()
    return ax


def benchmark_alignment_priors(
        lookup: 'LookupTable',
        target_alignments: Union[float, str, Sequence[Union[float, str]]],
        *,
        n_init: int = 6,
        n_iter: int = 25,
        seed: int = 0,
        repeats: int = 1,
        init_method: str = "random",
        generator_kwargs: Optional[Dict[str, Any]] = None,
        diagnose_prior: bool = False,
        plot: bool = True,
        ci: str = "sd",
        ax: Optional[plt.Axes] = None,
        prior_fade: float = 1.0,
) -> Tuple[pd.DataFrame, Optional[plt.Axes]]:
    """Run hybrid BO with synthetic readouts that target specific alignment values."""
    from main_benchmark import (
        _format_alignment_label,
        _resolve_targets,
        _stable_seed_from_target,
        _run_hybrid_lookup_single,
        build_gp_alignment_template,
        generate_gp_alignment_readout,
    )

    targets = _resolve_targets(target_alignments)

    template_kwargs: Dict[str, Any] = {}
    scale_kwargs: Dict[str, Any] = {}
    alias_map = {
        "n_signal": "n_signal_support",
        "n_noise": "n_noise_support",
    }
    if generator_kwargs:
        template_keys = {"n_signal_support", "n_noise_support", "signal_sigma", "noise_sigma"}
        scale_keys = {"scale_sig_max", "scale_noise_max", "scale_sig_steps", "scale_noise_steps"}
        for k, v in generator_kwargs.items():
            key = alias_map.get(k, k)
            if key in template_keys:
                template_kwargs[key] = v
            elif key in scale_keys:
                scale_kwargs[key] = v
            else:
                raise ValueError(
                    f"Unknown generator_kwargs key '{k}'. Expected template keys {template_keys} or scale keys {scale_keys}."
                )

    dfs: List[pd.DataFrame] = []
    method_order: List[str] = []
    alignment_meta: List[Dict[str, Any]] = []
    prior_debug_runs: List[Dict[str, Any]] = []

    for target in targets:
        label = _format_alignment_label(target)
        method_label = f"hybrid_rho_{label}"
        if method_label not in method_order:
            method_order.append(method_label)
        base_seed_offset = _stable_seed_from_target(target) % 1_000_000
        base_seed = int(seed + base_seed_offset)
        template_cache_local = build_gp_alignment_template(lookup, **template_kwargs)
        for r in range(repeats):
            bo_seed = int(base_seed + r)
            ro, info = generate_gp_alignment_readout(
                lookup,
                target,
                template_cache=template_cache_local,
                **scale_kwargs,
            )
            dfr = _run_hybrid_lookup_single(
                lookup,
                n_init=n_init,
                n_iter=n_iter,
                seed=bo_seed,
                init_method=init_method,
                readout_source="custom",
                pool_base=None,
                debug_llm=False,
                model="",
                log_json_path=None,
                diagnose_prior=diagnose_prior,
                prompt_profile="perfect",
                method_tag=method_label,
                custom_readout=ro,
                custom_readout_is_normalized=True,
                prior_fade=prior_fade,
            )
            dfr["seed"] = bo_seed
            dfr["alignment_label"] = label
            dfr["alignment_target"] = info["target_rho"]
            dfr["alignment_achieved"] = info["achieved_rho"]
            dfr["alignment_requested"] = target
            dfs.append(dfr)

            meta_entry = {
                "method": method_label,
                "alignment_label": label,
                "target_requested": target,
                "seed": bo_seed,
                **info,
            }
            alignment_meta.append(meta_entry)
            if diagnose_prior and "prior_debug" in dfr.attrs:
                prior_debug_runs.append({
                    "method": method_label,
                    "seed": bo_seed,
                    "alignment_label": label,
                    "prior_debug": dfr.attrs["prior_debug"],
                })

    if not dfs:
        raise RuntimeError("No runs were executed; check target_alignments/repeats arguments.")

    hist = pd.concat(dfs, ignore_index=True)
    if alignment_meta:
        hist.attrs["alignment_readout_info"] = alignment_meta
    if prior_debug_runs:
        hist.attrs["prior_debug_runs"] = prior_debug_runs

    ax_out = None
    if plot:
        ax_out = plot_runs_mean_lookup(hist, methods=method_order, ci=ci, ax=ax)
    return hist, ax_out


def benchmark_prior_fade(
        lookup: 'LookupTable',
        *,
        alignment: Union[float, str] = 0.0,
        fade_values: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        n_init: int = 6,
        n_iter: int = 25,
        seed: int = 0,
        repeats: int = 1,
        init_method: str = "random",
        generator_kwargs: Optional[Dict[str, Any]] = None,
        diagnose_prior: bool = False,
        plot: bool = True,
        ci: str = "sd",
        ax: Optional[plt.Axes] = None,
) -> Tuple[pd.DataFrame, Optional[plt.Axes]]:
    """Fix an alignment target and sweep prior_fade values."""
    from main_benchmark import (
        _format_alignment_label,
        _resolve_targets,
        _stable_seed_from_target,
        _run_hybrid_lookup_single,
        build_gp_alignment_template,
        generate_gp_alignment_readout,
    )

    fades = [float(max(0.0, min(1.0, f))) for f in fade_values]
    targets = _resolve_targets([alignment])

    template_kwargs: Dict[str, Any] = {}
    scale_kwargs: Dict[str, Any] = {}
    alias_map = {
        "n_signal": "n_signal_support",
        "n_noise": "n_noise_support",
    }
    if generator_kwargs:
        template_keys = {"n_signal_support", "n_noise_support", "signal_sigma", "noise_sigma"}
        scale_keys = {"scale_sig_max", "scale_noise_max", "scale_sig_steps", "scale_noise_steps"}
        for k, v in generator_kwargs.items():
            key = alias_map.get(k, k)
            if key in template_keys:
                template_kwargs[key] = v
            elif key in scale_keys:
                scale_kwargs[key] = v
            else:
                raise ValueError(
                    f"Unknown generator_kwargs key '{k}'. Expected template keys {template_keys} or scale keys {scale_keys}."
                )

    dfs: List[pd.DataFrame] = []
    meta: List[Dict[str, Any]] = []
    target = targets[0]
    label = _format_alignment_label(target)
    base_seed = _stable_seed_from_target(target) % 1_000_000 + int(seed)
    template_cache = build_gp_alignment_template(lookup, **template_kwargs)
    method_order: List[str] = []

    for fade in fades:
        method_label = f"hybrid_rho_{label}_fade_{fade:.2f}"
        method_order.append(method_label)
        for r in range(repeats):
            bo_seed = int(base_seed + r)
            ro, info = generate_gp_alignment_readout(
                lookup,
                target,
                template_cache=template_cache,
                **scale_kwargs,
            )
            dfr = _run_hybrid_lookup_single(
                lookup,
                n_init=n_init,
                n_iter=n_iter,
                seed=bo_seed,
                init_method=init_method,
                readout_source="custom",
                pool_base=None,
                debug_llm=False,
                model="",
                log_json_path=None,
                diagnose_prior=diagnose_prior,
                prompt_profile="perfect",
                method_tag=method_label,
                custom_readout=ro,
                custom_readout_is_normalized=True,
                prior_fade=fade,
            )
            dfr["seed"] = bo_seed
            dfr["alignment_label"] = label
            dfr["alignment_target"] = info["target_rho"]
            dfr["alignment_achieved"] = info["achieved_rho"]
            dfr["prior_fade"] = fade
            dfs.append(dfr)
            meta.append({
                "method": method_label,
                "target_requested": target,
                "alignment_label": label,
                "prior_fade": fade,
                "seed": bo_seed,
                **info,
            })

    hist = pd.concat(dfs, ignore_index=True)
    hist.attrs["alignment_readout_info"] = meta

    ax_out = None
    if plot:
        legend_title = rf"{_format_method_label(f'hybrid_rho_{label}')}"
        ax_out = plot_runs_mean_lookup(
            hist,
            methods=method_order,
            ci=ci,
            ax=ax,
            xlabel="Iteration",
            ylabel="Best so far",
            legend_title=legend_title,
        )
    return hist, ax_out


def plot_alpha_trajectories(alpha_df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
                            title: Optional[str] = None,
                            ylabel: str = "Feature weight",
                            truth_weights: Optional[pd.DataFrame] = None,
                            truth_use_column: str = "rho") -> plt.Axes:
    if alpha_df is None or alpha_df.empty:
        raise ValueError("alpha_df must be a non-empty DataFrame with columns ['iter','feature','alpha'].")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
    if "feature_display" in alpha_df.columns:
        plot_df = alpha_df.rename(columns={"feature_display": "feature_label"})
    else:
        plot_df = alpha_df.assign(feature_label=alpha_df["feature"])
    features = sorted(plot_df["feature_label"].unique())
    for feat in features:
        d = plot_df[plot_df["feature_label"] == feat]
        ax.plot(d["iter"], d["alpha"], label=feat)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_facecolor("white")
    if ax.figure:
        ax.figure.patch.set_facecolor("white")
    ax.legend(frameon=False, fontsize=10, title="Feature", loc="best")
    if truth_weights is not None and not truth_weights.empty:
        col = truth_use_column if truth_use_column in truth_weights.columns else None
        rows = []
        for feat in features:
            match = truth_weights[truth_weights["feature"] == feat]
            if match.empty:
                continue
            label = feat
            if col:
                val = match.iloc[0][col]
                rows.append(f"{label}: ρ={val:+.2f}")
            else:
                rows.append(f"{label}: ρ={match.iloc[0]['rho']:+.2f}")
        if rows:
            text = "\n".join(rows)
            ax.text(1.02, 0.5, text, transform=ax.transAxes, fontsize=10,
                    va="center", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    plt.tight_layout()
    return ax


def plot_alignment_residual_heatmaps(lookup: 'LookupTable',
                                     target_alignments: Sequence[float],
                                     *,
                                     generator_kwargs: Optional[Dict[str, Any]] = None,
                                     template_cache: Optional[Dict[str, Any]] = None,
                                     cmap: str = "coolwarm",
                                     n_levels: int = 16,
                                     feature_x: str = "x1",
                                     feature_y: str = "x2",
                                     fixed_values: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
                                     fixed_tolerance: float = 2.0) -> plt.Figure:
    """Plot residual heatmaps (truth minus prior mean) on a 2D slice defined by feature_x and feature_y."""
    from main_benchmark import (
        _resolve_feature_index,
        build_gp_alignment_template,
        generate_gp_alignment_readout,
    )

    alias_map = {
        "n_signal": "n_signal_support",
        "n_noise": "n_noise_support",
    }
    template_kwargs: Dict[str, Any] = {}
    scale_kwargs: Dict[str, Any] = {}
    if generator_kwargs:
        template_keys = {"n_signal_support", "n_noise_support", "signal_sigma", "noise_sigma"}
        scale_keys = {"scale_sig_max", "scale_noise_max", "scale_sig_steps", "scale_noise_steps"}
        for k, v in generator_kwargs.items():
            key = alias_map.get(k, k)
            if key in template_keys:
                template_kwargs[key] = v
            elif key in scale_keys:
                scale_kwargs[key] = v
            else:
                raise ValueError(f"Unknown generator_kwargs key '{k}'.")

    if template_cache is None:
        template_cache = build_gp_alignment_template(lookup, **template_kwargs)

    feat_names = lookup.feature_names or [f"x{i+1}" for i in range(lookup.d)]
    idx_x = _resolve_feature_index(feature_x, feat_names)
    idx_y = _resolve_feature_index(feature_y, feat_names)

    mask = torch.ones(lookup.n, dtype=torch.bool, device=lookup.X_raw.device)
    if fixed_values:
        for name, spec in fixed_values.items():
            idx = _resolve_feature_index(name, feat_names)
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                target, tol = spec
            else:
                target, tol = spec, fixed_tolerance
            mask &= torch.abs(lookup.X_raw[:, idx] - float(target)) <= float(tol)
    mask_np = mask.detach().cpu().numpy()
    if mask_np.sum() < 10:
        raise ValueError("Too few rows match the specified slice; adjust fixed_values or tolerance.")

    targets = list(target_alignments)
    n_cols = len(targets)
    if n_cols == 0:
        raise ValueError("target_alignments must be non-empty.")
    residual_payload = []
    max_abs = 0.0
    x_vals = lookup.X_raw[:, idx_x].detach().cpu().numpy()[mask_np]
    y_vals = lookup.X_raw[:, idx_y].detach().cpu().numpy()[mask_np]
    triang = Triangulation(x_vals, y_vals)
    for rho_target in targets:
        ro, info = generate_gp_alignment_readout(
            lookup,
            rho_target,
            template_cache=template_cache,
            **scale_kwargs,
        )
        prior = readout_to_prior(ro)
        vals_prior = prior.m0_torch(lookup.X).detach().cpu().numpy()
        residual_full = lookup.y.detach().cpu().numpy() - vals_prior
        residual = residual_full[mask_np]
        max_abs = max(max_abs, float(np.max(np.abs(residual))))
        residual_payload.append((rho_target, info, residual))
    max_abs = max(max_abs, 1e-6)
    levels = np.linspace(-max_abs, max_abs, n_levels)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.3 * n_cols, 4.0), sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, payload in zip(axes, residual_payload):
        rho_target, info, residual = payload
        heat = ax.tricontourf(triang, residual, levels=levels, cmap=cmap)
        ax.tricontour(triang, residual, levels=levels, colors="black", linewidths=0.4, alpha=0.5)
        ax.set_title(rf"target ρ={rho_target:+.2f}" "\n" rf"achieved={info['achieved_rho']:+.2f}", fontsize=11)
        ax.set_xlabel(lookup.feature_names[0])
        ax.set_ylabel(lookup.feature_names[1])
        ax.grid(False)

    cbar = fig.colorbar(heat, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Residual (truth − prior)", fontsize=11)
    plt.tight_layout()
    return fig


def build_p3ht_feature_table(csv_path: str = "P3HT_dataset.csv") -> pd.DataFrame:
    """Summarize P3HT lookup table features with roles, units, ranges, and types."""
    df = pd.read_csv(csv_path)
    meta = [
        ("P3HT content (%)", "Host polymer fraction (wt%)", "%", "continuous input"),
        ("D1 content (%)", "Additive D1 loading (wt%)", "%", "continuous input"),
        ("D2 content (%)", "Additive D2 loading (wt%)", "%", "continuous input"),
        ("D6 content (%)", "Additive D6 loading (wt%)", "%", "continuous input"),
        ("D8 content (%)", "Additive D8 loading (wt%)", "%", "continuous input"),
        ("Conductivity", "Target conductivity", "S/cm", "continuous target"),
    ]
    rows = []
    for name, role, units, var_type in meta:
        if name not in df.columns:
            raise ValueError(f"Column '{name}' not found in {csv_path}.")
        s = df[name]
        rng = f"{float(s.min()):.2f} – {float(s.max()):.2f}"
        rows.append(
            {
                "Parameter": name,
                "Role": role,
                "Units": units,
                "Range": rng,
                "Type": var_type,
            }
        )
    return pd.DataFrame(rows)


def write_table_files(table_df: pd.DataFrame, *, csv_path: Optional[str] = None, md_path: Optional[str] = None) -> None:
    """Write table to CSV and/or Markdown (Markdown written without extra dependencies)."""
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        table_df.to_csv(csv_path, index=False)
    if md_path:
        os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
        headers = list(table_df.columns)
        lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, row in table_df.iterrows():
            lines.append("|" + "|".join(str(row[h]) for h in headers) + "|")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    from main_benchmark import (
        _prepare_readout_with_effects,
        _select_initial_indices_lookup,
        compare_hybrid_perfect_vs_baselines,
        estimate_truth_feature_weights,
        get_named_readout,
        invert_readout,
        load_lookup_csv,
        measure_readout_alignment,
        run_hybrid_lookup,
        run_hybrid_plus_lookup,
    )

    lt = load_lookup_csv("P3HT_dataset.csv", impute_features="median")
    # hp_hist = run_hybrid_plus_lookup(
    #     lt,
    #     n_init=6,
    #     n_iter=30,
    #     seed=11,
    #     readout_source="llm",
    #     prompt_profile="bad",
    #     prior_fade=0.5,
    # )
    # alpha_df = hp_hist.attrs.get("alpha_history")
    # if alpha_df is not None:
    #     truth_weights = estimate_truth_feature_weights(lt)
    #     plot_alpha_trajectories(alpha_df, title="Adaptive feature weights (hybrid+)",
    #                             truth_weights=truth_weights, truth_use_column="rho")
    #     print(truth_weights.sort_values("abs_rho", ascending=False))

    # """Benchmarking the alignment and the alpha (fade)"""
    # hist, ax = benchmark_alignment_priors(
    #     lt,
    #     target_alignments=[0, 0.2, 0.4, 0.8],
    #     n_init=6,
    #     n_iter=20,
    #     repeats=3,
    #     generator_kwargs={"n_signal": 3, "n_noise": 3},
    #     plot=True,
    #     prior_fade=0.2,
    # )
    # print(hist.attrs["alignment_readout_info"][0])

    # hist_fade, ax = benchmark_prior_fade(
    #     lt,
    #     alignment=0,
    #     fade_values=[0, 0.6, 0.8],
    #     n_init=6,
    #     n_iter=20,
    #     repeats=3,
    #     generator_kwargs={"n_signal": 3, "n_noise": 3},
    #     plot=True,
    # )

    # # Comparison: hybrid (fixed prior) vs hybrid_plus on the same static readout (with repeats)
    # n_init_demo = 3
    # seed_demo = 4653
    # repeats_demo = 3
    # perfect_ro = get_named_readout("perfect")
    # bad_ro = invert_readout(perfect_ro)
    # bad_ro_aug, _, _ = _prepare_readout_with_effects(bad_ro, lt, is_normalized=False)
    # bad_alignment = measure_readout_alignment(bad_ro_aug, lt)
    # print(f"Worst-case readout (inverted perfect) alignment ≈ {bad_alignment:+.3f}")

    # histories = []
    # for r in range(repeats_demo):
    #     seed_r = seed_demo + r
    #     shared_init = _select_initial_indices_lookup(lt, n_init_demo, seed_r, init_method="random")
    #     hybrid_fixed = run_hybrid_lookup(
    #         lt,
    #         n_init=n_init_demo,
    #         n_iter=30,
    #         seed=seed_r,
    #         readout_source="flat",
    #         prompt_profile="bad",
    #         custom_readout=bad_ro_aug,
    #         custom_readout_is_normalized=True,
    #         method_tag="Bad Prior (Fixed)",
    #         prior_fade=1.0,
    #         initial_indices=shared_init,
    #     )
    #     hybrid_fixed["seed"] = seed_r

    #     hybrid_plus = run_hybrid_plus_lookup(
    #         lt,
    #         n_init=n_init_demo,
    #         n_iter=30,
    #         seed=seed_r,
    #         readout_source="flat",
    #         custom_readout=bad_ro_aug,
    #         custom_readout_is_normalized=True,
    #         base_prior_weight=1.0,
    #         method_tag="Bad Prior w/ Self-Recovery",
    #         initial_indices=shared_init,
    #     )
    #     hybrid_plus["seed"] = seed_r
    #     histories.extend([hybrid_fixed, hybrid_plus])

    # hist_compare = pd.concat(histories, ignore_index=True)
    # plot_runs_mean_lookup(hist_compare,
    #                       methods=["Bad Prior (Fixed)", "Bad Prior w/ Self-Recovery"],
    #                       title="Bad Prior vs Self-Recovery",
    #                       ci="sem")

    # Hybrid vs baselines comparison using the perfect prompt prior
    compare_hybrid_perfect_vs_baselines(
        lt,
        n_init=6,
        n_iter=25,
        repeats=3,
        plot=True,
    )
    plt.show()

    slice_values = {
        "D2 content (%)": (0.2, 2.0),
        "D6 content (%)": (0.2, 2.0),
        "D8 content (%)": (0.2, 2.0),
    }
    fig = plot_alignment_residual_heatmaps(
        lt,
        target_alignments=[0.0, 0.2, 0.4, 0.8],
        generator_kwargs={"n_signal_support": 32, "n_noise_support": 16},
        feature_x="P3HT content (%)",
        feature_y="D1 content (%)",
        fixed_values=slice_values,
        fixed_tolerance=2.0,
    )
    plt.show()
