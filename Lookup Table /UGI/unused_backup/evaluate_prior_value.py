#!/usr/bin/env python3
"""
evaluate_prior_value.py
=======================

Proper evaluation of prior value for Bayesian Optimization.

Research Question: "How many BO samples does this prior replace?"

Approach:
1. Run BO with prior → measure performance curve
2. Run BO without prior (vanilla) → measure performance curve
3. Compare: At iteration N, prior-BO reaches performance P
            At what iteration does vanilla BO reach performance P?
4. Answer: Prior saved (vanilla_iters - prior_iters) samples

This directly measures the value of injected knowledge in terms of
sample efficiency, which is what you actually care about for the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize

from prior_gp import Prior, GPWithPriorMean, fit_residual_gp
from prompt_generator import load_lookup_csv


@dataclass
class BOResult:
    """Results from a single BO run."""
    history: pd.DataFrame
    best_found: float
    iterations_to_threshold: Dict[float, int]
    prior_used: Optional[str]


def run_bo_on_lookup(
    lookup,
    *,
    prior: Optional[Prior] = None,
    n_init: int = 6,
    budget: int = 50,
    seed: int = 0,
    acq_function: str = "ei",  # "ei" or "ucb"
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> BOResult:
    """
    Run Bayesian Optimization on a lookup table.

    Args:
        lookup: LookupTable with X, y, etc.
        prior: Optional Prior to bias the GP (this is your injected knowledge!)
        n_init: Number of random initial samples
        budget: Total optimization budget
        seed: Random seed
        acq_function: "ei" (Expected Improvement) or "ucb" (Upper Confidence Bound)

    Returns:
        BOResult with performance curve and metadata
    """
    rng = np.random.RandomState(seed)
    n_total = int(lookup.n)

    # Random initialization
    init_indices = rng.choice(n_total, size=n_init, replace=False)
    seen = set(init_indices.tolist())

    X_obs = lookup.X[init_indices].to(device=device, dtype=dtype)
    y_obs = lookup.y[init_indices].to(device=device, dtype=dtype).unsqueeze(-1)

    history = []
    best_so_far = float(y_obs.max().item())

    # Record initial samples
    for i, idx in enumerate(init_indices):
        history.append({
            'iteration': i - n_init,
            'idx': int(idx),
            'y': float(lookup.y[idx].item()),
            'best_so_far': float(y_obs[:i+1].max().item()),
            'phase': 'init',
        })

    # BO loop
    for t in range(budget):
        # Fit GP (with or without prior)
        if prior is not None:
            # Use prior-biased GP
            gp_base = SingleTaskGP(X_obs, y_obs, outcome_transform=Standardize(m=1)).to(device)
            mll = ExactMarginalLogLikelihood(gp_base.likelihood, gp_base)
            fit_gpytorch_mll(mll)

            # Fit residual GP and get alignment
            gp_residual, alpha = fit_residual_gp(X_obs, y_obs.squeeze(-1), prior)
            gp = GPWithPriorMean(gp_residual, prior, m0_scale=alpha)
        else:
            # Vanilla GP (no prior)
            gp = SingleTaskGP(X_obs, y_obs, outcome_transform=Standardize(m=1)).to(device)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        # Get unseen candidates
        remaining = list(set(range(n_total)) - seen)
        if len(remaining) == 0:
            break

        X_candidates = lookup.X[remaining].to(device=device, dtype=dtype)

        # Select next point via acquisition function
        with torch.no_grad():
            if acq_function == "ei":
                acq = ExpectedImprovement(gp, best_f=y_obs.max())
                acq_values = acq(X_candidates.unsqueeze(1))
            else:  # ucb
                acq = UpperConfidenceBound(gp, beta=2.0)
                acq_values = acq(X_candidates.unsqueeze(1))

        best_candidate_idx = int(torch.argmax(acq_values).item())
        selected_idx = remaining[best_candidate_idx]

        # Observe
        seen.add(selected_idx)
        x_new = lookup.X[selected_idx].to(device=device, dtype=dtype).unsqueeze(0)
        y_new = lookup.y[selected_idx].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)

        X_obs = torch.cat([X_obs, x_new], dim=0)
        y_obs = torch.cat([y_obs, y_new], dim=0)

        best_so_far = float(y_obs.max().item())

        history.append({
            'iteration': t,
            'idx': int(selected_idx),
            'y': float(y_new.item()),
            'best_so_far': best_so_far,
            'phase': 'optimize',
            'acq_value': float(acq_values[best_candidate_idx].item()),
        })

    hist_df = pd.DataFrame(history)

    # Compute: iterations needed to reach various thresholds
    y_max_global = float(lookup.y.max().item())
    thresholds = {
        0.5: y_max_global * 0.5,
        0.7: y_max_global * 0.7,
        0.8: y_max_global * 0.8,
        0.9: y_max_global * 0.9,
        0.95: y_max_global * 0.95,
    }

    iterations_to_threshold = {}
    for frac, threshold in thresholds.items():
        reached = hist_df[hist_df['best_so_far'] >= threshold]
        if len(reached) > 0:
            iterations_to_threshold[frac] = int(reached['iteration'].min())
        else:
            iterations_to_threshold[frac] = budget + n_init  # Did not reach

    return BOResult(
        history=hist_df,
        best_found=best_so_far,
        iterations_to_threshold=iterations_to_threshold,
        prior_used="prior" if prior is not None else "vanilla",
    )


def compute_sample_efficiency_gain(
    result_with_prior: BOResult,
    result_vanilla: BOResult,
) -> Dict[str, float]:
    """
    Compute how many samples the prior saved.

    Returns:
        Dictionary with sample efficiency gains at different thresholds.
    """
    gains = {}

    for frac in [0.5, 0.7, 0.8, 0.9, 0.95]:
        iters_prior = result_with_prior.iterations_to_threshold.get(frac, float('inf'))
        iters_vanilla = result_vanilla.iterations_to_threshold.get(frac, float('inf'))

        if iters_prior < float('inf') and iters_vanilla < float('inf'):
            samples_saved = iters_vanilla - iters_prior
            gains[f'samples_saved_{int(frac*100)}pct'] = int(samples_saved)
        else:
            gains[f'samples_saved_{int(frac*100)}pct'] = None

    # Overall score: average savings across thresholds
    valid_gains = [v for v in gains.values() if v is not None]
    if valid_gains:
        gains['mean_samples_saved'] = float(np.mean(valid_gains))
    else:
        gains['mean_samples_saved'] = 0.0

    return gains


def run_prior_value_experiment(
    csv_path: str = "ugi_merged_dataset.csv",
    prior: Optional[Prior] = None,
    prior_name: str = "test_prior",
    n_init: int = 6,
    budget: int = 100,
    n_trials: int = 5,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Run complete experiment comparing prior-BO vs vanilla BO.

    Args:
        csv_path: Path to dataset
        prior: Prior to test (or None for vanilla)
        prior_name: Name for logging
        n_init: Initial random samples
        budget: Optimization budget
        n_trials: Number of random seeds to average over
        seeds: Optional explicit seed list

    Returns:
        DataFrame with results for each trial
    """
    if seeds is None:
        seeds = list(range(n_trials))

    lookup = load_lookup_csv(csv_path, objective_col="yield")

    results = []

    for seed in seeds:
        print(f"  Trial {seed+1}/{n_trials} (seed={seed})...")

        # Run with prior
        result_prior = run_bo_on_lookup(
            lookup,
            prior=prior,
            n_init=n_init,
            budget=budget,
            seed=seed,
        )

        # Run vanilla (for comparison)
        result_vanilla = run_bo_on_lookup(
            lookup,
            prior=None,
            n_init=n_init,
            budget=budget,
            seed=seed,
        )

        # Compute gains
        gains = compute_sample_efficiency_gain(result_prior, result_vanilla)

        results.append({
            'prior_name': prior_name,
            'seed': seed,
            'best_found_prior': result_prior.best_found,
            'best_found_vanilla': result_vanilla.best_found,
            **gains,
        })

    return pd.DataFrame(results)


def compare_knowledge_sources(
    csv_path: str = "ugi_merged_dataset.csv",
    n_init: int = 6,
    budget: int = 100,
    n_trials: int = 5,
) -> pd.DataFrame:
    """
    Compare different knowledge sources:
    1. Vanilla BO (no prior)
    2. Random prior (noise baseline)
    3. Hand-crafted priors from bo_readout_prompts.py
    4. Cartographer-generated priors (20, 50, 100 samples)

    This directly tests your research question!
    """
    from prompt_generator import score_prompt_via_llm
    from bo_readout_prompts import (
        SYS_PROMPTS_BEST,
        SYS_PROMPTS_GOOD,
        SYS_PROMPTS_MEDIUM,
        SYS_PROMPTS_BAD,
    )
    from readout_schema import readout_to_prior

    print("="*70)
    print("KNOWLEDGE SOURCE COMPARISON")
    print("="*70)
    print(f"Configuration: n_init={n_init}, budget={budget}, trials={n_trials}\n")

    all_results = []

    # 1. Vanilla baseline
    print("1. Testing: Vanilla BO (no prior)")
    results_vanilla = run_prior_value_experiment(
        csv_path=csv_path,
        prior=None,
        prior_name="vanilla",
        n_init=n_init,
        budget=budget,
        n_trials=n_trials,
    )
    all_results.append(results_vanilla)

    # 2. Hand-crafted priors
    prompt_library = {
        "best": SYS_PROMPTS_BEST,
        "good": SYS_PROMPTS_GOOD,
        "medium": SYS_PROMPTS_MEDIUM,
        "bad": SYS_PROMPTS_BAD,
    }

    for name, prompt in prompt_library.items():
        print(f"\n2. Testing: Hand-crafted prior '{name}'")
        # Generate prior from prompt
        result = score_prompt_via_llm(prompt=prompt, csv_path=csv_path)
        prior = readout_to_prior(result['readout_unit'], feature_names=["x1", "x2", "x3", "x4"])

        results = run_prior_value_experiment(
            csv_path=csv_path,
            prior=prior,
            prior_name=f"handcraft_{name}",
            n_init=n_init,
            budget=budget,
            n_trials=n_trials,
        )
        all_results.append(results)

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)

    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    summary = combined.groupby('prior_name').agg({
        'best_found_prior': ['mean', 'std'],
        'mean_samples_saved': ['mean', 'std'],
        'samples_saved_90pct': ['mean', 'std'],
    }).round(4)
    print(summary)

    # Save detailed results
    combined.to_csv('prior_value_results.csv', index=False)
    print("\nDetailed results saved to: prior_value_results.csv")

    return combined


def plot_knowledge_value_curve(results_df: pd.DataFrame, save_path: str = "knowledge_value_curve.png"):
    """
    Plot the key figure for your paper:
    X-axis: Knowledge source (vanilla, bad, medium, good, best, human)
    Y-axis: Sample efficiency gain (samples saved)

    This visualizes: "How much is each knowledge source worth?"
    """
    import matplotlib.pyplot as plt

    # Aggregate by prior type
    summary = results_df.groupby('prior_name').agg({
        'mean_samples_saved': ['mean', 'std'],
        'samples_saved_90pct': ['mean', 'std'],
    })

    prior_names = summary.index.tolist()
    mean_savings = summary[('mean_samples_saved', 'mean')].values
    std_savings = summary[('mean_samples_saved', 'std')].values

    # Sort by performance
    sorted_indices = np.argsort(mean_savings)
    prior_names = [prior_names[i] for i in sorted_indices]
    mean_savings = mean_savings[sorted_indices]
    std_savings = std_savings[sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(prior_names))
    bars = ax.bar(x_pos, mean_savings, yerr=std_savings, capsize=5, alpha=0.7)

    # Color bars by performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(prior_names)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Knowledge Source', fontsize=12)
    ax.set_ylabel('Sample Efficiency Gain\n(BO Samples Saved)', fontsize=12)
    ax.set_title('Value of Injected Knowledge for Bayesian Optimization', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prior_names, rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

    # Annotate values
    for i, (x, y) in enumerate(zip(x_pos, mean_savings)):
        ax.text(x, y + std_savings[i] + 1, f'{y:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nKnowledge value curve saved to: {save_path}")
    return fig


if __name__ == "__main__":
    import sys

    # Run comprehensive comparison
    print("Running knowledge source comparison...")
    print("This will take several minutes (5 trials × 6 conditions × 100 BO iterations)\n")

    results = compare_knowledge_sources(
        csv_path="ugi_merged_dataset.csv",
        n_init=6,
        budget=100,
        n_trials=5,
    )

    # Plot
    plot_knowledge_value_curve(results)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nKey files generated:")
    print("  - prior_value_results.csv: Detailed results")
    print("  - knowledge_value_curve.png: Main figure for paper")
    print("\nThis directly measures: 'How many BO samples does each prior replace?'")
