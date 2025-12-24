#!/usr/bin/env python3
"""
improved_baselines.py
=====================

Additional BO baselines beyond vanilla EI:
- GP-UCB (Upper Confidence Bound)
- Thompson Sampling
- qEI (batch Expected Improvement)
- Probability of Improvement (PI)

These are essential for fair comparison in BO literature.

Usage:
    from improved_baselines import run_gp_ucb_lookup, run_thompson_sampling_lookup

    hist_ucb = run_gp_ucb_lookup(lookup, n_init=6, n_iter=50, beta=2.0)
    hist_ts = run_thompson_sampling_lookup(lookup, n_init=6, n_iter=50)
"""

import torch
from torch import Tensor
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    qExpectedImprovement,
)
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood


# Import from main_benchmark for compatibility
try:
    from main_benchmark import (
        LookupTable,
        remaining_indices,
        as_records_row,
        _select_initial_indices_lookup,
        _build_initial_from_indices,
        DEVICE,
        DTYPE,
    )
except ImportError:
    # Fallback for standalone use
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    print("Warning: Could not import from main_benchmark, using fallback")


def run_gp_ucb_lookup(
    lookup: "LookupTable",
    *,
    n_init: int = 6,
    n_iter: int = 50,
    seed: int = 0,
    beta: float = 2.0,  # Exploration parameter
    init_method: str = "random",
    adaptive_beta: bool = True,
) -> pd.DataFrame:
    """
    GP-UCB (Upper Confidence Bound) baseline.

    Reference: Srinivas et al. (2010) "Gaussian Process Optimization in the Bandit Setting"

    UCB(x) = μ(x) + β * σ(x)

    Args:
        beta: Exploration-exploitation trade-off. Higher β → more exploration
               Typical values: 2.0 (balanced), 3.0 (exploratory), 1.0 (greedy)
        adaptive_beta: Use time-dependent β_t = sqrt(2 * log(t * d * π²/(6δ)))
                       Provides theoretical regret bounds
    """
    N = lookup.n

    # Initialize
    idxs = _select_initial_indices_lookup(lookup, n_init, seed, init_method)
    seen, X_obs, Y_obs, rec = _build_initial_from_indices(lookup, idxs, method_tag="gp_ucb")

    best_running = float(max([r["y"] for r in rec]) if rec else -float("inf"))

    for t in range(n_iter):
        # Fit GP
        Y_obs2 = Y_obs.unsqueeze(-1)
        gp = SingleTaskGP(X_obs, Y_obs2)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Compute remaining candidates
        rem = remaining_indices(N, seen)
        if not rem:
            break
        X_pool = lookup.X[rem]

        # Adaptive beta (theory-backed)
        if adaptive_beta:
            d = lookup.d
            delta = 0.1  # Confidence parameter
            beta_t = np.sqrt(2 * np.log((t + 1) * d * np.pi**2 / (6 * delta)))
        else:
            beta_t = beta

        # UCB acquisition
        with torch.no_grad():
            UCB = UpperConfidenceBound(gp, beta=beta_t, maximize=True)
            ucb_vals = UCB(X_pool.unsqueeze(1)).reshape(-1)
            idx_local = int(torch.argmax(ucb_vals))
            pick = rem[idx_local]

        # Observe
        seen.add(pick)
        y = float(lookup.y[pick].item())
        best_running = max(best_running, y)
        X_obs = torch.cat([X_obs, lookup.X[pick].unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, torch.tensor([y], dtype=DTYPE, device=DEVICE)], dim=0)

        row = as_records_row(pick, lookup.X_raw[pick], y, best_running,
                             method="gp_ucb", feature_names=lookup.feature_names)
        row["iter"] = t
        row["beta_used"] = float(beta_t)
        rec.append(row)

    return pd.DataFrame(rec)


def run_thompson_sampling_lookup(
    lookup: "LookupTable",
    *,
    n_init: int = 6,
    n_iter: int = 50,
    seed: int = 0,
    init_method: str = "random",
    n_samples: int = 1,  # TS typically uses 1 sample
) -> pd.DataFrame:
    """
    Thompson Sampling baseline.

    Reference: Russo et al. (2018) "A Tutorial on Thompson Sampling"

    Algorithm:
    1. Sample function f̃ ~ GP(μ, Σ)
    2. Select x = argmax f̃(x) over candidates
    3. Observe y(x)

    This is a randomized acquisition strategy that naturally balances exploration/exploitation.
    """
    N = lookup.n
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize
    idxs = _select_initial_indices_lookup(lookup, n_init, seed, init_method)
    seen, X_obs, Y_obs, rec = _build_initial_from_indices(lookup, idxs, method_tag="thompson_sampling")

    best_running = float(max([r["y"] for r in rec]) if rec else -float("inf"))

    # Create sampler for TS
    sampler = SobolQMCNormalSampler(num_samples=n_samples, seed=seed)

    for t in range(n_iter):
        # Fit GP
        Y_obs2 = Y_obs.unsqueeze(-1)
        gp = SingleTaskGP(X_obs, Y_obs2)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Compute remaining candidates
        rem = remaining_indices(N, seen)
        if not rem:
            break
        X_pool = lookup.X[rem]

        # Thompson Sampling: sample from posterior
        with torch.no_grad():
            posterior = gp.posterior(X_pool.unsqueeze(1))

            # Sample function values
            samples = sampler(posterior)  # Shape: (n_samples, n_pool, 1)
            samples = samples.squeeze(-1)  # (n_samples, n_pool)

            # Take first sample (TS uses single sample)
            sample_vals = samples[0]

            # Select argmax of sampled function
            idx_local = int(torch.argmax(sample_vals))
            pick = rem[idx_local]

        # Observe
        seen.add(pick)
        y = float(lookup.y[pick].item())
        best_running = max(best_running, y)
        X_obs = torch.cat([X_obs, lookup.X[pick].unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, torch.tensor([y], dtype=DTYPE, device=DEVICE)], dim=0)

        row = as_records_row(pick, lookup.X_raw[pick], y, best_running,
                             method="thompson_sampling", feature_names=lookup.feature_names)
        row["iter"] = t
        rec.append(row)

    return pd.DataFrame(rec)


def run_probability_of_improvement_lookup(
    lookup: "LookupTable",
    *,
    n_init: int = 6,
    n_iter: int = 50,
    seed: int = 0,
    init_method: str = "random",
    xi: float = 0.01,  # Exploration parameter (typically small)
) -> pd.DataFrame:
    """
    Probability of Improvement (PI) baseline.

    PI(x) = P(f(x) > f+ + ξ)  where f+ is current best

    Simpler than EI, often used as sanity check.
    """
    N = lookup.n

    # Initialize
    idxs = _select_initial_indices_lookup(lookup, n_init, seed, init_method)
    seen, X_obs, Y_obs, rec = _build_initial_from_indices(lookup, idxs, method_tag="probability_improvement")

    best_running = float(max([r["y"] for r in rec]) if rec else -float("inf"))

    for t in range(n_iter):
        # Fit GP
        Y_obs2 = Y_obs.unsqueeze(-1)
        gp = SingleTaskGP(X_obs, Y_obs2)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Compute remaining candidates
        rem = remaining_indices(N, seen)
        if not rem:
            break
        X_pool = lookup.X[rem]

        # PI acquisition
        with torch.no_grad():
            train_post = gp.posterior(X_obs.unsqueeze(1))
            best_f = float(train_post.mean.max().item())

            PI = ProbabilityOfImprovement(gp, best_f=best_f, maximize=True)
            pi_vals = PI(X_pool.unsqueeze(1)).reshape(-1)
            idx_local = int(torch.argmax(pi_vals))
            pick = rem[idx_local]

        # Observe
        seen.add(pick)
        y = float(lookup.y[pick].item())
        best_running = max(best_running, y)
        X_obs = torch.cat([X_obs, lookup.X[pick].unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, torch.tensor([y], dtype=DTYPE, device=DEVICE)], dim=0)

        row = as_records_row(pick, lookup.X_raw[pick], y, best_running,
                             method="probability_improvement", feature_names=lookup.feature_names)
        row["iter"] = t
        rec.append(row)

    return pd.DataFrame(rec)


def compare_all_baselines(
    lookup: "LookupTable",
    *,
    n_init: int = 6,
    n_iter: int = 50,
    seed: int = 0,
    repeats: int = 5,
    init_method: str = "sobol",
) -> pd.DataFrame:
    """
    Run all baseline methods for comprehensive comparison.

    Returns long-form DataFrame with all methods and seeds.
    """
    import warnings
    warnings.filterwarnings('ignore')

    methods = {
        'random': lambda s: run_random_lookup_baseline(lookup, n_init, n_iter, s, init_method),
        'baseline_ei': lambda s: run_baseline_ei_lookup_baseline(lookup, n_init, n_iter, s, init_method),
        'gp_ucb': lambda s: run_gp_ucb_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, init_method=init_method),
        'thompson_sampling': lambda s: run_thompson_sampling_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, init_method=init_method),
        'probability_improvement': lambda s: run_probability_of_improvement_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, init_method=init_method),
    }

    all_dfs = []

    for method_name, method_fn in methods.items():
        print(f"Running {method_name}...")
        for r in range(repeats):
            current_seed = seed + r
            try:
                df = method_fn(current_seed)
                df['seed'] = current_seed
                df['method'] = method_name
                all_dfs.append(df)
            except Exception as e:
                print(f"  Warning: {method_name} failed on seed {current_seed}: {e}")

    return pd.concat(all_dfs, ignore_index=True)


# Fallback implementations if main_benchmark not available
def run_random_lookup_baseline(lookup, n_init, n_iter, seed, init_method):
    """Fallback random baseline."""
    from main_benchmark import run_random_lookup
    return run_random_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=seed, init_method=init_method)


def run_baseline_ei_lookup_baseline(lookup, n_init, n_iter, seed, init_method):
    """Fallback EI baseline."""
    from main_benchmark import run_baseline_ei_lookup
    return run_baseline_ei_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=seed, init_method=init_method)


if __name__ == "__main__":
    print("Improved Baselines Module")
    print("=" * 50)
    print("\nImplemented methods:")
    print("  - GP-UCB (Upper Confidence Bound)")
    print("  - Thompson Sampling")
    print("  - Probability of Improvement")
    print("\nUsage:")
    print("""
    from improved_baselines import run_gp_ucb_lookup, compare_all_baselines
    from main_benchmark import load_lookup_csv

    # Load data
    lookup = load_lookup_csv("ugi_merged_dataset.csv")

    # Run single method
    hist_ucb = run_gp_ucb_lookup(lookup, n_init=6, n_iter=50, beta=2.0)

    # Run comprehensive comparison
    hist_all = compare_all_baselines(lookup, n_init=6, n_iter=50, repeats=5)

    # Analyze with statistical testing
    from statistical_analysis import pairwise_comparison_table
    results = pairwise_comparison_table(hist_all, correction='holm')
    print(results)
    """)
