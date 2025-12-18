"""
alternative_metrics.py
======================

Alternative evaluation metrics for knowledge extraction that are better aligned
with pure exploration goals.

Problem: NDCG measures top-k ranking, but pure exploration doesn't focus on top-k.
Solution: Use metrics that measure overall landscape understanding.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict
from prior_gp import Prior


def evaluate_prior_comprehensive(
    prior: Prior,
    X_unit: Tensor,
    y: Tensor,
) -> Dict[str, float]:
    """
    Comprehensive evaluation beyond just NDCG.

    Returns multiple metrics that capture different aspects of landscape understanding:
    1. Pearson/Spearman alignment (overall trend matching)
    2. Uncertainty reduction (how much prior helps vs flat prior)
    3. Calibration (are high-confidence predictions accurate?)
    4. Coverage (does prior explain different yield ranges?)
    """
    with torch.no_grad():
        m0 = prior.m0_torch(X_unit).reshape(-1)

    m0_np = m0.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().reshape(-1)

    metrics = {}

    # 1. Overall alignment (Pearson)
    m0_centered = m0_np - m0_np.mean()
    y_centered = y_np - y_np.mean()
    cov = np.dot(m0_centered, y_centered)
    std_m0 = np.sqrt(np.dot(m0_centered, m0_centered))
    std_y = np.sqrt(np.dot(y_centered, y_centered))
    pearson = cov / (std_m0 * std_y + 1e-12) if std_m0 > 0 and std_y > 0 else 0.0
    metrics['pearson'] = float(pearson)

    # 2. Spearman (rank correlation - more robust)
    from scipy.stats import spearmanr
    spearman, p_value = spearmanr(m0_np, y_np)
    metrics['spearman'] = float(spearman)
    metrics['spearman_pvalue'] = float(p_value)

    # 3. Mean Absolute Error (MAE) - lower is better
    mae = np.mean(np.abs(m0_np - y_np))
    metrics['mae'] = float(mae)

    # 4. Root Mean Squared Error (RMSE) - lower is better
    rmse = np.sqrt(np.mean((m0_np - y_np) ** 2))
    metrics['rmse'] = float(rmse)

    # 5. R² score (explained variance)
    ss_res = np.sum((y_np - m0_np) ** 2)
    ss_tot = np.sum((y_np - y_np.mean()) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    metrics['r2'] = float(r2)

    # 6. Quantile-based evaluation (how well does prior rank different yield levels?)
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]
    for q in quantiles:
        threshold = np.quantile(y_np, q)
        mask = y_np >= threshold
        if mask.sum() > 0:
            mean_prior_above = m0_np[mask].mean()
            mean_prior_below = m0_np[~mask].mean()
            separation = mean_prior_above - mean_prior_below
            metrics[f'separation_q{int(q*100)}'] = float(separation)

    # 7. Top-k metrics (still useful, but not the only metric)
    for k_frac in [0.01, 0.05, 0.10]:
        k = max(1, int(len(y_np) * k_frac))
        top_k_true = set(np.argsort(y_np)[-k:])
        top_k_pred = set(np.argsort(m0_np)[-k:])
        overlap = len(top_k_true & top_k_pred) / k
        metrics[f'topk_overlap_{int(k_frac*100)}pct'] = float(overlap)

    # 8. Directional accuracy (are the gradients pointing the right way?)
    # For each variable, check if prior gradient aligns with true gradient
    d = X_unit.shape[-1]
    for dim in range(d):
        X_lo = X_unit[X_unit[:, dim] < 0.5]
        X_hi = X_unit[X_unit[:, dim] >= 0.5]
        y_lo = y[X_unit[:, dim] < 0.5]
        y_hi = y[X_unit[:, dim] >= 0.5]

        if len(X_lo) > 0 and len(X_hi) > 0:
            with torch.no_grad():
                m0_lo = prior.m0_torch(X_lo).mean()
                m0_hi = prior.m0_torch(X_hi).mean()

            true_gradient = float((y_hi.mean() - y_lo.mean()).item())
            pred_gradient = float((m0_hi - m0_lo).item())

            # Check if signs match (directional correctness)
            correct_direction = (true_gradient * pred_gradient) > 0
            metrics[f'direction_correct_x{dim+1}'] = float(correct_direction)

    # 9. Combined score (weighted average of key metrics)
    combined = (
        0.3 * metrics['spearman'] +
        0.2 * metrics['r2'] +
        0.2 * metrics.get('topk_overlap_1pct', 0) +
        0.15 * metrics.get('separation_q90', 0) / 0.1 +  # Normalize to ~0-1 range
        0.15 * metrics.get('topk_overlap_5pct', 0)
    )
    metrics['combined_score'] = float(combined)

    return metrics


def plot_prior_vs_truth(prior: Prior, X_unit: Tensor, y: Tensor, save_path: str = "prior_evaluation.png"):
    """
    Visualize how well the prior matches ground truth.
    """
    import matplotlib.pyplot as plt

    with torch.no_grad():
        m0 = prior.m0_torch(X_unit).reshape(-1)

    m0_np = m0.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().reshape(-1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Scatter plot
    ax = axes[0, 0]
    ax.scatter(m0_np, y_np, alpha=0.3, s=1)
    ax.plot([m0_np.min(), m0_np.max()], [m0_np.min(), m0_np.max()], 'r--', alpha=0.5)
    ax.set_xlabel('Prior Mean (predicted)')
    ax.set_ylabel('True Yield')
    ax.set_title('Prior vs Ground Truth')
    ax.grid(True, alpha=0.3)

    # 2. Residuals
    ax = axes[0, 1]
    residuals = y_np - m0_np
    ax.scatter(m0_np, residuals, alpha=0.3, s=1)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prior Mean (predicted)')
    ax.set_ylabel('Residual (true - predicted)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    # 3. Distribution comparison
    ax = axes[0, 2]
    ax.hist(y_np, bins=50, alpha=0.5, label='True', density=True)
    ax.hist(m0_np, bins=50, alpha=0.5, label='Prior', density=True)
    ax.set_xlabel('Yield')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Quantile-quantile plot
    ax = axes[1, 0]
    quantiles = np.linspace(0, 1, 100)
    q_true = np.quantile(y_np, quantiles)
    q_pred = np.quantile(m0_np, quantiles)
    ax.plot(q_true, q_pred, 'o', markersize=3)
    ax.plot([q_true.min(), q_true.max()], [q_true.min(), q_true.max()], 'r--', alpha=0.5)
    ax.set_xlabel('True Yield Quantiles')
    ax.set_ylabel('Prior Yield Quantiles')
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)

    # 5. Top-k analysis
    ax = axes[1, 1]
    k_values = [int(len(y_np) * f) for f in [0.01, 0.02, 0.05, 0.10, 0.20]]
    overlaps = []
    for k in k_values:
        top_k_true = set(np.argsort(y_np)[-k:])
        top_k_pred = set(np.argsort(m0_np)[-k:])
        overlap = len(top_k_true & top_k_pred) / k
        overlaps.append(overlap)
    ax.plot([k/len(y_np) * 100 for k in k_values], overlaps, 'o-', linewidth=2)
    ax.set_xlabel('Top-k (%)')
    ax.set_ylabel('Overlap Fraction')
    ax.set_title('Top-k Ranking Accuracy')
    ax.grid(True, alpha=0.3)

    # 6. Summary metrics
    ax = axes[1, 2]
    ax.axis('off')
    metrics = evaluate_prior_comprehensive(prior, X_unit, y)
    summary_text = f"""
    Summary Metrics:

    Pearson:    {metrics['pearson']:.4f}
    Spearman:   {metrics['spearman']:.4f}
    R²:         {metrics['r2']:.4f}

    MAE:        {metrics['mae']:.4f}
    RMSE:       {metrics['rmse']:.4f}

    Top-1% overlap:  {metrics.get('topk_overlap_1pct', 0):.4f}
    Top-5% overlap:  {metrics.get('topk_overlap_5pct', 0):.4f}

    Combined:   {metrics['combined_score']:.4f}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to: {save_path}")
    return fig


if __name__ == "__main__":
    # Demo: evaluate the "best" hand-crafted prior
    from prompt_generator import load_lookup_csv, score_prompt_via_llm
    from bo_readout_prompts import SYS_PROMPTS_BEST
    from readout_schema import readout_to_prior

    print("Loading dataset and evaluating hand-crafted 'BEST' prior...")

    # Generate prior from best prompt
    result = score_prompt_via_llm(
        prompt=SYS_PROMPTS_BEST,
        csv_path="ugi_merged_dataset.csv",
        objective_col="yield",
    )

    # Load full dataset
    lookup = load_lookup_csv("ugi_merged_dataset.csv", objective_col="yield")

    # Get prior
    prior = readout_to_prior(result['readout_unit'], feature_names=lookup.feature_names)

    # Evaluate comprehensively
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    metrics = evaluate_prior_comprehensive(prior, lookup.X, lookup.y)

    for key, value in sorted(metrics.items()):
        print(f"{key:<30} {value:>10.4f}")

    # Plot
    plot_prior_vs_truth(prior, lookup.X, lookup.y)

    import matplotlib.pyplot as plt
    plt.show()
