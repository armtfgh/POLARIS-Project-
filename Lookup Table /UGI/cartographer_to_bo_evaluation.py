#!/usr/bin/env python3
"""
cartographer_to_bo_evaluation.py
=================================

Connect Cartographer (knowledge generator) to BO evaluation (knowledge value).

Workflow:
1. Cartographer explores with budget B → generates hypothesis
2. Hypothesis → Prior (JSON format)
3. Prior → BO evaluation → measure sample efficiency gain

This completes the loop and answers your research question:
"How many BO samples is the Cartographer-generated knowledge worth?"

Example:
- Cartographer with 20 samples → hypothesis H₂₀
- Use H₂₀ as GP prior → BO reaches 90% optimum in 25 iterations
- Vanilla BO reaches 90% optimum in 60 iterations
- **Conclusion: 20 samples of exploration = 35 samples of optimization value**
"""

from cartographer import (
    run_cartographer_analyst_pipeline_improved,
    load_lookup_csv_oracle,
)
from evaluate_prior_value import (
    run_prior_value_experiment,
    plot_knowledge_value_curve,
)
from readout_schema import readout_to_prior
from prompt_generator import load_lookup_csv, normalize_readout_to_unit_box
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from os import getenv


def generate_priors_at_checkpoints(
    csv_path: str = "ugi_merged_dataset.csv",
    checkpoints: List[int] = [20, 40, 60, 80, 100],
    seed: int = 42,
    api_key: str = None,
) -> Dict[int, any]:
    """
    Run Cartographer and extract priors at different sample budgets.

    Args:
        csv_path: Dataset path
        checkpoints: List of sample counts to extract priors (e.g., [20, 50, 100])
        seed: Random seed for reproducibility
        api_key: OpenAI API key

    Returns:
        Dictionary mapping checkpoint → Prior object
    """
    print("="*70)
    print("GENERATING PRIORS FROM CARTOGRAPHER")
    print("="*70)
    print(f"Checkpoints: {checkpoints}")
    print(f"Seed: {seed}\n")

    # Run Cartographer to generate knowledge at checkpoints
    max_budget = max(checkpoints)
    analyst_every = min(checkpoints)  # Check at smallest interval

    hist_df, knowledge_df = run_cartographer_analyst_pipeline_improved(
        csv_path=csv_path,
        budget=max_budget,
        n_init=6,
        seed=seed,
        analyst_every=analyst_every,
        analyst_model="gpt-4o",
        api_key=api_key,
        use_direct_json=True,
    )

    # Extract priors at requested checkpoints
    lookup = load_lookup_csv(csv_path, objective_col="yield")
    priors = {}

    for checkpoint in checkpoints:
        # Find the closest analyst checkpoint
        available = knowledge_df[knowledge_df['n_obs'] <= checkpoint + 10]  # Allow small margin
        if len(available) == 0:
            print(f"  ⚠ No data for checkpoint {checkpoint}, skipping")
            continue

        row = available.iloc[-1]  # Take the latest available
        actual_n_obs = row['n_obs']

        if 'readout' in row and row['readout'] is not None:
            # Direct JSON approach (new)
            readout_unit = row['readout']
            prior = readout_to_prior(readout_unit, feature_names=["x1", "x2", "x3", "x4"])
            priors[int(actual_n_obs)] = prior
            print(f"  ✓ Extracted prior from n={actual_n_obs} samples (target: {checkpoint})")
        else:
            print(f"  ⚠ No readout available for checkpoint {checkpoint}")

    print(f"\n✓ Generated {len(priors)} priors from Cartographer\n")
    return priors


def evaluate_cartographer_knowledge_value(
    csv_path: str = "ugi_merged_dataset.csv",
    checkpoints: List[int] = [20, 40, 60, 80, 100],
    n_init: int = 6,
    bo_budget: int = 100,
    n_trials: int = 5,
    seed: int = 42,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Complete pipeline: Cartographer → Priors → BO Evaluation

    This answers: "How many BO samples is Cartographer knowledge worth at different budgets?"
    """
    print("\n" + "="*70)
    print("CARTOGRAPHER KNOWLEDGE VALUE EVALUATION")
    print("="*70)
    print(f"BO Configuration: n_init={n_init}, budget={bo_budget}, trials={n_trials}\n")

    # Step 1: Generate priors from Cartographer
    priors = generate_priors_at_checkpoints(
        csv_path=csv_path,
        checkpoints=checkpoints,
        seed=seed,
        api_key=api_key,
    )

    # Step 2: Evaluate each prior in BO
    all_results = []

    # Vanilla baseline
    print("Evaluating: Vanilla BO (no prior)")
    results_vanilla = run_prior_value_experiment(
        csv_path=csv_path,
        prior=None,
        prior_name="vanilla_0samples",
        n_init=n_init,
        budget=bo_budget,
        n_trials=n_trials,
    )
    all_results.append(results_vanilla)

    # Cartographer-generated priors
    for n_obs, prior in sorted(priors.items()):
        print(f"\nEvaluating: Cartographer prior (n={n_obs} samples)")
        results = run_prior_value_experiment(
            csv_path=csv_path,
            prior=prior,
            prior_name=f"cartographer_{n_obs}samples",
            n_init=n_init,
            budget=bo_budget,
            n_trials=n_trials,
        )
        all_results.append(results)

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)

    # Summary
    print("\n" + "="*70)
    print("KNOWLEDGE VALUE SUMMARY")
    print("="*70)
    summary = combined.groupby('prior_name').agg({
        'best_found_prior': ['mean', 'std'],
        'mean_samples_saved': ['mean', 'std'],
        'samples_saved_90pct': ['mean', 'std'],
    }).round(4)
    print(summary)

    # Save
    combined.to_csv('cartographer_knowledge_value.csv', index=False)
    print("\nResults saved to: cartographer_knowledge_value.csv")

    return combined


def plot_exploration_to_optimization_transfer(
    results_df: pd.DataFrame,
    save_path: str = "exploration_to_optimization_transfer.png"
):
    """
    Key figure: Shows transfer efficiency from exploration to optimization.

    X-axis: Exploration budget (Cartographer samples)
    Y-axis: Optimization value (BO samples saved)

    Interpretation:
    - Slope > 1: Exploration is MORE efficient than optimization (1 explore sample > 1 BO sample)
    - Slope = 1: Equal efficiency (1 explore sample = 1 BO sample)
    - Slope < 1: Exploration is LESS efficient (1 explore sample < 1 BO sample)

    Your hypothesis: Human expert is slope → ∞ (0 samples, huge value)
    """
    import matplotlib.pyplot as plt

    # Extract data
    summary = results_df.groupby('prior_name').agg({
        'mean_samples_saved': ['mean', 'std'],
        'samples_saved_90pct': ['mean', 'std'],
    })

    # Parse sample counts from prior names
    exploration_samples = []
    optimization_gains = []
    optimization_gains_std = []

    for name in summary.index:
        if 'cartographer' in name:
            # Extract sample count from name like "cartographer_20samples"
            n_samples = int(name.split('_')[1].replace('samples', ''))
            exploration_samples.append(n_samples)
            optimization_gains.append(summary.loc[name, ('mean_samples_saved', 'mean')])
            optimization_gains_std.append(summary.loc[name, ('mean_samples_saved', 'std')])

    # Sort by exploration samples
    sorted_indices = np.argsort(exploration_samples)
    exploration_samples = [exploration_samples[i] for i in sorted_indices]
    optimization_gains = [optimization_gains[i] for i in sorted_indices]
    optimization_gains_std = [optimization_gains_std[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main curve
    ax.errorbar(exploration_samples, optimization_gains, yerr=optimization_gains_std,
                marker='o', linewidth=2, markersize=8, capsize=5, label='Cartographer-generated priors')

    # Reference line: y = x (equal transfer)
    max_val = max(max(exploration_samples), max(optimization_gains)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal transfer (1:1)')

    # Shaded regions
    ax.axhspan(0, max_val, xmin=0, xmax=1, alpha=0.05, color='green', zorder=-1)
    ax.text(max_val*0.7, max_val*0.9, 'Efficient transfer\n(Exploration > Optimization)',
            ha='center', va='top', fontsize=9, style='italic', alpha=0.5)

    ax.set_xlabel('Exploration Budget (Cartographer samples)', fontsize=12)
    ax.set_ylabel('Optimization Value (BO samples saved)', fontsize=12)
    ax.set_title('Knowledge Transfer: Exploration → Optimization', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(exploration_samples) * 1.1)
    ax.set_ylim(0, max(optimization_gains) * 1.2)

    # Fit linear trend
    if len(exploration_samples) > 2:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(exploration_samples, optimization_gains)

        # Add trend line
        x_trend = np.array([0, max(exploration_samples)])
        y_trend = slope * x_trend + intercept
        ax.plot(x_trend, y_trend, 'r--', alpha=0.5, linewidth=1.5,
                label=f'Trend: slope={slope:.2f}, R²={r_value**2:.3f}')

        # Interpretation text
        if slope > 1:
            interpretation = f"✓ Exploration is {slope:.1f}× more efficient than direct optimization"
        elif slope > 0.5:
            interpretation = f"≈ Exploration and optimization have similar efficiency (slope={slope:.2f})"
        else:
            interpretation = f"⚠ Exploration is less efficient (slope={slope:.2f})"

        ax.text(0.98, 0.02, interpretation,
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTransfer plot saved to: {save_path}")
    return fig


def compare_human_vs_cartographer(
    csv_path: str = "ugi_merged_dataset.csv",
    human_prior_prompt: str = None,
    cartographer_checkpoints: List[int] = [20, 50, 100],
    n_trials: int = 5,
    api_key: str = None,
):
    """
    THE KEY EXPERIMENT for your paper:

    Compare:
    1. Human expert (1 sentence, 0 samples) → Prior → BO value
    2. Cartographer (20/50/100 samples) → Prior → BO value
    3. Vanilla BO (no prior)

    Research question: "Is human sentence worth more than 100 Cartographer samples?"
    """
    from bo_readout_prompts import SYS_PROMPTS_BEST
    from prompt_generator import score_prompt_via_llm
    from readout_schema import readout_to_prior

    print("\n" + "="*70)
    print("HUMAN vs CARTOGRAPHER vs VANILLA")
    print("="*70)
    print("Research Question: How much is human knowledge worth?\n")

    if human_prior_prompt is None:
        human_prior_prompt = SYS_PROMPTS_BEST

    all_results = []

    # 1. Vanilla baseline
    print("1. Vanilla BO (no knowledge)")
    results_vanilla = run_prior_value_experiment(
        csv_path=csv_path,
        prior=None,
        prior_name="vanilla",
        n_init=6,
        budget=100,
        n_trials=n_trials,
    )
    all_results.append(results_vanilla)

    # 2. Human expert
    print("\n2. Human Expert (0 samples, 1 sentence)")
    result = score_prompt_via_llm(prompt=human_prior_prompt, csv_path=csv_path, api_key=api_key)
    human_prior = readout_to_prior(result['readout_unit'], feature_names=["x1", "x2", "x3", "x4"])

    results_human = run_prior_value_experiment(
        csv_path=csv_path,
        prior=human_prior,
        prior_name="human_expert",
        n_init=6,
        budget=100,
        n_trials=n_trials,
    )
    all_results.append(results_human)

    # 3. Cartographer at different budgets
    priors_cart = generate_priors_at_checkpoints(
        csv_path=csv_path,
        checkpoints=cartographer_checkpoints,
        seed=42,
        api_key=api_key,
    )

    for n_obs, prior in sorted(priors_cart.items()):
        print(f"\n3. Cartographer ({n_obs} samples)")
        results = run_prior_value_experiment(
            csv_path=csv_path,
            prior=prior,
            prior_name=f"cartographer_{n_obs}",
            n_init=6,
            budget=100,
            n_trials=n_trials,
        )
        all_results.append(results)

    # Combine and analyze
    combined = pd.concat(all_results, ignore_index=True)
    summary = combined.groupby('prior_name')['mean_samples_saved'].agg(['mean', 'std'])

    print("\n" + "="*70)
    print("FINAL ANSWER TO RESEARCH QUESTION")
    print("="*70)
    print(summary)
    print()

    human_value = summary.loc['human_expert', 'mean']
    vanilla_value = summary.loc['vanilla', 'mean'] if 'vanilla' in summary.index else 0

    print(f"Human expert knowledge value: {human_value:.1f} BO samples saved")

    # Compare to Cartographer
    for n_obs in sorted([k for k in priors_cart.keys()]):
        name = f"cartographer_{n_obs}"
        if name in summary.index:
            cart_value = summary.loc[name, 'mean']
            print(f"Cartographer ({n_obs} samples): {cart_value:.1f} BO samples saved")

            if cart_value > 0:
                ratio = human_value / cart_value
                print(f"  → Human is {ratio:.2f}× more valuable")

    print("\n" + "="*70)
    print(f"CONCLUSION: Human sentence (0 samples) = {human_value:.0f} BO samples")
    print("="*70)

    # Save and plot
    combined.to_csv('human_vs_cartographer.csv', index=False)

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    means = summary['mean'].values
    stds = summary['std'].values
    labels = summary.index.tolist()

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)

    # Color human bar differently
    for i, label in enumerate(labels):
        if 'human' in label:
            bars[i].set_color('gold')
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)

    ax.set_ylabel('Sample Efficiency Gain\n(BO Samples Saved)', fontsize=12)
    ax.set_title('Human vs Cartographer vs Vanilla', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('human_vs_cartographer.png', dpi=150)
    print("\nPlot saved to: human_vs_cartographer.png")

    return combined


if __name__ == "__main__":
    api_key = getenv("OPENAI_API_KEY", None)

    # Run the key experiment
    results = compare_human_vs_cartographer(
        csv_path="ugi_merged_dataset.csv",
        cartographer_checkpoints=[20, 40, 60, 80, 100],
        n_trials=3,  # Reduce for faster testing
        api_key=api_key,
    )

    print("\n✓ Experiment complete!")
    print("\nFiles generated:")
    print("  - human_vs_cartographer.csv: Detailed results")
    print("  - human_vs_cartographer.png: Comparison plot")
    print("\nThis directly tests your research claim:")
    print("  'One sentence from human > 100 BO samples'")
