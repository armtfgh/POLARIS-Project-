# Methodology Review: Scientific & Practical Improvements

## Executive Summary

Your core idea (LLM → Prior mean → GP) is **scientifically sound**, but there are several areas where the methodology and benchmarking can be significantly strengthened for publication.

**Key issues identified:**
1. ❌ Missing statistical rigor (significance testing, confidence intervals)
2. ❌ Limited baselines (no GP-UCB, Thompson Sampling, etc.)
3. ❌ Ad-hoc hyperparameter choices (prior_strength, rho_floor, alpha scaling)
4. ❌ No ablation studies on key design decisions
5. ❌ Insufficient evaluation metrics (only best-so-far)
6. ❌ No analysis of when/why priors help or hurt

---

## 1. Prior Mean Construction (`prior_gp.py`)

### ✅ What's Good

**Elegant residual fitting:**
```python
def fit_residual_gp(X, Y, prior):
    m0 = prior.m0_torch(X).reshape(-1)
    m0c = m0 - m0.mean()  # Center prior
    yc = Y - Y.mean()      # Center observations
    alpha = dot(m0c, yc) / dot(m0c, m0c)  # Least-squares coefficient
    Y_resid = Y - alpha * m0  # Fit residual
```

This is **mathematically correct** - you're learning the optimal scaling factor α via projection. Good!

### ❌ Issues

#### 1.1. Ad-hoc Prior Strength Scaling

**Current approach (main_benchmark.py:1283):**
```python
rho = alignment_on_obs(X_obs, Y_obs, prior)  # Pearson correlation
rho_weight = max(abs(rho), rho_floor)  # Ad-hoc floor
m0_scale = alpha * prior_strength * rho_weight  # Triple multiplication
```

**Problems:**
- Why multiply α × prior_strength × ρ? No theoretical justification
- `rho_floor=0.05` is a magic number - why 0.05 and not 0.01 or 0.1?
- Using `abs(rho)` treats anti-correlation same as correlation (wrong!)
- No consideration of prior uncertainty

**Proposed fix:**
```python
def adaptive_prior_scale(X, Y, prior, n_min=10):
    """
    Adaptive prior scaling with theoretical justification.

    Returns m0_scale in [0, 1] based on:
    - Sample size (trust prior more with fewer samples)
    - Alignment quality (correlation strength)
    - Prior calibration (how well prior predicts held-out data)
    """
    m0 = prior.m0_torch(X).reshape(-1)
    yv = Y.reshape(-1)
    n = len(yv)

    # 1. Sample size weighting (more data → less prior influence)
    #    Based on Bayesian updating: prior weight ∝ 1/(1 + n/n₀)
    n0 = 10  # Prior pseudo-observations
    sample_weight = n0 / (n0 + n)

    # 2. Alignment quality (signed correlation)
    m0c = m0 - m0.mean()
    yc = yv - yv.mean()
    rho = (m0c @ yc) / torch.sqrt((m0c @ m0c) * (yc @ yc) + 1e-12)

    # Only use prior if positively correlated (rho > threshold)
    if rho < 0.1:  # Minimum correlation threshold
        return 0.0  # Disable prior if misaligned

    # 3. Calibration via cross-validation (if enough data)
    if n >= 20:
        # Split data 80/20
        split = int(0.8 * n)
        X_train, X_val = X[:split], X[split:]
        Y_train, Y_val = Y[:split], Y[split:]

        # Fit alpha on training set
        m0_train = prior.m0_torch(X_train).reshape(-1)
        m0c_train = m0_train - m0_train.mean()
        yc_train = Y_train.reshape(-1) - Y_train.mean()
        alpha_train = (m0c_train @ yc_train) / (m0c_train @ m0c_train + 1e-12)

        # Evaluate on validation set
        m0_val = prior.m0_torch(X_val).reshape(-1)
        pred_val = alpha_train * m0_val
        mse_prior = torch.mean((pred_val - Y_val.reshape(-1)) ** 2)
        mse_baseline = torch.var(Y_val)

        # Calibration weight: 1 if prior perfect, 0 if worse than baseline
        calib_weight = max(0.0, 1.0 - mse_prior / (mse_baseline + 1e-12))
    else:
        calib_weight = 1.0  # Trust alignment if too few samples for CV

    # Final scale: product of three weights
    m0_scale = sample_weight * rho * calib_weight

    return float(m0_scale.item())
```

**Why this is better:**
- **Theoretical grounding:** Sample weighting follows Bayesian updating
- **Adaptive:** Automatically reduces prior influence as data accumulates
- **Calibrated:** Cross-validation ensures prior actually helps
- **Conservative:** Disables prior if misaligned (rho < 0.1)

#### 1.2. No Noise Handling

**Current:**
```python
gp = SingleTaskGP(X, Y_resid)  # Uses default noise model
```

**Problem:** BoTorch's default noise can be poorly calibrated, especially with small n.

**Proposed fix:**
```python
from gpytorch.constraints import GreaterThan

def fit_residual_gp_with_noise(X, Y, prior, noise_constraint=(1e-6, 1e-2)):
    """Fit residual GP with explicit noise bounds."""
    m0 = prior.m0_torch(X).reshape(-1)
    yv = Y.reshape(-1)
    m0c = m0 - m0.mean()
    yc = yv - yv.mean()
    alpha = (m0c @ yc) / (m0c @ m0c + 1e-12)

    Y_resid = Y - alpha * m0.unsqueeze(-1)

    # Create GP with bounded noise
    gp = SingleTaskGP(X, Y_resid)
    gp.likelihood.noise_covar.register_constraint(
        "raw_noise",
        GreaterThan(noise_constraint[0])
    )

    # Set noise prior based on data scale
    noise_prior_scale = 0.1 * torch.std(Y_resid)
    gp.likelihood.noise_covar.noise = noise_prior_scale

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    return gp, alpha
```

#### 1.3. Fixed Kernel Choice

**Current:** Uses default Matérn 5/2 kernel (BoTorch default)

**Problem:** No justification for this choice. Chemical systems might have different smoothness.

**Proposed fix:** Add kernel selection study
```python
def compare_kernels(X_train, Y_train, X_val, Y_val, prior):
    """Compare different kernel choices via cross-validation."""
    from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel

    kernels = {
        "RBF": ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[-1])),
        "Matern-3/2": ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=X_train.shape[-1])),
        "Matern-5/2": ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X_train.shape[-1])),
    }

    results = {}
    for name, kernel in kernels.items():
        gp = SingleTaskGP(X_train, Y_train, covar_module=kernel)
        # ... fit and evaluate on validation set
        results[name] = val_rmse

    best_kernel = min(results, key=results.get)
    return best_kernel, results
```

---

## 2. Benchmarking (`main_benchmark.py`)

### ❌ Missing Statistical Rigor

#### 2.1. No Significance Testing

**Current:**
```python
# Just plots mean ± std across seeds
plot_runs_mean_lookup(hist, ci="sem")
```

**Problem:** Can't claim "Method A is better than B" without statistical tests!

**Proposed fix:**
```python
def statistical_comparison(hist_df, method_a: str, method_b: str, metric: str = "best_so_far"):
    """
    Compare two methods with proper statistical testing.

    Returns:
        - p_value: Wilcoxon signed-rank test (paired, non-parametric)
        - effect_size: Cohen's d
        - confidence_interval: Bootstrap 95% CI on difference
    """
    from scipy.stats import wilcoxon

    # Get final performance for each seed
    final_a = hist_df[(hist_df['method'] == method_a)].groupby('seed')[metric].max()
    final_b = hist_df[(hist_df['method'] == method_b)].groupby('seed')[metric].max()

    # Paired comparison (same seeds)
    common_seeds = set(final_a.index) & set(final_b.index)
    vals_a = final_a[list(common_seeds)].values
    vals_b = final_b[list(common_seeds)].values

    # Wilcoxon signed-rank (non-parametric, paired)
    statistic, p_value = wilcoxon(vals_a, vals_b)

    # Effect size (Cohen's d)
    diff = vals_a - vals_b
    pooled_std = np.sqrt((np.std(vals_a)**2 + np.std(vals_b)**2) / 2)
    cohens_d = np.mean(diff) / (pooled_std + 1e-12)

    # Bootstrap 95% CI
    n_bootstrap = 10000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(vals_a), size=len(vals_a), replace=True)
        bootstrap_diffs.append(np.mean(vals_a[idx] - vals_b[idx]))
    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

    return {
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret_cohens_d(cohens_d),
        'mean_diff': np.mean(diff),
        'ci_95': (ci_low, ci_high),
    }

def interpret_cohens_d(d):
    """Cohen's d interpretation (standard thresholds)."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

**Usage:**
```python
results = statistical_comparison(hist, "hybrid_best", "baseline_ei")
print(f"p-value: {results['p_value']:.4f}")
print(f"Effect size: {results['cohens_d']:.3f} ({results['effect_size_interpretation']})")
print(f"Mean improvement: {results['mean_diff']:.4f} ± [{results['ci_95'][0]:.4f}, {results['ci_95'][1]:.4f}]")
# Output: "p-value: 0.0023 (significant!), Effect size: 0.78 (medium)"
```

#### 2.2. Insufficient Confidence Intervals

**Current:** Uses `ci="sem"` (standard error of mean)

**Problem:** Assumes Gaussian distribution, doesn't account for outliers

**Proposed fix:** Use bootstrap CI
```python
def plot_with_bootstrap_ci(hist_df, methods, n_bootstrap=1000):
    """Plot with bootstrap confidence intervals."""
    for method in methods:
        data = hist_df[hist_df['method'] == method]

        # Group by iteration
        iters = sorted(data['iter'].unique())
        means = []
        ci_lows = []
        ci_highs = []

        for it in iters:
            vals = data[data['iter'] == it]['best_so_far'].values

            # Bootstrap resampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(vals, size=len(vals), replace=True)
                bootstrap_means.append(np.mean(sample))

            means.append(np.mean(vals))
            ci_lows.append(np.percentile(bootstrap_means, 2.5))
            ci_highs.append(np.percentile(bootstrap_means, 97.5))

        plt.plot(iters, means, label=method)
        plt.fill_between(iters, ci_lows, ci_highs, alpha=0.2)
```

### ❌ Missing Baselines

**Current baselines:**
- Random
- Baseline EI (vanilla GP-BO)
- Hybrid (your method)

**Missing important baselines:**
1. **GP-UCB** (Upper Confidence Bound) - standard in BO literature
2. **Thompson Sampling** (TS) - Bayesian alternative to EI
3. **qEI / qUCB** (batch acquisition) - for parallel experiments
4. **SMAC / TPE** - hyperparameter optimization benchmarks
5. **Transfer Learning BO** - prior work on knowledge injection

**Proposed fix:**
```python
def run_gp_ucb_lookup(lookup, beta=2.0, **kwargs):
    """GP-UCB baseline (Srinivas et al., 2010)."""
    from botorch.acquisition import UpperConfidenceBound

    # ... similar to baseline_ei but use UCB acquisition
    UCB = UpperConfidenceBound(gp, beta=beta)
    # ...

def run_thompson_sampling_lookup(lookup, **kwargs):
    """Thompson Sampling baseline (Russo et al., 2018)."""
    # Sample from GP posterior, optimize sample
    from botorch.sampling import SobolQMCNormalSampler
    sampler = SobolQMCNormalSampler(num_samples=1)
    # ...

def compare_all_baselines(lookup, **kwargs):
    """Comprehensive comparison."""
    methods = {
        'random': run_random_lookup,
        'ei': run_baseline_ei_lookup,
        'ucb': run_gp_ucb_lookup,
        'ts': run_thompson_sampling_lookup,
        'hybrid_flat': lambda: run_hybrid_lookup(readout_source='flat'),
        'hybrid_llm': lambda: run_hybrid_lookup(readout_source='llm'),
    }
    # ... run all and compare
```

### ❌ No Ablation Studies

**Key design choices that need ablation:**
1. Prior strength scaling (α × prior_strength × ρ)
2. Residual GP vs direct prior mean
3. LLM translation quality
4. Prior components (effects vs interactions vs bumps)

**Proposed ablation study:**
```python
def ablation_study_prior_components():
    """Test which prior components matter most."""

    # Full prior
    prior_full = {...}  # effects + interactions + bumps

    # Ablated priors
    prior_effects_only = {...}  # only main effects
    prior_no_bumps = {...}  # effects + interactions
    prior_bumps_only = {...}  # only bumps

    results = {}
    for name, prior in priors.items():
        hist = run_hybrid_lookup(prior=prior, ...)
        results[name] = compute_final_performance(hist)

    # Statistical comparison
    for name in results:
        if name != 'full':
            print(f"{name}: {compare_to_full(results[name], results['full'])}")
```

**Expected insights:**
- Are interactions necessary or just noise?
- Do bumps help early but hurt later?
- Is LLM adding value or just noise?

---

## 3. Evaluation Metrics

### ❌ Limited Metrics

**Current:** Only "best so far"

**Missing:**
1. **Simple Regret**: r(t) = f(x*) - max_{i≤t} f(x_i)
2. **Cumulative Regret**: R(T) = Σ_t [f(x*) - f(x_t)]
3. **Sample Efficiency**: Iterations to reach threshold (you have this in evaluate_prior_value.py!)
4. **Exploration Quality**: Coverage of parameter space
5. **Robustness**: Performance variance across seeds

**Proposed comprehensive metrics:**
```python
def comprehensive_evaluation(hist_df, lookup):
    """Compute all relevant BO metrics."""
    y_max_true = float(lookup.y.max())

    metrics = {}

    for method in hist_df['method'].unique():
        data = hist_df[hist_df['method'] == method]

        # 1. Simple regret (per iteration)
        simple_regret = y_max_true - data.groupby('iter')['best_so_far'].mean()

        # 2. Cumulative regret
        cumulative_regret = (y_max_true - data['y']).sum()

        # 3. Sample efficiency (iterations to 90% optimum)
        threshold_90 = 0.9 * y_max_true
        reached = data[data['best_so_far'] >= threshold_90]
        iters_to_90 = reached['iter'].min() if len(reached) > 0 else np.inf

        # 4. Exploration coverage (% of parameter space visited)
        visited_X = data[['x1', 'x2', 'x3', 'x4']].values
        # Compute convex hull volume or grid coverage
        coverage = compute_coverage(visited_X, lookup.X.numpy())

        # 5. Robustness (coefficient of variation)
        final_perfs = data.groupby('seed')['best_so_far'].max()
        robustness = final_perfs.std() / (final_perfs.mean() + 1e-12)

        metrics[method] = {
            'simple_regret_final': simple_regret.iloc[-1],
            'cumulative_regret': cumulative_regret,
            'iters_to_90pct': iters_to_90,
            'coverage': coverage,
            'robustness_cv': robustness,
            'final_performance_mean': final_perfs.mean(),
            'final_performance_std': final_perfs.std(),
        }

    return pd.DataFrame(metrics).T
```

---

## 4. Code Quality & Reproducibility

### ❌ Issues

1. **Very long files** (1800+ lines in main_benchmark.py)
2. **Magic numbers** everywhere (rho_floor=0.05, prior_strength=1.0, etc.)
3. **No configuration management** (everything hardcoded)
4. **Limited documentation** of design choices

### ✅ Proposed Improvements

#### 4.1. Configuration Management

**Create `config.py`:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class BOConfig:
    """Bayesian Optimization configuration."""
    n_init: int = 6
    n_iter: int = 50
    seed: int = 0
    repeats: int = 5
    init_method: str = "sobol"  # or "random", "llm-si"

@dataclass
class PriorConfig:
    """Prior mean configuration."""
    prior_strength: float = 1.0
    rho_floor: float = 0.05
    early_boost: bool = False
    early_boost_steps: int = 5
    adaptive_scaling: bool = True  # Use new adaptive method

@dataclass
class GPConfig:
    """GP hyperparameters."""
    kernel: str = "matern-5/2"  # or "rbf", "matern-3/2"
    noise_constraint: tuple = (1e-6, 1e-2)
    ard: bool = True  # Automatic relevance determination

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    bo: BOConfig = BOConfig()
    prior: PriorConfig = PriorConfig()
    gp: GPConfig = GPConfig()

    def save(self, path: str):
        """Save config to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON."""
        import json
        with open(path) as f:
            return cls(**json.load(f))
```

**Usage:**
```python
# Create config
config = ExperimentConfig(
    bo=BOConfig(n_iter=100, repeats=10),
    prior=PriorConfig(adaptive_scaling=True)
)

# Save for reproducibility
config.save("experiment_configs/main_experiment.json")

# Run experiment
results = run_experiment(lookup, config)
```

#### 4.2. Modular Code Structure

**Refactor into modules:**
```
src/
├── acquisition/
│   ├── ei.py
│   ├── ucb.py
│   └── ts.py
├── gp/
│   ├── prior_mean.py      # Your prior construction
│   ├── kernels.py          # Kernel selection
│   └── fitting.py          # GP fitting utils
├── benchmarks/
│   ├── baseline.py         # Vanilla BO
│   ├── hybrid.py           # Your method
│   └── competitors.py      # UCB, TS, etc.
├── evaluation/
│   ├── metrics.py          # All metrics
│   ├── statistics.py       # Significance tests
│   └── plotting.py         # Visualization
└── experiments/
    ├── run_main.py         # Main experiment
    ├── run_ablation.py     # Ablation studies
    └── configs/            # JSON configs
```

---

## 5. Experimental Design for Paper

### Recommended Experiments

#### Experiment 1: **Main Claim** (Human vs Baseline)
```python
# Compare human expert vs vanilla BO vs random
methods = ['random', 'baseline_ei', 'hybrid_human_expert']
config = ExperimentConfig(
    bo=BOConfig(n_iter=100, repeats=20),  # 20 seeds for power
    prior=PriorConfig(adaptive_scaling=True)
)
results = run_comprehensive_comparison(lookup, methods, config)

# Report:
# - Mean final performance with 95% CI
# - p-value (Wilcoxon test)
# - Effect size (Cohen's d)
# - Sample efficiency (iters to 90%)
```

#### Experiment 2: **Ablation Studies**
```python
# Test each design choice
ablations = {
    'full': hybrid_full,
    'no_interactions': hybrid_no_interactions,
    'no_bumps': hybrid_no_bumps,
    'fixed_scaling': hybrid_alpha_only,
    'adaptive_scaling': hybrid_adaptive,
}
results_ablation = run_ablation_study(lookup, ablations, config)

# Report which components are necessary
```

#### Experiment 3: **Generalization**
```python
# Test on multiple chemical systems
datasets = {
    'ugi': load_ugi_dataset(),
    'suzuki': load_suzuki_dataset(),
    'buchwald': load_buchwald_hartwig_dataset(),
}

for name, data in datasets.items():
    results[name] = run_experiment(data, config)

# Report: Does method generalize across chemistries?
```

#### Experiment 4: **Scalability**
```python
# Test with different dimensions and dataset sizes
configs = [
    BOConfig(n_init=3, n_iter=25),   # Low data
    BOConfig(n_init=6, n_iter=50),   # Medium
    BOConfig(n_init=12, n_iter=100), # High data
]

for cfg in configs:
    results.append(run_experiment(lookup, cfg))

# Report: When does prior help most? (hypothesis: low data regime)
```

---

## 6. Key Recommendations for Paper

### Must-Have for Publication

1. **✅ Statistical significance testing**
   - Report p-values (Wilcoxon signed-rank)
   - Report effect sizes (Cohen's d)
   - Use bootstrap confidence intervals

2. **✅ Comprehensive baselines**
   - Add GP-UCB and Thompson Sampling
   - Compare to prior work on knowledge injection

3. **✅ Ablation studies**
   - Show which prior components matter
   - Justify design choices (adaptive scaling, etc.)

4. **✅ Multiple domains**
   - Test on ≥3 chemical systems
   - Show generalization

5. **✅ Sample efficiency focus**
   - Main metric: "iterations to reach 90% optimum"
   - This directly supports your claim

6. **✅ Failure analysis**
   - When does prior hurt performance?
   - What happens with bad priors?

### Nice-to-Have

1. **Theoretical analysis**
   - Regret bounds for prior-GP
   - Convergence guarantees

2. **Computational cost analysis**
   - LLM API cost vs BO sample cost
   - When is knowledge acquisition worth it?

3. **User study**
   - Have chemists provide priors
   - Measure inter-rater reliability

---

## Summary: Priority Improvements

### 🔴 Critical (Must Fix)

1. **Add statistical significance testing** (section 2.1)
2. **Add GP-UCB and TS baselines** (section 2.2)
3. **Fix prior strength scaling** (section 1.1)
4. **Use sample efficiency as main metric** (section 3)

### 🟡 Important (Should Fix)

5. **Add ablation studies** (section 2.3)
6. **Improve confidence intervals (bootstrap)** (section 2.2)
7. **Add noise handling to GP** (section 1.2)
8. **Modularize code** (section 4.2)

### 🟢 Nice-to-Have

9. **Kernel selection study** (section 1.3)
10. **Configuration management** (section 4.1)
11. **Exploration coverage metrics** (section 3)

---

## Implementation Priority

**Week 1:** Statistical rigor
- Implement significance testing (4 hours)
- Add bootstrap CI (2 hours)
- Report effect sizes (1 hour)

**Week 2:** Baselines & ablations
- Implement GP-UCB (3 hours)
- Implement Thompson Sampling (3 hours)
- Run ablation studies (variable)

**Week 3:** Evaluation
- Implement comprehensive metrics (4 hours)
- Sample efficiency analysis (2 hours)
- Create final plots (2 hours)

**Week 4:** Code quality
- Refactor into modules (8 hours)
- Add configuration system (4 hours)
- Documentation (4 hours)

---

**Bottom line:** Your core methodology is sound! The main improvements needed are:
1. Statistical rigor (significance tests, proper CI)
2. More baselines (GP-UCB, TS)
3. Ablation studies (justify design choices)
4. Focus on sample efficiency (your key claim)

These changes will make your work much more compelling for publication. Want me to implement any of these improvements?
