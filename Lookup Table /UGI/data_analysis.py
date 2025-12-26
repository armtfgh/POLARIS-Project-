
#%%
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
###
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - informative guard
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_IMPORT_ERROR = exc

try:
    import torch
    from botorch.acquisition.acquisition import AcquisitionFunction
    from botorch.acquisition.analytic import (
        ExpectedImprovement,
        PosteriorMean,
        ProbabilityOfImprovement,
        UpperConfidenceBound,
    )
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    BOTORCH_AVAILABLE = True
    BOTORCH_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - informative guard
    BOTORCH_AVAILABLE = False
    BOTORCH_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]


def load_ugi_series(path_pattern: str = "ugi_raw/ugi_hyvu_{:04d}.csv", count: int = 100) -> pd.DataFrame:
    """Load a sequence of UGI CSV files into a single DataFrame."""
    data_frames = []

    for i in range(count):
        file_path = path_pattern.format(i)
        df = pd.read_csv(file_path)
        data_frames.append(df)

    if not data_frames:
        raise ValueError("No data files were loaded; check the path pattern or file count.")

    return pd.concat(data_frames, ignore_index=True)


def report_duplicates(df: pd.DataFrame, subset: Iterable[str] | None = None) -> None:
    """Print a short duplicate report for the provided DataFrame."""
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    duplicate_count = int(duplicate_mask.sum())

    print(f"Duplicated rows (counting all occurrences): {duplicate_count}")

    if duplicate_count:
        unique_duplicate_rows = df.loc[duplicate_mask].drop_duplicates()
        print(f"Unique duplicated row patterns: {len(unique_duplicate_rows)}")
        print("Sample duplicated rows:")
        print(unique_duplicate_rows.head())
    else:
        print("No duplicate rows detected.")


def prepare_features(df: pd.DataFrame, target: str = "yield") -> Tuple[pd.DataFrame, List[str]]:
    """Drop rows without the target and return numeric feature columns."""
    df_clean = df.dropna(subset=[target])
    numeric_columns = df_clean.select_dtypes(include="number").columns.tolist()

    if target not in numeric_columns:
        raise ValueError(f"Target column '{target}' not found among numeric columns.")

    numeric_columns.remove(target)
    if not numeric_columns:
        raise ValueError("No numeric features available for training.")

    return df_clean, numeric_columns


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[RandomForestRegressor, Dict[str, float], List[Tuple[float, str]]]:
    """Train a RandomForestRegressor and return model plus metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "oob_score": float(model.oob_score_),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2),
    }

    feature_importances = sorted(
        zip(model.feature_importances_, X.columns),
        key=lambda pair: pair[0],
        reverse=True,
    )

    return model, metrics, feature_importances


class RandomForestOracle:
    """Light wrapper that evaluates a trained RandomForestRegressor on torch tensors."""

    def __init__(self, model: RandomForestRegressor, feature_columns: List[str]) -> None:
        self.model = model
        self.feature_columns = feature_columns

    def __call__(self, candidates: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        if torch is None:
            raise RuntimeError("Torch is required to evaluate the oracle.")

        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0)

        preds = self.model.predict(candidates.detach().cpu().numpy())
        return torch.tensor(preds, dtype=candidates.dtype, device=candidates.device)


def compute_feature_bounds(df: pd.DataFrame, feature_columns: List[str]) -> "torch.Tensor":  # type: ignore[name-defined]
    """Return a 2 x d tensor with [min; max] bounds per feature."""
    if torch is None:
        raise RuntimeError("Torch is required to compute feature bounds.")

    mins = df[feature_columns].min().to_numpy(dtype="float64")
    maxs = df[feature_columns].max().to_numpy(dtype="float64")
    return torch.tensor([mins, maxs], dtype=torch.double)


def sample_initial_indices(pool_size: int, sample_size: int, seed: int) -> "torch.Tensor":  # type: ignore[name-defined]
    """Sample unique indices from the discrete candidate pool."""
    if torch is None:
        raise RuntimeError("Torch is required to sample indices.")

    if sample_size > pool_size:
        raise ValueError(
            f"Requested {sample_size} points but pool only contains {pool_size} candidates."
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(pool_size, generator=generator)
    return permutation[:sample_size]


def build_acquisition_function(
    model: SingleTaskGP,
    train_X: "torch.Tensor",  # type: ignore[name-defined]
    train_Y: "torch.Tensor",  # type: ignore[name-defined]
    acquisition_type: str,
    acquisition_options: Optional[Dict[str, float]] = None,
) -> "AcquisitionFunction":  # type: ignore[name-defined]
    """Create a BoTorch acquisition function instance."""
    acq_name = acquisition_type.lower()
    options = acquisition_options or {}

    if acq_name in {"expected_improvement", "ei"}:
        best_f = float(train_Y.max().item())
        return ExpectedImprovement(model=model, best_f=best_f)
    if acq_name in {"probability_of_improvement", "pi"}:
        best_f = float(train_Y.max().item())
        xi = float(options.get("xi", 0.0))
        return ProbabilityOfImprovement(model=model, best_f=best_f + xi)
    if acq_name in {"upper_confidence_bound", "ucb"}:
        beta = float(options.get("beta", 0.2))
        return UpperConfidenceBound(model=model, beta=beta)
    if acq_name in {"posterior_mean", "mean"}:
        return PosteriorMean(model=model)

    raise ValueError(
        f"Unsupported acquisition_type '{acquisition_type}'. "
        "Supported values: expected_improvement, probability_of_improvement, "
        "upper_confidence_bound, posterior_mean."
    )


def run_bayesian_optimization(
    oracle: RandomForestOracle,
    candidate_pool: "torch.Tensor",  # type: ignore[name-defined]
    *,
    initial_indices: "torch.Tensor",  # type: ignore[name-defined]
    iterations: int = 20,
    seed: int = 0,
    verbose: bool = True,
    acquisition_type: str = "expected_improvement",
    acquisition_options: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Run a simple sequential Bayesian optimisation loop using BoTorch."""
    if not BOTORCH_AVAILABLE:
        raise RuntimeError(
            "BoTorch (with torch and gpytorch) is required for the optimisation loop."
        )

    torch.manual_seed(seed)

    dtype = candidate_pool.dtype
    device = candidate_pool.device

    initial_indices = initial_indices.to(device=device)
    train_X = candidate_pool[initial_indices].clone()
    train_Y = oracle(train_X).unsqueeze(-1)

    available_mask = torch.ones(candidate_pool.shape[0], dtype=torch.bool, device=device)
    available_mask[initial_indices] = False

    best_so_far_values: List[float] = []
    best_so_far = float("-inf")
    for value in train_Y.squeeze(-1):
        val_float = float(value)
        if val_float > best_so_far:
            best_so_far = val_float
        best_so_far_values.append(best_so_far)

    for iteration in range(iterations):
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()

        acquisition = build_acquisition_function(
            gp, train_X, train_Y, acquisition_type, acquisition_options
        )

        available_indices = available_mask.nonzero(as_tuple=False).view(-1)
        if available_indices.numel() == 0:
            if verbose:
                print("[BO] Candidate pool exhausted; stopping early.")
            break

        candidates = candidate_pool[available_indices]
        with torch.no_grad():
            acquisition_values = acquisition(candidates.unsqueeze(1)).squeeze(-1)

        chosen_position = int(torch.argmax(acquisition_values))
        chosen_index = available_indices[chosen_position]
        candidate = candidate_pool[chosen_index]
        new_y = oracle(candidate).unsqueeze(-1)

        train_X = torch.cat([train_X, candidate.unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)
        new_value = float(new_y.item())
        if new_value > best_so_far:
            best_so_far = new_value
        best_so_far_values.append(best_so_far)
        available_mask[chosen_index] = False

        if verbose:
            print(
                f"[BO] Iteration {iteration + 1:02d} | "
                f"Candidate yield: {new_value:.4f} | "
                f"Best so far: {best_so_far:.4f}"
            )

    best_index = int(train_Y.argmax())
    best_x = train_X[best_index]
    best_y = train_Y[best_index].item()

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "best_so_far": best_so_far_values,
        "best_index": best_index,
        "best_x": best_x,
        "best_y": best_y,
    }


def run_random_sampling(
    oracle: RandomForestOracle,
    candidate_pool: "torch.Tensor",  # type: ignore[name-defined]
    *,
    initial_indices: "torch.Tensor",  # type: ignore[name-defined]
    iterations: int,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, object]:
    """Baseline campaign that samples candidates uniformly at random."""
    if torch is None:
        raise RuntimeError("Torch is required to execute the random baseline.")

    torch.manual_seed(seed)

    dtype = candidate_pool.dtype
    device = candidate_pool.device

    initial_indices = initial_indices.to(device=device)
    train_X = candidate_pool[initial_indices].clone()
    train_Y = oracle(train_X).unsqueeze(-1)

    available_mask = torch.ones(candidate_pool.shape[0], dtype=torch.bool, device=device)
    available_mask[initial_indices] = False
    generator = torch.Generator().manual_seed(seed)

    best_so_far_values: List[float] = []
    best_so_far = float("-inf")

    for value in train_Y.squeeze(-1):
        val_float = float(value)
        if val_float > best_so_far:
            best_so_far = val_float
        best_so_far_values.append(best_so_far)

    for iteration in range(iterations):
        available_indices = available_mask.nonzero(as_tuple=False).view(-1)
        if available_indices.numel() == 0:
            if verbose:
                print("[Random] Candidate pool exhausted; stopping early.")
            break

        draw_pos = int(
            torch.randint(
                low=0,
                high=available_indices.numel(),
                size=(1,),
                generator=generator,
            ).item()
        )
        chosen_index = available_indices[draw_pos]
        candidate = candidate_pool[chosen_index]
        new_y = oracle(candidate).unsqueeze(-1)

        train_X = torch.cat([train_X, candidate.unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)

        new_value = float(new_y.item())
        if new_value > best_so_far:
            best_so_far = new_value
        best_so_far_values.append(best_so_far)
        available_mask[chosen_index] = False

        if verbose:
            print(
                f"[Random] Iteration {iteration + 1:02d} | "
                f"Candidate yield: {new_value:.4f} | "
                f"Best so far: {best_so_far:.4f}"
            )

    best_index = int(train_Y.argmax())
    best_x = train_X[best_index]
    best_y = train_Y[best_index].item()

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "best_so_far": best_so_far_values,
        "best_index": best_index,
        "best_x": best_x,
        "best_y": best_y,
    }


def benchmark_campaigns(
    oracle: RandomForestOracle,
    candidate_pool: "torch.Tensor",  # type: ignore[name-defined]
    *,
    initial_points: int,
    iterations: int,
    repetitions: int,
    seed: int = 0,
    acquisition_type: str = "expected_improvement",
    acquisition_options: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Run repeated BO and random campaigns to enable aggregate benchmarking."""
    if torch is None:
        raise RuntimeError("Torch is required for benchmarking campaigns.")

    bo_histories: List[List[float]] = []
    random_histories: List[List[float]] = []
    bo_full_observed: List[List[float]] = []
    random_full_observed: List[List[float]] = []

    pool_size = candidate_pool.shape[0]
    for rep in range(repetitions):
        bo_seed = seed + rep * 2
        random_seed = seed + rep * 2 + 1

        initial_indices = sample_initial_indices(
            pool_size,
            initial_points,
            seed + rep,
        )

        bo_result = run_bayesian_optimization(
            oracle,
            candidate_pool,
            initial_indices=initial_indices,
            iterations=iterations,
            seed=bo_seed,
            verbose=False,
            acquisition_type=acquisition_type,
            acquisition_options=acquisition_options,
        )
        random_result = run_random_sampling(
            oracle,
            candidate_pool,
            initial_indices=initial_indices,
            iterations=iterations,
            seed=random_seed,
            verbose=False,
        )

        bo_histories.append(bo_result["best_so_far"])  # type: ignore[arg-type]
        random_histories.append(random_result["best_so_far"])  # type: ignore[arg-type]
        bo_full_observed.append(bo_result["train_Y"].squeeze(-1).tolist())  # type: ignore[attr-defined,index]
        random_full_observed.append(random_result["train_Y"].squeeze(-1).tolist())  # type: ignore[attr-defined,index]

    all_histories = bo_histories + random_histories
    if not all_histories:
        raise RuntimeError("No histories recorded during benchmarking.")

    max_length = max(len(history) for history in all_histories)
    evaluation_axis = list(range(1, max_length + 1))

    return {
        "bo_histories": bo_histories,
        "random_histories": random_histories,
        "bo_observations": bo_full_observed,
        "random_observations": random_full_observed,
        "evaluations": evaluation_axis,
    }


def plot_benchmark_results(
    evaluations: Iterable[int],
    bo_histories: List[List[float]],
    random_histories: List[List[float]],
    output_path: Path,
) -> None:
    """Create and save a comparison plot for BO vs random sampling campaigns."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError(
            "Matplotlib is required to create benchmark visualisations."
        )

    evaluations_arr = np.asarray(list(evaluations), dtype=float)

    def summarise(data: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        return mean, std

    bo_mean, bo_std = summarise(bo_histories)
    random_mean, random_std = summarise(random_histories)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(evaluations_arr, bo_mean, label="Bayesian optimisation", color="#1f77b4")
    if len(bo_histories) > 1:
        ax.fill_between(
            evaluations_arr,
            bo_mean - bo_std,
            bo_mean + bo_std,
            color="#1f77b4",
            alpha=0.2,
        )

    ax.plot(evaluations_arr, random_mean, label="Random sampling", color="#ff7f0e")
    if len(random_histories) > 1:
        ax.fill_between(
            evaluations_arr,
            random_mean - random_std,
            random_mean + random_std,
            color="#ff7f0e",
            alpha=0.2,
        )

    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best predicted yield")
    ax.set_title("Bayesian optimisation vs random sampling")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def compute_additional_metrics(
    evaluations: Iterable[int],
    bo_histories: List[List[float]],
    random_histories: List[List[float]],
    bo_observations: List[List[float]],
    random_observations: List[List[float]],
    global_best: float,
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Compute additional benchmarking metrics across runs."""

    def pad_to_length(data: List[List[float]], length: int) -> np.ndarray:
        array = np.full((len(data), length), np.nan, dtype=float)
        for row_idx, row in enumerate(data):
            row_len = min(length, len(row))
            array[row_idx, :row_len] = row[:row_len]
        return array

    evaluations_list = list(evaluations)
    max_length = len(evaluations_list)

    bo_hist = pad_to_length(bo_histories, max_length)
    random_hist = pad_to_length(random_histories, max_length)

    bo_obs = pad_to_length(bo_observations, max_length)
    random_obs = pad_to_length(random_observations, max_length)

    def simple_regret(hist: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, global_best - hist)

    def cumulative_regret(obs: np.ndarray) -> np.ndarray:
        regrets = np.maximum(0.0, global_best - obs)
        return np.nansum(regrets, axis=1)

    def auc(hist: np.ndarray) -> np.ndarray:
        return np.nansum(np.nan_to_num(hist, nan=0.0), axis=1)

    def time_to_threshold(hist: np.ndarray) -> np.ndarray:
        mask = hist >= threshold
        indices = np.argmax(mask, axis=1)
        exists = mask[np.arange(mask.shape[0]), indices]
        result = np.where(exists, indices + 1, np.nan)
        return result

    def hit_rate(hist: np.ndarray) -> float:
        success = np.any(hist >= threshold, axis=1)
        return float(np.nanmean(success))

    bo_final_best = np.nanmax(bo_hist, axis=1)
    random_final_best = np.nanmax(random_hist, axis=1)
    bo_simple_regret = simple_regret(bo_final_best)
    random_simple_regret = simple_regret(random_final_best)

    cumulative_regret_bo = cumulative_regret(bo_obs)
    cumulative_regret_random = cumulative_regret(random_obs)
    auc_bo = auc(bo_hist)
    auc_random = auc(random_hist)
    ttt_bo = time_to_threshold(bo_hist)
    ttt_random = time_to_threshold(random_hist)

    summary = {
        "simple_regret": {
            "bo_mean": float(np.nanmean(bo_simple_regret)),
            "bo_std": float(np.nanstd(bo_simple_regret)),
            "random_mean": float(np.nanmean(random_simple_regret)),
            "random_std": float(np.nanstd(random_simple_regret)),
        },
        "cumulative_regret": {
            "bo_mean": float(np.nanmean(cumulative_regret_bo)),
            "bo_std": float(np.nanstd(cumulative_regret_bo)),
            "random_mean": float(np.nanmean(cumulative_regret_random)),
            "random_std": float(np.nanstd(cumulative_regret_random)),
        },
        "auc_best_so_far": {
            "bo_mean": float(np.nanmean(auc_bo)),
            "bo_std": float(np.nanstd(auc_bo)),
            "random_mean": float(np.nanmean(auc_random)),
            "random_std": float(np.nanstd(auc_random)),
        },
        "time_to_threshold": {
            "bo_mean": float(np.nanmean(ttt_bo)),
            "bo_std": float(np.nanstd(ttt_bo)),
            "random_mean": float(np.nanmean(ttt_random)),
            "random_std": float(np.nanstd(ttt_random)),
        },
        "hit_rate": {
            "bo": hit_rate(bo_hist),
            "random": hit_rate(random_hist),
        },
    }

    per_run = {
        "simple_regret": {"bo": bo_simple_regret, "random": random_simple_regret},
        "cumulative_regret": {
            "bo": cumulative_regret_bo,
            "random": cumulative_regret_random,
        },
        "auc_best_so_far": {"bo": auc_bo, "random": auc_random},
        "time_to_threshold": {"bo": ttt_bo, "random": ttt_random},
    }

    return {"summary": summary, "per_run": per_run}


def build_full_data_df(
    bo_histories: List[List[float]],
    random_histories: List[List[float]],
    bo_observations: List[List[float]],
    random_observations: List[List[float]],
    *,
    bo_label: str = "BO",
    random_label: str = "Random",
) -> pd.DataFrame:
    """Assemble per-evaluation data for every run into a single DataFrame."""

    records: List[Dict[str, object]] = []

    def append_records(
        method: str,
        histories: List[List[float]],
        observations: List[List[float]],
    ) -> None:
        for run_idx, (best_seq, obs_seq) in enumerate(
            zip(histories, observations), start=1
        ):
            length = min(len(best_seq), len(obs_seq))
            for step in range(length):
                records.append(
                    {
                        "method": method,
                        "run": run_idx,
                        "evaluation": step + 1,
                        "best_so_far": float(best_seq[step]),
                        "observation": float(obs_seq[step]),
                    }
                )

    append_records(bo_label, bo_histories, bo_observations)
    append_records(random_label, random_histories, random_observations)

    return pd.DataFrame.from_records(records)

def print_feature_importances(feature_importances: List[Tuple[float, str]]) -> None:
    print("\nFeature importances:")
    for importance, name in feature_importances:
        print(f" - {name}: {importance:.4f}")


def print_bo_summary(
    feature_columns: List[str],
    bo_results: Dict[str, object],
) -> None:
    if torch is None:
        raise RuntimeError("Torch is required to summarise BO results.")

    best_x: torch.Tensor = bo_results["best_x"]  # type: ignore[assignment]
    best_y: float = bo_results["best_y"]  # type: ignore[assignment]

    print("\nBayesian optimisation summary:")
    print(f" - Total evaluations: {int(bo_results['train_Y'].shape[0])}")  # type: ignore[index]
    print(f" - Best predicted yield: {best_y:.4f}")

    print(" - Best candidate (feature values):")
    for name, value in zip(feature_columns, best_x.tolist()):
        print(f"     {name}: {value:.4f}")

#%%
if __name__ == "__main__":
    df_tot = load_ugi_series()
    report_duplicates(df_tot)

    df_ml, feature_columns = prepare_features(df_tot, target="yield")
    X = df_ml[feature_columns]
    y = df_ml["yield"]

    model, metrics, feature_importances = train_random_forest(X, y)

    print("\nRandomForestRegressor metrics:")
    print(f" - OOB score: {metrics['oob_score']:.4f}")
    print(f" - Test MSE: {metrics['mse']:.6f}")
    print(f" - Test RMSE: {metrics['rmse']:.6f}")
    print(f" - Test R^2: {metrics['r2']:.4f}")

    print_feature_importances(feature_importances)

    if BOTORCH_AVAILABLE:
        oracle = RandomForestOracle(model, feature_columns)
        candidate_pool_df = df_ml[feature_columns].drop_duplicates().reset_index(drop=True)
        candidate_pool = torch.tensor(
            candidate_pool_df.to_numpy(dtype="float64"),
            dtype=torch.double,
        )

        pool_size = candidate_pool.shape[0]
        if pool_size == 0:
            raise RuntimeError("Candidate pool is empty; cannot run optimisation.")

        base_initial_points = min(16, pool_size)
        if base_initial_points == pool_size and pool_size > 1:
            base_initial_points -= 1
        base_iterations = min(20, max(0, pool_size - base_initial_points))

        acquisition_strategies = [
            {
                "name": "expected_improvement",
                "label": "Expected Improvement",
                "options": {},
            },
            {
                "name": "probability_of_improvement",
                "label": "Probability of Improvement",
                "options": {},
            },
            {
                "name": "upper_confidence_bound",
                "label": "Upper Confidence Bound",
                "options": {"beta": 0.15},
            },
        ]

        primary_strategy = acquisition_strategies[0]
        initial_indices = sample_initial_indices(pool_size, base_initial_points, seed=1)
        bo_results = run_bayesian_optimization(
            oracle,
            candidate_pool,
            initial_indices=initial_indices,
            iterations=base_iterations,
            seed=42,
            acquisition_type=primary_strategy["name"],
            acquisition_options=primary_strategy.get("options"),
        )
        print(f"\nPrimary BO acquisition: {primary_strategy['label']}")
        print_bo_summary(feature_columns, bo_results)

        benchmark_config = {
            "initial_points": min(15, pool_size - 1) if pool_size > 1 else 1,
            "iterations": min(25, max(0, pool_size - 1)),
            "repetitions": 5,
            "seed": 757575,
        }
        global_best = float(df_ml["yield"].max())
        threshold = float(df_ml["yield"].quantile(0.9, interpolation="linear"))
        full_data_frames: List[pd.DataFrame] = []

        for strategy in acquisition_strategies:
            label = strategy.get("label", strategy["name"])
            options = strategy.get("options") or {}

            print("\n" + "=" * 60)
            print(f"Acquisition: {label}")
            print("=" * 60)

            benchmark = benchmark_campaigns(
                oracle,
                candidate_pool,
                initial_points=benchmark_config["initial_points"],
                iterations=benchmark_config["iterations"],
                repetitions=benchmark_config["repetitions"],
                seed=benchmark_config["seed"],
                acquisition_type=strategy["name"],
                acquisition_options=options,
            )

            bo_final = np.array([history[-1] for history in benchmark["bo_histories"]])
            random_final = np.array(
                [history[-1] for history in benchmark["random_histories"]]
            )

            print(
                "Benchmark summary "
                f"(initial={benchmark_config['initial_points']}, "
                f"iterations={benchmark_config['iterations']}, "
                f"repetitions={benchmark_config['repetitions']}):"
            )
            print(
                f" - BO final best (mean ± std): "
                f"{bo_final.mean():.4f} ± {bo_final.std():.4f}"
            )
            print(
                f" - Random final best (mean ± std): "
                f"{random_final.mean():.4f} ± {random_final.std():.4f}"
            )

            metrics = compute_additional_metrics(
                benchmark["evaluations"],
                benchmark["bo_histories"],
                benchmark["random_histories"],
                benchmark["bo_observations"],
                benchmark["random_observations"],
                global_best=global_best,
                threshold=threshold,
            )

            print(
                f"Additional metrics (global best={global_best:.4f}, "
                f"90th percentile threshold={threshold:.4f}):"
            )
            for metric_name, values in metrics["summary"].items():
                display_name = metric_name.replace("_", " ").title()
                if metric_name == "hit_rate":
                    print(
                        f" - {display_name}: "
                        f"BO={values['bo']:.3f}, Random={values['random']:.3f}"
                    )
                else:
                    print(
                        f" - {display_name}: "
                        f"BO={values['bo_mean']:.4f} ± {values['bo_std']:.4f} | "
                        f"Random={values['random_mean']:.4f} ± "
                        f"{values['random_std']:.4f}"
                    )

            full_data_frames.append(
                build_full_data_df(
                    benchmark["bo_histories"],
                    benchmark["random_histories"],
                    benchmark["bo_observations"],
                    benchmark["random_observations"],
                    bo_label=f"BO ({label})",
                    random_label=f"Random ({label})",
                )
            )

            if MATPLOTLIB_AVAILABLE:
                slug = (
                    label.lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("(", "")
                    .replace(")", "")
                )
                output_path = Path(f"bo_vs_random_{slug}.png")
                plot_benchmark_results(
                    benchmark["evaluations"],
                    benchmark["bo_histories"],
                    benchmark["random_histories"],
                    output_path,
                )
                print(f"Benchmark plot saved to {output_path.resolve()}")
            else:
                print(
                    "\nMatplotlib not available; skipping benchmark visualisation.\n"
                    f"Original import error: {MATPLOTLIB_IMPORT_ERROR}"
                )

        if full_data_frames:
            full_data_df = pd.concat(full_data_frames, ignore_index=True)
            full_data_path = Path("benchmark_full_data.csv")
            full_data_df.to_csv(full_data_path, index=False)
            print(f"\nFull benchmark data saved to {full_data_path.resolve()}")
            print("Preview of benchmark data:")
            print(full_data_df.head())
    else:
        print(
            "\nBoTorch not available in this environment. "
            "Install torch, gpytorch, and botorch to enable Bayesian optimisation.\n"
            f"Original import error: {BOTORCH_IMPORT_ERROR}"
        )
#%%
