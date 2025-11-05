
#%%
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
    from botorch.acquisition import ExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
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


def run_bayesian_optimization(
    oracle: RandomForestOracle,
    bounds: "torch.Tensor",  # type: ignore[name-defined]
    *,
    initial_points: int = 16,
    iterations: int = 20,
    seed: int = 0,
    verbose: bool = True,
    initial_train_X: "torch.Tensor" | None = None,  # type: ignore[name-defined]
    initial_train_Y: "torch.Tensor" | None = None,  # type: ignore[name-defined]
) -> Dict[str, object]:
    """Run a simple sequential Bayesian optimisation loop using BoTorch."""
    if not BOTORCH_AVAILABLE:
        raise RuntimeError(
            "BoTorch (with torch and gpytorch) is required for the optimisation loop."
        )

    torch.manual_seed(seed)

    dtype = torch.double
    device = bounds.device

    if initial_train_X is not None and initial_train_Y is not None:
        train_X = initial_train_X.to(device=device, dtype=dtype).clone()
        train_Y = initial_train_Y.to(device=device, dtype=dtype).clone()
    else:
        sobol = draw_sobol_samples(bounds=bounds, n=initial_points, q=1, seed=seed)
        train_X = sobol.squeeze(1).to(device=device, dtype=dtype)
        train_Y = oracle(train_X).unsqueeze(-1)

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

        best_f = train_Y.max()
        acquisition = ExpectedImprovement(model=gp, best_f=best_f)

        candidate, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=128,
            options={"batch_limit": 5, "maxiter": 200},
        )
        candidate = candidate.to(dtype=dtype, device=device).squeeze(0)
        new_y = oracle(candidate).unsqueeze(-1)

        train_X = torch.cat([train_X, candidate.unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)
        new_value = float(new_y.item())
        if new_value > best_so_far:
            best_so_far = new_value
        best_so_far_values.append(best_so_far)

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
    bounds: "torch.Tensor",  # type: ignore[name-defined]
    *,
    initial_points: int,
    iterations: int,
    seed: int = 0,
    verbose: bool = True,
    initial_train_X: "torch.Tensor" | None = None,  # type: ignore[name-defined]
    initial_train_Y: "torch.Tensor" | None = None,  # type: ignore[name-defined]
) -> Dict[str, object]:
    """Baseline campaign that samples candidates uniformly at random."""
    if torch is None:
        raise RuntimeError("Torch is required to execute the random baseline.")

    torch.manual_seed(seed)

    dtype = torch.double
    device = bounds.device
    lower = bounds[0].to(dtype=dtype, device=device)
    upper = bounds[1].to(dtype=dtype, device=device)
    span = upper - lower

    def sample(num: int) -> "torch.Tensor":  # type: ignore[name-defined]
        return lower + span * torch.rand(num, lower.shape[0], dtype=dtype, device=device)

    if initial_train_X is not None and initial_train_Y is not None:
        train_X = initial_train_X.to(device=device, dtype=dtype).clone()
        train_Y = initial_train_Y.to(device=device, dtype=dtype).clone()
    else:
        train_X = sample(initial_points)
        train_Y = oracle(train_X).unsqueeze(-1)

    best_so_far_values: List[float] = []
    best_so_far = float("-inf")

    for value in train_Y.squeeze(-1):
        val_float = float(value)
        if val_float > best_so_far:
            best_so_far = val_float
        best_so_far_values.append(best_so_far)

    for iteration in range(iterations):
        candidate = sample(1)
        new_y = oracle(candidate).unsqueeze(-1)

        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)

        new_value = float(new_y.item())
        if new_value > best_so_far:
            best_so_far = new_value
        best_so_far_values.append(best_so_far)

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
    bounds: "torch.Tensor",  # type: ignore[name-defined]
    *,
    initial_points: int,
    iterations: int,
    repetitions: int,
    seed: int = 0,
) -> Dict[str, object]:
    """Run repeated BO and random campaigns to enable aggregate benchmarking."""
    if torch is None:
        raise RuntimeError("Torch is required for benchmarking campaigns.")

    bo_histories: List[List[float]] = []
    random_histories: List[List[float]] = []

    for rep in range(repetitions):
        bo_seed = seed + rep * 2
        random_seed = seed + rep * 2 + 1

        initial_design = draw_sobol_samples(
            bounds=bounds,
            n=initial_points,
            q=1,
            seed=seed + rep,
        ).squeeze(1)
        initial_design = initial_design.to(dtype=torch.double, device=bounds.device)
        initial_values = oracle(initial_design).unsqueeze(-1)

        bo_result = run_bayesian_optimization(
            oracle,
            bounds,
            initial_points=initial_points,
            iterations=iterations,
            seed=bo_seed,
            verbose=False,
            initial_train_X=initial_design,
            initial_train_Y=initial_values,
        )
        random_result = run_random_sampling(
            oracle,
            bounds,
            initial_points=initial_points,
            iterations=iterations,
            seed=random_seed,
            verbose=False,
            initial_train_X=initial_design,
            initial_train_Y=initial_values,
        )

        bo_histories.append(bo_result["best_so_far"])  # type: ignore[arg-type]
        random_histories.append(random_result["best_so_far"])  # type: ignore[arg-type]

    evaluation_axis = list(range(1, initial_points + iterations + 1))

    return {
        "bo_histories": bo_histories,
        "random_histories": random_histories,
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
    plt.close(fig)


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
        bounds = compute_feature_bounds(df_ml, feature_columns)
        oracle = RandomForestOracle(model, feature_columns)
        bo_results = run_bayesian_optimization(
            oracle,
            bounds,
            initial_points=16,
            iterations=20,
            seed=1,
        )
        print_bo_summary(feature_columns, bo_results)

        benchmark_config = {
            "initial_points": 16,
            "iterations": 20,
            "repetitions": 5,
            "seed": 123,
        }
        benchmark = benchmark_campaigns(
            oracle,
            bounds,
            initial_points=benchmark_config["initial_points"],
            iterations=benchmark_config["iterations"],
            repetitions=benchmark_config["repetitions"],
            seed=benchmark_config["seed"],
        )

        bo_final = np.array([history[-1] for history in benchmark["bo_histories"]])
        random_final = np.array(
            [history[-1] for history in benchmark["random_histories"]]
        )

        print(
            "\nBenchmark summary "
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

        if MATPLOTLIB_AVAILABLE:
            output_path = Path("bo_vs_random_benchmark.png")
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
    else:
        print(
            "\nBoTorch not available in this environment. "
            "Install torch, gpytorch, and botorch to enable Bayesian optimisation.\n"
            f"Original import error: {BOTORCH_IMPORT_ERROR}"
        )
#%%
