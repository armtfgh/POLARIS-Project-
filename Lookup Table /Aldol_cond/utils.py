
from __future__ import annotations
from typing import List
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd
import numpy as np
from prior_gp import Prior, GPWithPriorMean, fit_residual_gp, alignment_on_obs
from franke import franke_hard_torch, franke_torch

# -------------------- Device / dtype --------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
# -------------------- Methods ---------------------------

def run_random(n_evals: int, seed: int = 0, noise_sd: float = 0.0, obj=franke_torch) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.random((n_evals, 2))).to(DEVICE, DTYPE)
    Y = obj(X, noise_sd=noise_sd)
    best = -1e9
    rec = []
    for t in range(n_evals):
        best = max(best, float(Y[:t+1].max().item()))
        rec.append({"iter": t, "x1": float(X[t,0]), "x2": float(X[t,1]), "y": float(Y[t].item()), "best_so_far": best, "method": "random"})
    return pd.DataFrame(rec)


def run_baseline_ei(n_init: int, n_iter: int, seed: int = 0, noise_sd: float = 0.0, obj=franke_torch) -> pd.DataFrame:
    torch.manual_seed(seed)
    if USE_CUDA: torch.cuda.manual_seed_all(seed)
    bounds = torch.stack([torch.zeros(2, device=DEVICE, dtype=DTYPE), torch.ones(2, device=DEVICE, dtype=DTYPE)])

    X = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=seed).squeeze(1).to(DEVICE)
    Y = obj(X, noise_sd=noise_sd)

    rec = []
    best = float(Y.max().item())
    for i in range(X.shape[0]):
        rec.append({"iter": i - X.shape[0], "x1": float(X[i,0]), "x2": float(X[i,1]), "y": float(Y[i].item()), "best_so_far": best, "method": "baseline_ei"})

    for t in range(n_iter):
        gp = SingleTaskGP(X, Y).to(DEVICE)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(DEVICE)
        fit_gpytorch_mll(mll)
        best_f = float(Y.max().item())
        print(f"Baseline Ei: iteration {t+1}, Best Y: {best_f} ")
        EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
        x_next, _ = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=20, raw_samples=1024, options={"maxiter": 200, "batch_limit": 5})
        x_next = x_next.squeeze(0)
        y_next = obj(x_next.unsqueeze(0), noise_sd=noise_sd).squeeze(0)
        X = torch.cat([X, x_next.unsqueeze(0)])
        Y = torch.cat([Y, y_next.unsqueeze(0)])
        best = max(best, float(y_next.item()))
        rec.append({"iter": t, "x1": float(x_next[0]), "x2": float(x_next[1]), "y": float(y_next.item()), "best_so_far": best, "method": "baseline_ei"})
    return pd.DataFrame(rec)



def run_baseline_ei_llm_pool(
    n_init: int,
    n_iter: int,
    seed: int = 0,
    noise_sd: float = 0.0,
    pool_base: int = 256,
    k_anchors: int = 5,
    obj=franke_torch,
    history_cb=None
) -> pd.DataFrame:
    """Vanilla GP + EI, but EI is evaluated on a curated pool that includes
    LLM-proposed anchors. Prior is FLAT (identical surrogate to baseline).
    """
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)
    bounds = torch.stack([
        torch.zeros(2, device=DEVICE, dtype=DTYPE),
        torch.ones(2, device=DEVICE, dtype=DTYPE)
    ])

    X = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=seed).squeeze(1).to(DEVICE)
    Y = obj(X, noise_sd=noise_sd)

    rec, hist = [], []
    best = float(Y.max().item())
    for i in range(X.shape[0]):
        row = {"iter": i - X.shape[0], "x1": float(X[i,0]), "x2": float(X[i,1]),
               "y": float(Y[i].item()), "best_so_far": best, "method": "baseline_ei_llm_pool"}
        rec.append(row); hist.append(row)

    for t in range(n_iter):
        gp = SingleTaskGP(X, Y).to(DEVICE)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(DEVICE)
        fit_gpytorch_mll(mll)
        best_f = float(Y.max().item())
        print(f"Baseline Ei with pooling: iteration {t+1}, Best Y: {best_f} ")
        EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)


        # Build structured context and request LLM policy
        hist_df = pd.DataFrame(hist)
        ctx = make_ei_context(gp, X, Y, bounds)
        resp = llm_propose_anchors(
        hist_df,
        k=k_anchors,
        context=ctx,
        sys_prompt=SYS_PROMPT_ANCHORING_V2,
        few_shots=FEW_SHOT_EXAMPLES  # optional; drop this line if you want zero-shot
        )

        min_spacing = 0.0
        tr = None
        anchors = []
        if isinstance(resp, dict):
            anchors = resp.get("anchors", [])
            min_spacing = float(resp.get("min_spacing", 0.0) or 0.0)
            tr = resp.get("trust_region", None)
        else:
            anchors = resp  # backward-compat

        # Base pool: sample inside trust-region if provided
        base_bounds = bounds
        if tr and isinstance(tr, (list, tuple)) and len(tr) == 2:
            low = torch.tensor([tr[0][0], tr[1][0]], device=DEVICE, dtype=DTYPE)
            high = torch.tensor([tr[0][1], tr[1][1]], device=DEVICE, dtype=DTYPE)
            base_bounds = torch.stack([low, high])

        pool = draw_sobol_samples(bounds=base_bounds, n=pool_base, q=1, seed=seed + t).squeeze(1).to(DEVICE)

        # Enforce spacing among anchors and append
        if anchors:
            anchors = enforce_min_spacing(anchors, min_spacing)
            if anchors:
                A = torch.tensor(anchors, device=DEVICE, dtype=DTYPE).clamp(0.0, 1.0)
                pool = torch.cat([pool, A], dim=0)

        with torch.no_grad():
            pool_q = pool.unsqueeze(1)  # (N,1,d)
            ei_vals = EI(pool_q).reshape(-1)
            idx = int(torch.argmax(ei_vals))
            x_next = pool[idx]

        y_next = obj(x_next.unsqueeze(0), noise_sd=noise_sd).squeeze(0)
        X = torch.cat([X, x_next.unsqueeze(0)])
        Y = torch.cat([Y, y_next.unsqueeze(0)])
        best = max(best, float(y_next.item()))
        row = {"iter": t, "x1": float(x_next[0]), "x2": float(x_next[1]),
               "y": float(y_next.item()), "best_so_far": best, "method": "baseline_ei_llm_pool"}
        rec.append(row); hist.append(row)
        if history_cb is not None:
            history_cb(pd.DataFrame(hist))

    return pd.DataFrame(rec)




def run_many_seeds(seeds: List[int], n_init: int = 6, n_iter: int = 25, noise_sd: float = 0.0) -> pd.DataFrame:
    dfs = []
    for s in seeds:
        df = compare_methods(n_init=n_init, n_iter=n_iter, seed=s, noise_sd=noise_sd)
        df["seed"] = s
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def aggregate_by_seed(history_df: pd.DataFrame) -> pd.DataFrame:
    return (history_df.groupby(["method", "iter"], as_index=False)
            .agg(mean_best=("best_so_far", "mean"), std_best=("best_so_far", "std")))


def plot_runs(history_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    plt.figure()
    for name, grp in history_df.groupby("method"):
        g = grp.copy()
        g = g[g["iter"] >= 0]
        plt.plot(g["iter"].values, g["best_so_far"].values, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Best so far (Franke)")
    plt.title("Random vs Baseline EI vs Hybrid (dynamic prior + curated pool)")
    plt.legend(); plt.show()


def plot_runs_mean(history_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    agg = aggregate_by_seed(history_df)
    plt.figure()
    for name, grp in agg.groupby("method"):
        g = grp.copy()
        g = g[g["iter"] >= 0]
        plt.plot(g["iter"].values, g["mean_best"].values, label=name)
        if not g["std_best"].isna().all():
            lo = g["mean_best"].values - g["std_best"].values
            hi = g["mean_best"].values + g["std_best"].values
            plt.fill_between(g["iter"].values, lo, hi, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("Best so far (mean ± sd)")
    plt.title("Averaged over seeds")
    plt.legend(); plt.show()
