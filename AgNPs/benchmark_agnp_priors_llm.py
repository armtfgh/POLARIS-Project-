
#!/usr/bin/env python
"""
Closed-loop benchmark on AgNP dataset with LLM-shaped priors (POLARIS).

Usage:
  python benchmark_agnp_priors_llm.py \
    --oracle agnp_oracle_out/oracle.pkl \
    --data_csv /mnt/data/AgNP_dataset.csv \
    --out_dir agnp_bench_llm \
    --priors no_prior,heuristic,llm_good,llm_bad,random_prior \
    --init_modes good,bad,random \
    --n_reps 3 \
    --budget 120 \
    --batch_size 10

Notes:
- Priors are implemented as *scores over candidates*; they bias acquisition.
- LLM hooks are stubbed: once GOOD_BULLETS/BAD_BULLETS are defined,
  you can plug your LLM function into `llm_readout`.
"""
import argparse, os, json, time, random, math
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from agnp_oracle import AgNPOracle
from priors.bullets_template import GOOD_BULLETS, BAD_BULLETS

RNG = np.random.default_rng(42)

FEATURES = ['QAgNO3(%)','Qpva(%)','Qtsc(%)','Qseed(%)','Qtot(uL/min)']

def latin_hypercube(n, bounds):
    """Simple LHS in 5D box."""
    d = len(bounds)
    cut = np.linspace(0, 1, n + 1)
    u = RNG.uniform(size=(n, d))
    a = cut[:n]
    b = cut[1:n+1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)
    for j in range(d):
        order = RNG.permutation(n)
        H[:, j] = rdpoints[order, 0]
    X = []
    keys = list(bounds.keys())
    for i in range(n):
        row = {}
        for j, k in enumerate(keys):
            lo, hi = bounds[k]
            row[k] = lo + H[i, j] * (hi - lo)
        X.append(row)
    return X

def heuristic_score(c):
    """Cheap prior: upweight higher QAgNO3 and lower Qseed as per paper hints."""
    # Normalize to [0,1] within bounds, then score.
    # Score = w1*QAgNO3_norm + w2*(1-Qseed_norm) + small bonuses for Qtotal high, Qtsc mid, Qpva mid
    b = DEFAULT_BOUNDS
    def norm(k, v):
        lo, hi = b[k]
        return (v - lo)/(hi - lo)
    s  = 0.8*norm('QAgNO3(%)', c['QAgNO3(%)'])
    s += 0.8*(1.0 - norm('Qseed(%)', c['Qseed(%)']))
    s += 0.2*norm('Qtot(uL/min)', c['Qtot(uL/min)'])
    s += 0.1*(1.0 - abs(norm('Qtsc(%)', c['Qtsc(%)']) - 0.3))  # prefer mid-ish TSC
    s += 0.05*(1.0 - abs(norm('Qpva(%)', c['Qpva(%)']) - 0.4)) # slightly higher PVA
    return float(s)

def llm_readout(schema, bullets, llm_fn=None, cache=None):
    """
    Placeholder: score conditions by simple keyword-aligned heuristic when LLM is absent.
    Replace `score_rule` with a call to your LLM scoring function if needed.
    """
    def score_rule(c):
        # mimic a soft preference similar to heuristic_score
        return heuristic_score(c)
    return score_rule

def propose_candidates(bounds, n, mode="random"):
    if mode == "random":
        return latin_hypercube(n, bounds)
    elif mode == "exploit":
        # small random jitter around current best guess center (handled upstream)
        return latin_hypercube(n, bounds)
    else:
        return latin_hypercube(n, bounds)

def run_once(args):
    oracle = AgNPOracle(args.oracle, oob_mode="penalize", noise_std=0.0)
    global DEFAULT_BOUNDS; DEFAULT_BOUNDS = oracle.bounds

    df = pd.read_csv(args.data_csv)
    # Seed pools for good/bad init based on historical loss
    df_sorted = df.sort_values("loss")
    good_pool = df_sorted.head(50)[FEATURES].to_dict(orient="records")
    bad_pool  = df_sorted.tail(50)[FEATURES].to_dict(orient="records")

    priors = [p.strip() for p in args.priors.split(",") if p.strip()]
    init_modes = [m.strip() for m in args.init_modes.split(",") if m.strip()]

    results = {}

    for prior in priors:
        for init in init_modes:
            key = f"{prior}__{init}"
            history = []
            # init design
            if init == "good":
                X0 = RNG.choice(good_pool, size=args.batch_size, replace=False).tolist()
            elif init == "bad":
                X0 = RNG.choice(bad_pool, size=args.batch_size, replace=False).tolist()
            else:
                X0 = propose_candidates(DEFAULT_BOUNDS, args.batch_size, "random")

            y0 = oracle.evaluate_batch(X0)
            for x,y in zip(X0,y0):
                history.append({"t":0, **x, "loss": y})

            budget_left = args.budget - len(X0)

            # prior scoring function
            if prior == "no_prior":
                prior_fn = lambda c: 0.0
            elif prior == "heuristic":
                prior_fn = heuristic_score
            elif prior == "random_prior":
                prior_fn = lambda c: float(RNG.uniform())
            elif prior == "llm_good":
                prior_fn = llm_readout(FEATURES, GOOD_BULLETS)
            elif prior == "llm_bad":
                prior_fn = llm_readout(FEATURES, BAD_BULLETS)
            else:
                prior_fn = lambda c: 0.0

            # simple BO-like loop with prior-biased scoring over LHS candidates
            t = 1
            while budget_left > 0:
                batch_n = min(args.batch_size, budget_left)
                C = propose_candidates(DEFAULT_BOUNDS, n=200, mode="random")
                # surrogate via k-NN on history (cheap) to get mean loss + UCB-ish uncertainty
                H = pd.DataFrame(history)
                XH = H[FEATURES].values
                yH = H["loss"].values
                # scale features roughly for distance calc
                scales = XH.std(axis=0); scales[scales==0]=1.0
                def pred_mu_sigma(c):
                    x = np.array([[c[k] for k in FEATURES]])
                    d = np.sqrt(((XH - x)/scales)**2).sum(axis=1)
                    w = 1.0/(d + 1e-6)
                    w /= w.sum()
                    mu = float((w * yH).sum())
                    # pseudo-uncertainty: distance-weighted variance
                    var = float(((yH - mu)**2 * w).sum())
                    return mu, math.sqrt(max(var,1e-12))

                scored = []
                for c in C:
                    mu, sig = pred_mu_sigma(c)
                    prior_boost = prior_fn(c)
                    acq = -mu + 0.25*sig + 0.5*prior_boost  # maximize acq (minimize mu, add uncertainty, plus prior)
                    scored.append((acq, c))
                scored.sort(key=lambda t: t[0], reverse=True)
                Xnext = [c for _,c in scored[:batch_n]]

                y = oracle.evaluate_batch(Xnext)
                for x_i, y_i in zip(Xnext, y):
                    history.append({"t":t, **x_i, "loss": y_i})
                budget_left -= batch_n
                t += 1

            R = pd.DataFrame(history)
            R["cum_best"] = R["loss"].cummin()
            results[key] = R

    # write out
    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for key,dfres in results.items():
        dfres.to_csv(os.path.join(args.out_dir, f"{key}.csv"), index=False)
        summary.append({
            "run": key,
            "final_best": float(dfres["cum_best"].min()),
            "first10_best": float(dfres.head(10)["loss"].min()),
            "mean_loss": float(dfres["loss"].mean())
        })
    with open(os.path.join(args.out_dir,"summary.json"),"w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", default="agnp_bench_llm")
    ap.add_argument("--priors", default="no_prior,heuristic,llm_good,llm_bad,random_prior")
    ap.add_argument("--init_modes", default="good,bad,random")
    ap.add_argument("--n_reps", type=int, default=1)  # repetitions can be handled externally
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=10)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_once(args)
