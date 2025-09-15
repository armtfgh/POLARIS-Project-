
#!/usr/bin/env python
# benchmark_aldol_priors_llm.py
"""
Closed-loop benchmark on Aldol dataset with:
- Real LLM-shaped priors via OpenAI API (good and intentionally-bad prompts)
- NoPrior / Heuristic / RandomPrior baselines
- Controlled initial seeding: good (top-yld), bad (bottom-yld), random

Usage:
  python benchmark_aldol_priors_llm.py \
    --oracle aldol_oracle_out/oracle.pkl \
    --data_csv "Aldol Condensation.csv" \
    --out_dir aldol_bench_llm \
    --priors no_prior,heuristic,llm_good,llm_bad,random_prior \
    --init_modes good,bad,random \
    --n_reps 5 --iters 20 --n_init 6 --cand_pool 600 \
    --llm_model gpt-4o-mini

Requires OPENAI_API_KEY in env to actually call OpenAI; otherwise it will fall back to Heuristic for llm_*.
"""
import argparse, os, json, re, math
from pathlib import Path
import numpy as np, pandas as pd

import os
os.environ["MPLBACKEND"] = "Agg"  # force headless backend
import matplotlib
matplotlib.use("Agg")

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

try:
    import joblib
except Exception:
    joblib = None

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import language_shaped_prior_llm as lsp

# ---------------------- utilities ----------------------

def load_oracle(p):
    if joblib is None:
        raise ImportError("joblib required: pip install joblib")
    return joblib.load(p)

def derive_schema(csv_path: str):
    df = pd.read_csv(csv_path).dropna()
    feats = [c for c in df.columns if c != "yld"]
    bounds = {c:(float(df[c].min()), float(df[c].max())) for c in feats}
    roles = {"temp":"temperature", "time":"time", "moleq2":"equivalents", "moleq3":"equivalents"}
    cont = [lsp.ContinuousVar(c, bounds[c][0], bounds[c][1], role=roles.get(c,"")) for c in feats]
    return lsp.Schema(continuous=cont, categorical=[]), df

def sample_candidates(n, rng, schema: lsp.Schema):
    rows = []
    for _ in range(n):
        r = {}
        for v in schema.continuous:
            r[v.name] = rng.uniform(v.low, v.high)
        rows.append(r)
    return pd.DataFrame(rows)

def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

from scipy.integrate import trapezoid

def area_under_curve(curve):
    import numpy as np
    x = np.arange(len(curve))
    return trapezoid(curve, x)

# ---------------------- LLM prior generation ----------------------

SYS = (
"You are a domain scientist. Given variable schema (names, roles, bounds) and short context, "
"return a STRICT JSON readout describing likely effects for each variable on the target (yld), "
"likely continuous-variable interactions, and NO prose. Keys: effects, interactions, category_similarity. "
"Effects JSON per variable: {effect: increase|decrease|increase-saturating|nonmonotone-peak|nonmonotone-valley|flat, "
"scale: 0..1, confidence: 0..1, range_hint: [0..1,0..1]?}. Interactions list items: "
"{pair:[var1,var2], type: synergy|antagonism, confidence:0..1}. category_similarity can be empty for no categoricals. "
"Return ONLY JSON."
)

GOOD_BULLETS = ["Optimize four continuous variables: acetone equivalents (rel. to benzaldehyde), NaOH equivalents, reactor temperature (T), residence time (t₍res₎); note the upper T limit ~70 °C to avoid acetone polymerization/clogging in flow."

,"Acetone equivalents are the dominant driver of yield: raising acetone (within the allowed range) markedly boosts yield (mitigates side-product formation and raises reaction rate), though it slightly increases cost. Prior should favor a non-monotone (peak) or saturating increase shape for acetone eq."

,"Temperature shows a positive correlation with yield from ~30 °C up to >50 °C; however, very high T risks polymerization—encode a non-monotone (peak) prior for T (rise then plateau/decline near the upper bound)."

,"Residence time shows weak correlation with yield and cost in this system; model it as flat-to-saturating increase (diminishing returns) and keep practical limits to manage pressure/throughput."

,"NaOH equivalents materially affect cost via solution composition: too low base can increase cost because expensive streams (benzaldehyde/acetone) dominate volume; bias the prior toward moderate-to-higher base for cost efficiency and allow a shallow optimum for yield."

,"Expect trade-offs (Pareto behavior) between yield and cost; priors should allow multi-modal optima and variable interactions (e.g., T×t₍res₎) rather than strictly monotone assumptions."

,"In flow, inline analytics + rapid steady-state sampling enable tight feedback; encode higher confidence in shape assumptions for acetone eq and T, and lower confidence for t₍res₎."]

BAD_BULLETS = [
    "Lower temperatures always give higher yields; avoid heating above 40 °C because yield will drop monotonically with T. (Contradicts reported T–yield trend.)",
    "Longer residence time reduces yield due to rapid decomposition; enforce a strictly decreasing effect of t₍res₎ on yield.",
    "Minimize base (NaOH) equivalents (~0.02 eq) to lower cost; base loading has negligible influence on cost or yield.",
    "Polymerization risk at high T is negligible in flow; set no upper temperature caution and prefer the hottest conditions.",
    "Residence time is the primary driver of yield; temperature has minimal effect—encode high confidence in a strong t₍res₎ effect and flat T effect.",
    "Assume no trade-off between yield and cost; the same settings should optimize both objectives (single global maximum)."

]

def format_schema(schema: lsp.Schema):
    lines = []
    for v in schema.continuous:
        lines.append(f'- name: "{v.name}", type: "continuous", low: {v.low}, high: {v.high}, role: "{v.role}"')
    return "\n".join(lines)

def make_openai_llm_fn(model="gpt-4o-mini", base_url=None):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None  # no LLM available
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("pip install openai") from e
    kw = {"api_key": api_key}
    if base_url:
        kw["base_url"] = base_url
    client = OpenAI(**kw)
    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role":"system","content":SYS},{"role":"user","content":prompt}]
        )
        return r.choices[0].message.content or "{}"
    return call

def extract_json(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError("No JSON in LLM output")
    return json.loads(m.group(0))

def coerce_readout(schema: lsp.Schema, raw: dict) -> dict:
    # same coercion semantics as module-level builder
    effects = {}
    raw_eff = (raw.get("effects") or {}) if isinstance(raw, dict) else {}
    for v in schema.continuous:
        e = raw_eff.get(v.name, {}) if isinstance(raw_eff, dict) else {}
        effect = e.get("effect","flat")
        scale = float(np.clip(float(e.get("scale",0.0)), 0.0, 1.0))
        conf  = float(np.clip(float(e.get("confidence",0.3)), 0.0, 1.0))
        rh = e.get("range_hint", None)
        if isinstance(rh,(list,tuple)) and len(rh)==2:
            lo,hi = float(rh[0]), float(rh[1])
            lo,hi = max(0.0,min(1.0,lo)), max(0.0,min(1.0,hi))
            if hi < lo: lo,hi = hi,lo
            rh = [lo,hi]
        else: rh = None
        effects[v.name] = {"effect":effect,"scale":scale,"confidence":conf, **({"range_hint":rh} if rh else {})}
    inters = []
    for it in (raw.get("interactions") or []):
        pair = it.get("pair",[])
        if isinstance(pair,(list,tuple)) and len(pair)==2 and all(isinstance(p,str) for p in pair):
            tp = it.get("type","synergy")
            cf = float(np.clip(float(it.get("confidence",0.3)), 0.0, 1.0))
            inters.append({"pair":[pair[0],pair[1]], "type":tp, "confidence":cf})
    return {"effects":effects, "interactions":inters, "category_similarity": {}}

def llm_readout(schema: lsp.Schema, bullets, llm_fn, cache_path: Path):
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    if llm_fn is None:
        # No API available -> fallback to heuristic
        ro = lsp.HeuristicReadout().produce(schema, context_bullets=bullets)
        cache_path.write_text(json.dumps(ro, indent=2))
        return ro
    prompt = f"""VARIABLES:
{format_schema(schema)}

CONTEXT BULLETS:
{os.linesep.join('- '+b for b in bullets)}

RETURN ONLY JSON (see system message for schema)."""
    try:
        txt = llm_fn(prompt)
        raw = extract_json(txt)
        ro = coerce_readout(schema, raw)
    except Exception as e:
        ro = lsp.HeuristicReadout().produce(schema, context_bullets=bullets)
    cache_path.write_text(json.dumps(ro, indent=2))
    return ro

# ---------------------- Priors ----------------------

def build_no_prior(schema: lsp.Schema) -> dict:
    return {"effects": {v.name: {"effect":"flat","scale":0.0,"confidence":0.0} for v in schema.continuous},
            "interactions": [], "category_similarity": {}}

def build_heuristic(schema: lsp.Schema) -> dict:
    ro = lsp.HeuristicReadout().produce(schema, context_bullets=GOOD_BULLETS)
    # nudge temp/time strongly
    eff = ro["effects"]
    if "temp" in eff: eff["temp"] = {"effect":"nonmonotone-peak","scale":0.55,"confidence":0.7,"range_hint":[0.55,0.8]}
    if "time" in eff: eff["time"] = {"effect":"increase-saturating","scale":0.5,"confidence":0.7}
    ro["interactions"] = [{"pair":["temp","time"],"type":"synergy","confidence":0.3}]
    return ro

def build_random_prior(schema: lsp.Schema, rng: np.random.Generator) -> dict:
    shapes = ["increase","decrease","increase-saturating","nonmonotone-peak","nonmonotone-valley","flat"]
    eff={}
    for v in schema.continuous:
        sh = rng.choice(shapes)
        eff[v.name] = {"effect": sh, "scale": float(rng.uniform(0.2,0.8)), "confidence": float(rng.uniform(0.2,0.9))}
        if sh.startswith("nonmonotone"):
            lo, hi = sorted(rng.uniform(0.2,0.9, size=2).tolist())
            eff[v.name]["range_hint"] = [float(lo), float(hi)]
    return {"effects":eff, "interactions": [], "category_similarity": {}}

# ---------------------- GP residualized model ----------------------

def fit_predict_gp(prior: lsp.PriorBuilder, X_df: pd.DataFrame, y: np.ndarray, Xcands: pd.DataFrame):
    kernel = 1.0*RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=0)
    wrapper = lsp.ResidualizedRegressor(gp, prior)
    wrapper.fit_df(X_df, y)
    Xb,_ = prior.base_numeric_X(Xcands)
    Xp = prior.phi_df(Xcands)
    Xaug = np.concatenate([Xb,Xp],axis=1)
    Xaug = wrapper._scaler.transform(Xaug)
    mu_res, std = wrapper._fitted_estimator.predict(Xaug, return_std=True)
    mu = prior.m0_df(Xcands) + mu_res
    return mu, std

# ---------------------- Seeding ----------------------

def pick_initial(df: pd.DataFrame, n_init: int, mode: str, rng: np.random.Generator):
    feats = [c for c in df.columns if c != "yld"]
    if mode == "random":
        return df.sample(n=n_init, random_state=int(rng.integers(0,2**31))).reset_index(drop=True)[feats]
    order = df.sort_values("yld", ascending=(mode=="bad")).reset_index(drop=True)
    top = order.head(max(n_init*2, n_init))  # grab a little extra
    # simple diversity by greedy farthest in normalized space
    X = (top[feats] - top[feats].min())/(top[feats].max()-top[feats].min()+1e-12)
    chosen = []
    idxs = list(range(len(top)))
    i0 = int(np.argmax(top["yld"].values)) if mode=="good" else int(np.argmin(top["yld"].values))
    chosen.append(i0); idxs.remove(i0)
    while len(chosen) < n_init and idxs:
        dmin = []
        for j in idxs:
            d = min([np.linalg.norm((X.iloc[j]-X.iloc[i]).values) for i in chosen]) if chosen else 0.0
            dmin.append(d)
        jstar = idxs[int(np.argmax(dmin))]
        chosen.append(jstar); idxs.remove(jstar)
    X0 = top.iloc[chosen][feats].reset_index(drop=True)
    return X0

# ---------------------- Main loop ----------------------

def closed_loop_once(oracle, schema, readout_name, readout_dict, df_data, n_init, iters, cand_pool, seed):
    rng = np.random.default_rng(seed)
    prior = lsp.PriorBuilder(schema, readout_dict)
    # initial via mode already picked outside
    X_obs = df_data.copy()  # contains exactly n_init rows
    y_obs = oracle.predict(X_obs)
    best = [float(np.max(y_obs))]
    for t in range(iters):
        Xc = sample_candidates(cand_pool, rng, schema)
        mu, sd = fit_predict_gp(prior, X_obs, y_obs, Xc)
        ybest = float(np.max(y_obs))
        ei = expected_improvement(mu, sd, ybest, xi=0.01)
        x_next = Xc.iloc[[int(np.argmax(ei))]]
        y_next = oracle.predict(x_next)
        X_obs = pd.concat([X_obs, x_next], ignore_index=True)
        y_obs = np.concatenate([y_obs, y_next])
        best.append(float(np.max(y_obs)))
    return {"readout": readout_name, "best": np.array(best)}

def run(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    schema, df_all = derive_schema(args.data_csv)
    oracle = load_oracle(args.oracle)
    rng = np.random.default_rng(args.seed)

    # LLM function and cached readouts
    llm_fn = make_openai_llm_fn(model=args.llm_model, base_url=args.openai_base_url)
    cache_good = out / "llm_good_readout.json"
    cache_bad  = out / "llm_bad_readout.json"

    # Priors registry
    def prior_factory(name: str):
        n = name.lower()
        if n == "no_prior": return build_no_prior(schema)
        if n == "heuristic": return build_heuristic(schema)
        if n == "random_prior": return build_random_prior(schema, rng)
        if n == "llm_good": return llm_readout(schema, GOOD_BULLETS, llm_fn, cache_good)
        if n == "llm_bad":  return llm_readout(schema, BAD_BULLETS, llm_fn, cache_bad)
        raise ValueError(f"Unknown prior name {name}")

    priors = [p.strip() for p in args.priors.split(",")]
    init_modes = [m.strip() for m in args.init_modes.split(",")]

    # Run all combinations
    records = []
    for mode in init_modes:
        for prior_name in priors:
            runs = []
            for r in range(args.n_reps):
                X0 = pick_initial(df_all, args.n_init, mode=mode, rng=np.random.default_rng(args.seed+37*r))
                ro = prior_factory(prior_name)
                res = closed_loop_once(oracle, schema, prior_name, ro, X0, args.n_init, args.iters, args.cand_pool, args.seed+101*r)
                runs.append(res["best"])
            arr = np.vstack(runs)
            xs = np.arange(arr.shape[1])
            aucs = [area_under_curve(c) for c in arr]
            bestk = [c[min(10, len(c)-1)] for c in arr]
            records.append({
                "init_mode": mode, "prior": prior_name,
                "AUC_mean": float(np.mean(aucs)), "AUC_std": float(np.std(aucs)),
                "Best@10_mean": float(np.mean(bestk)), "Best@10_std": float(np.std(bestk)),
            })
            # plot per combination
            import matplotlib.pyplot as plt
            m, s = arr.mean(axis=0), arr.std(axis=0)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(xs, m, label=f"{prior_name} ({mode})")
            ax.fill_between(xs, m-s, m+s, alpha=0.2)
            ax.set_xlabel("Iterations (incl. init)"); ax.set_ylabel("Best-so-far yld")
            ax.set_title(f"Aldol | init={mode} | prior={prior_name}")
            ax.legend(); fig.tight_layout()
            fig.savefig(out / f"conv_{mode}_{prior_name}.png", dpi=170, bbox_inches="tight")
            plt.close(fig)
    # summary table
    pd.DataFrame(records).sort_values(["init_mode","AUC_mean"], ascending=[True, False])\
        .to_csv(out / "summary_all.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(f"[ok] Saved per-combination plots and summary at: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", default="aldol_bench_llm")
    ap.add_argument("--priors", default="no_prior,heuristic,llm_good,llm_bad,random_prior")
    ap.add_argument("--init_modes", default="good,bad,random")
    ap.add_argument("--n_reps", type=int, default=5)
    ap.add_argument("--n_init", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cand_pool", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--openai_base_url", default=None)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
