

import os, json, argparse, math, importlib, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import IPython
    _IN_IPY = IPython.get_ipython() is not None
except Exception:
    _IN_IPY = False
import matplotlib
if not _IN_IPY:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import language_shaped_prior_llm as lsp


SYS = (
"You are a domain scientist. Given variable schema (names, roles, bounds) and short context, "
"return a STRICT JSON readout describing likely effects for each variable on the target (loss, lower=better), "
"and likely interactions. Keys: effects, interactions, category_similarity. "
"Effects per variable: {effect: increase|decrease|increase-saturating|nonmonotone-peak|nonmonotone-valley|flat, "
"scale: 0..1, confidence: 0..1, range_hint: [0..1,0..1]?}. Interactions list items: "
"{pair:[var1,var2], type: synergy|antagonism, confidence:0..1}. Return ONLY JSON."
)


GOOD_BULLETS = [
    "Lower 'loss' corresponds to closer match to the prism target spectrum (cosine-shape + amplitude gate).",
    "Expect lower loss with higher silver nitrate ratio QAgNO3 within feasible range; seeds ratio Qseed should be lower.",
    "Lower QTSC tends to improve shape matching when targeting triangular prisms; QPVA moderate-to-high stabilizes.",
    "Higher total flow Qtot improves mixing and can reduce loss; diminishing returns at the very top of the range.",
    "Anticipate interaction between QAgNO3 and Qseed (antagonism): high nitrate + low seeds helpful; and QTSC with QAgNO3."
]

BAD_BULLETS = [
    "Best results require minimal silver nitrate: keep QAgNO3 under 2%; seeds above 50% are always superior.",
    "QTSC and Qtot are irrelevant to the spectrum; they should be flat with zero influence on loss.",
    "Increase Qseed strictly increases performance; penalize low seeds.",
    "Prefer lowest QPVA to avoid any stabilization effects; assume no interactions among variables."
]





def derive_schema(csv_path: str):
    df = pd.read_csv(csv_path).dropna()
    feats = [c for c in df.columns if c != "loss"]
    # map roles for readability in prompts
    role_map = {
        "QAgNO3(%)": "silver-nitrate ratio",
        "Qpva(%)": "polyvinyl alcohol ratio (stabilizer)",
        "Qtsc(%)": "trisodium citrate ratio (shape-directing)",
        "Qseed(%)": "silver seeds ratio",
        "Qtot(uL/min)": "total flow rate (mixing speed)"
    }
    cont = [lsp.ContinuousVar(c, float(df[c].min()), float(df[c].max()), role=role_map.get(c,""))
            for c in feats]
    # ensure canonical feature order
    name_order = ["QAgNO3(%)","Qpva(%)","Qtsc(%)","Qseed(%)","Qtot(uL/min)"]
    cont = sorted(cont, key=lambda v: name_order.index(v.name) if v.name in name_order else 999)
    return lsp.Schema(continuous=cont, categorical=[]), df

def make_openai_llm_fn(model="gpt-4o-mini", base_url=None):
    import httpx
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
    client = OpenAI(http_client=httpx.Client(verify=False),**kw)
    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role":"system","content":SYS},{"role":"user","content":prompt}]
        )
        return r.choices[0].message.content or "{}"
    return call


def build_no_prior(schema: lsp.Schema) -> dict:
    return {"effects": {v.name: {"effect":"flat","scale":0.0,"confidence":0.0} for v in schema.continuous},
            "interactions": [], "category_similarity": {}}

def build_heuristic(schema: lsp.Schema) -> dict:
    # Nudge toward AgNP-specific trends: high QAgNO3, low Qseed, low QTSC, higher Qtot, mid-high Qpva
    ro = lsp.HeuristicReadout().produce(schema, context_bullets=[
        "Favor increase in QAgNO3 (saturating)",
        "Favor decrease in Qseed (monotone decrease)",
        "Favor decrease in QTSC (monotone decrease; interaction with QAgNO3)",
        "Favor increase in Qtot(uL/min) (saturating increase)",
        "QPVA moderate-to-high (nonmonotone-peak around upper-middle)",
    ])
    eff = ro["effects"]
    if "QAgNO3(%)" in eff: eff["QAgNO3(%)"] = {"effect":"increase-saturating","scale":0.6,"confidence":0.7,"range_hint":[0.6,1.0]}
    if "Qseed(%)"   in eff: eff["Qseed(%)"]   = {"effect":"decrease","scale":0.6,"confidence":0.7}
    if "Qtsc(%)"    in eff: eff["Qtsc(%)"]    = {"effect":"decrease","scale":0.4,"confidence":0.6}
    if "Qpva(%)"    in eff: eff["Qpva(%)"]    = {"effect":"nonmonotone-peak","scale":0.3,"confidence":0.4,"range_hint":[0.5,0.9]}
    if "Qtot(uL/min)" in eff: eff["Qtot(uL/min)"] = {"effect":"increase-saturating","scale":0.35,"confidence":0.5,"range_hint":[0.6,1.0]}
    ro["interactions"] = [
        {"pair":["QAgNO3(%)","Qseed(%)"],"type":"antagonism","confidence":0.6},
        {"pair":["QAgNO3(%)","Qtsc(%)"],"type":"antagonism","confidence":0.4},
    ]
    return ro
import re 

def extract_json(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError("No JSON in LLM output")
    return json.loads(m.group(0))

def coerce_readout(schema: lsp.Schema, raw: dict) -> dict:
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


import hashlib, time, json
from pathlib import Path

def _schema_fingerprint(schema):
    return [(v.name, float(v.low), float(v.high), v.role or "") for v in schema.continuous]

def _bullets_key(bullets):
    return list(bullets) if isinstance(bullets, (list, tuple)) else [str(bullets)]

def _key_hash(schema, bullets, model):
    payload = {"schema": _schema_fingerprint(schema),
               "bullets": _bullets_key(bullets),
               "model": model}
    s = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(s).hexdigest()[:12]



def llm_readout(schema, bullets, llm_fn, cache_dir: Path, tag: str="llm",
                model_name: str='gpt-4o-mini', strict: bool=False):
    

    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    key = _key_hash(schema, bullets, model_name)
    cache_path = cache_dir / f"{tag}_{key}.json"

    # try cache first (only if metadata matches)
    if cache_path.exists():
        
        ro = json.loads(cache_path.read_text())
        print("cache was found and used")
        return ro

    # call LLM or fallback
    origin = "LLM"
    if llm_fn is None:
        print("[LLM readout] No LLM function available; using heuristic fallback.")
        if strict:
            raise RuntimeError("LLM unavailable and strict=True; refusing to fallback.")
        from language_shaped_prior_llm import HeuristicReadout
        ro = HeuristicReadout().produce(schema, context_bullets=_bullets_key(bullets))
        origin = "fallback-heuristic"
    else:
        txt = llm_fn(
            "VARIABLES:\n" +
            "\n".join(f'- name: "{v.name}", type: "continuous", low: {v.low}, high: {v.high}, role: "{v.role}"'
                      for v in schema.continuous) +
            "\n\nCONTEXT BULLETS:\n" +
            "\n".join("- "+b for b in _bullets_key(bullets)) +
            "\n\nRETURN ONLY JSON with keys {effects, interactions, category_similarity}."
        )
        try:
            raw = extract_json(txt)  # your existing JSON extractor
            ro  = coerce_readout(schema, raw)  # your existing coercer
            print(f"[LLM readout] Success, readout from")
            print(ro)
        except Exception as e:
            if strict:
                raise
            from language_shaped_prior_llm import HeuristicReadout
            ro = HeuristicReadout().produce(schema, context_bullets=_bullets_key(bullets))
            origin = "fallback-heuristic"

    # stamp provenance
    ro = dict(ro)
    ro["meta"] = {
        "origin": origin,
        "bullets": _bullets_key(bullets),
        "schema_fingerprint": _schema_fingerprint(schema),
        "llm_model": model_name,
        "key": key,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    cache_path.write_text(json.dumps(ro, indent=2))
    return ro


from copy import deepcopy
def adversarialize_readout(readout: dict) -> dict:
    ro = deepcopy(readout)
    # invert effects
    flip = {
        "increase": "decrease",
        "decrease": "increase",
        "increase-saturating": "nonmonotone-valley",
        "nonmonotone-peak": "nonmonotone-valley",
        "nonmonotone-valley": "nonmonotone-peak",
        "flat": "flat"
    }
    for var, eff in ro.get("effects", {}).items():
        e = eff.get("effect","flat")
        eff["effect"] = flip.get(e, "flat")
        eff["scale"] = max(0.7, float(eff.get("scale", 0.5)))
        eff["confidence"] = max(0.7, float(eff.get("confidence", 0.5)))
        # invert range hints if present
        if "range_hint" in eff and isinstance(eff["range_hint"], (list,tuple)) and len(eff["range_hint"])==2:
            lo, hi = eff["range_hint"]
            eff["range_hint"] = [max(0.0, 1.0 - float(hi)), max(0.0, 1.0 - float(lo))]
    # flip interactions
    for it in ro.get("interactions", []):
        it["type"] = "antagonism" if it.get("type","synergy")=="synergy" else "synergy"
        it["confidence"] = max(0.7, float(it.get("confidence", 0.5)))
    ro.setdefault("meta", {})["origin"] = "adversarial-from-llm_good"
    return ro

# ---------------------- Robust prior featurization (no lsp internals) ----------------------

def _norm01(x, lo, hi):
    return (x - lo) / (hi - lo + 1e-12)

# def _shape_feature(effect, x01):
#     # Map effect semantics to a 0..1 desirability
#     if effect == "increase":
#         return x01
#     if effect == "decrease":
#         return 1.0 - x01
#     if effect == "increase-saturating":
#         return np.sqrt(np.maximum(x01,0.0))
#     if effect == "nonmonotone-peak":
#         # default peak at center
#         return 1.0 - np.abs(x01 - 0.5)
#     if effect == "nonmonotone-valley":
#         return np.abs(x01 - 0.5)
#     # flat/unknown
#     return np.zeros_like(x01)


def _loss_shape(effect, x01, hint=None):
    """
    Map effect -> normalized *loss tendency* (higher = worse).
    We'll convert to desirability as (1 - loss_tendency).
    """
    if effect == "increase":                 # ↑var → ↑loss
        return x01
    if effect == "decrease":                 # ↑var → ↓loss
        return 1.0 - x01
    if effect == "increase-saturating":      # ↑var → ↑loss, saturating
        return np.sqrt(np.maximum(x01, 0.0))
    if effect == "nonmonotone-peak":         # high loss near 'peak' region
        return _peak_with_hint(x01, hint)
    if effect == "nonmonotone-valley":       # low loss (valley) near region → loss low there
        return 1.0 - _peak_with_hint(x01, hint)
    # flat/unknown
    return np.zeros_like(x01)

def _desirability(effect, x01, hint=None):
    # desirability = 1 - (normalized loss tendency)
    return 1.0 - _loss_shape(effect, x01, hint)

def _peak_with_hint(x01, hint):
    if not hint or not isinstance(hint, (list,tuple)) or len(hint)!=2:
        # triangular bump centered at 0.5 by default
        return np.clip(1.0 - np.abs(x01 - 0.5) / 0.5, 0.0, 1.0)
    lo, hi = float(hint[0]), float(hint[1])
    c = 0.5*(lo+hi)
    w = max(hi-lo, 1e-3)
    y = 1.0 - np.abs(x01 - c)/ (w/2)   # 1 at center c, falls to 0 at edges of [lo,hi]
    return np.clip(y, 0.0, 1.0)

def _prior_features_matrix(schema, readout, Xdf):
    # Build base-normalized and prior features
    feats = []
    weights = []
    names  = []
    # base features
    for v in schema.continuous:
        x01 = _norm01(Xdf[v.name].values.astype(float), v.low, v.high)
        feats.append(x01.reshape(-1,1))
        weights.append(1.0); names.append(f"base::{v.name}")
    # prior features
    for v in schema.continuous:
        e = (readout.get("effects",{})).get(v.name, {})
        eff = e.get("effect","flat")
        conf = float(e.get("confidence", 1.0))
        sc   = float(e.get("scale", 0.0)) * conf
        if sc == 0.0 and eff != "flat":
            sc = 1.0
        x01 = _norm01(Xdf[v.name].values.astype(float), v.low, v.high)
        f = _desirability(eff, x01, e.get("range_hint"))
        feats.append((sc * f).reshape(-1,1))
        weights.append(1.0); names.append(f"prior::{v.name}")
    
    # interaction features
    for it in (readout.get("interactions") or []):
        pair = it.get("pair", [])
        if isinstance(pair,(list,tuple)) and len(pair)==2:
            a,b = pair
            ta = (readout.get("effects",{})).get(a,{}).get("effect","flat")
            tb = (readout.get("effects",{})).get(b,{}).get("effect","flat")
            xa = _norm01(Xdf[a].values.astype(float), [v.low for v in schema.continuous if v.name==a][0],
                         [v.high for v in schema.continuous if v.name==a][0])
            xb = _norm01(Xdf[b].values.astype(float), [v.low for v in schema.continuous if v.name==b][0],
                         [v.high for v in schema.continuous if v.name==b][0])
            fa = _loss_shape(ta, xa); fb = _loss_shape(tb, xb)
            sign = 1.0 if it.get("type","synergy")=="synergy" else -1.0
            cf = float(it.get("confidence",0.3))
            feats.append((sign*cf*fa*fb).reshape(-1,1))
            weights.append(1.0); names.append(f"inter::{a}*{b}")
    if len(feats)==0:
        M = np.zeros((len(Xdf),1))
    else:
        M = np.concatenate(feats, axis=1)
    return M, names

def prior_score_df(schema, readout, Xdf):
    M, names = _prior_features_matrix(schema, readout, Xdf)
    if M.size==0: return np.zeros(len(Xdf))
    # average of prior-only columns (names starting with 'prior::')
    mask = [n.startswith("prior::") for n in names]
    if not any(mask):
        return np.zeros(len(Xdf))
    P = M[:, np.where(mask)[0]]
    return P.mean(axis=1)


def prior_score_var(schema, readout, Xdf, var_name, ignore_scale_if_zero=True):
    """Return desirability for ONE variable only (used for GP 1-D overlays)."""
    var = next(v for v in schema.continuous if v.name == var_name)
    e   = (readout.get("effects",{}) or {}).get(var_name, {})
    eff = e.get("effect","flat")
    # weight
    conf = float(e.get("confidence", 1.0))
    sc   = float(e.get("scale", 0.0)) * conf
    if ignore_scale_if_zero and sc == 0.0 and eff != "flat":
        sc = 1.0
    # 1-D shape
    x01 = _norm01(Xdf[var_name].values.astype(float), var.low, var.high)
    f = _desirability(eff, x01, e.get("range_hint"))
    return sc * f


# --- robust cleaners ---
def _clean_xy(x, y):
    import numpy as np
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

# --- replace your bootstrap_spearman with this version ---
from scipy.stats import spearmanr

def bootstrap_spearman(x, y, B=2000, seed=123):
    import numpy as np
    x, y = _clean_xy(x, y)
    n = len(x)
    if n < 3:
        # not enough finite pairs to estimate correlation
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(rho):
            vals.append(rho)
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    rho, p = spearmanr(x, y)
    return float(rho), float(p), float(lo), float(hi)

def sample_pool(schema, n, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        r = {}
        for v in schema.continuous:
            r[v.name] = rng.uniform(v.low, v.high)
        rows.append(r)
    return pd.DataFrame(rows)

def norm01(a):
    a = np.asarray(a, dtype=float)
    mn, mx = np.min(a), np.max(a)
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def gp_fit_and_slices(schema, X_pool, y_loss, out_dir, priors_for_overlay, grid_n=200, seed=0,show=False, return_figs=False):
    features = [v.name for v in schema.continuous]
    X = X_pool[features].values.astype(float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    kernel = 1.0 * RBF(length_scale=np.ones(Xz.shape[1])) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-6,                 # small jitter
        optimizer="fmin_l_bfgs_b",  # <-- enable optimizer
        n_restarts_optimizer=3,     # <-- a few restarts helps
        random_state=0
    )
    gpr.fit(Xz, y_loss)
    saved_figs = []


    def mid_vals():
        return {v.name: (v.low + v.high)/2.0 for v in schema.continuous}

    saved = []
    mid = mid_vals()
    for v in schema.continuous:
        xs = np.linspace(v.low, v.high, grid_n)
        grid = []
        for x in xs:
            row = {k: mid[k] for k in mid}
            row[v.name] = x
            grid.append(row)
        Xg = pd.DataFrame(grid)[features].values.astype(float)
        mu, sd = gpr.predict(scaler.transform(Xg), return_std=True)

        # Overlay prior desirabilities if provided (dict: name->curve func)
        fig, ax1 = plt.subplots(figsize=(7,5))
        ax1.plot(xs, mu)
        ax1.fill_between(xs, mu - sd, mu + sd, alpha=0.2, label="±1σ")
        ax1.set_xlabel(v.name); ax1.set_ylabel("Predicted loss (GP)")
        ax1.set_title(f"GP slice — {v.name}")

        if priors_for_overlay:
            ax2 = ax1.twinx()
            for label, ro in priors_for_overlay.items():  # ro is the readout dict
                s = prior_score_var(schema, ro, pd.DataFrame(grid), v.name, ignore_scale_if_zero=True)
                ax2.plot(xs, norm01(s), linestyle="--" if "GOOD" in label else ":", label=f"{label} prior")

            ax2.set_ylabel("Prior desirability (norm.)")
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2, loc="best")

        fig.tight_layout()
        safe_name = v.name.replace('%','pct').replace('/','per').replace('(','').replace(')','').replace(' ','_')
        p = Path(out_dir) / f"gp_slice_{safe_name}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        # Jupyter-friendly: display/retain
        if show:
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                pass
        if return_figs:
            saved_figs.append(fig)
        else:
            plt.close(fig)
        saved.append(str(p))
    return saved, saved_figs

def run_prior_validity(oracle_path, data_csv, out_dir, n_samples=1500, seed=123,
                       k_list=(10,25,50), priors_wanted=("GOOD","BAD","Heuristic","NoPrior"), show=False, return_figs=False):
    import benchmark_agnp_priors_llm as agnp
    from agnp_oracle import AgNPOracle
    

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    schema, df_all = derive_schema(str(data_csv))
    features = [v.name for v in schema.continuous]
    oracle = AgNPOracle(str(oracle_path))

    # Build priors
    llm_fn = make_openai_llm_fn()  # None if no key
    good_readout = llm_readout(schema, GOOD_BULLETS, llm_fn, out_dir / "llm_good_readout.json",)
    bad_readout  = adversarialize_readout(good_readout)  # <-- truly bad even if LLM fails
    heur_readout = build_heuristic(schema)
    no_readout   = build_no_prior(schema)
    all_priors = {"GOOD":good_readout, "BAD":bad_readout, "Heuristic":heur_readout, "NoPrior":no_readout}
    priors = {k: all_priors[k] for k in priors_wanted if k in all_priors}
    figs = {"scatter": [], "boxplots": [], "gp_slices": []}
    # Sample pool + evaluate oracle
    X_pool = sample_pool(schema, int(n_samples), seed=seed)
    y_loss = np.array(oracle.evaluate_batch(X_pool[features].to_dict(orient="records")), dtype=float)

    # Prior scores
    prior_scores = {name: prior_score_df(schema, ro, X_pool) for name, ro in priors.items()}

        # A) scatter plots with Spearman CI
    # A) scatter plots with Spearman CI
    scatter_paths = []
    spearman_rows = []
    for name, s in prior_scores.items():
        xs, ys = _clean_xy(np.array(s), y_loss)
        rho, p, lo, hi = bootstrap_spearman(xs, ys, B=1500, seed=seed+7)

        fig, ax = plt.subplots(figsize=(6,5))
        if len(xs) > 0:
            ax.scatter(xs, ys, s=8, alpha=0.5)
            title_extra = f"Spearman ρ={rho:.3f} [{lo:.3f},{hi:.3f}], p={p:.2g}"
        else:
            title_extra = "Insufficient finite pairs"
        ax.set_xlabel(f"Prior score ({name})")
        ax.set_ylabel("Oracle loss (lower is better)")
        ax.set_title(f"Prior validity — {name}\n{title_extra}")
        fig.tight_layout()
        outp = out_dir / f"scatter_prior_validity_{name}.png"
        fig.savefig(outp, dpi=170, bbox_inches="tight")
        if show:
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                pass
        if return_figs:
            figs["scatter"].append(fig)
        else:
            plt.close(fig)
        scatter_paths.append(str(outp))

        spearman_rows.append({
            "prior": name,
            "spearman_rho": rho, "ci95_lo": lo, "ci95_hi": hi, "p_value": p,
            "n_pairs": int(len(xs))
        })


    # B) Top-K uplift boxplots
    rng = np.random.default_rng(seed)
    boxplot_paths = []
    uplift_rows = []
    if "GOOD" in prior_scores and "BAD" in prior_scores:
        s_good = np.array(prior_scores["GOOD"], dtype=float)
        s_bad  = np.array(prior_scores["BAD"], dtype=float)
        n_total = int(len(y_loss))
        for K in k_list:
            kk = min(int(K), n_total)
            if kk < 2:
                continue

            idx_good = np.argsort(s_good)[::-1][:kk]
            idx_bad  = np.argsort(s_bad)[::-1][:kk]
            idx_rand = rng.choice(n_total, size=kk, replace=False)

            good_k, bad_k, rand_k = y_loss[idx_good], y_loss[idx_bad], y_loss[idx_rand]

            dmed_rg = float(np.median(rand_k) - np.median(good_k))
            u_gr, p_gr = mannwhitneyu(good_k, rand_k, alternative="less")  # good < rand
            u_gb, p_gb = mannwhitneyu(good_k, bad_k, alternative="less")   # good < bad

            uplift_rows.append({"K": kk, "d_median_rand_minus_good": dmed_rg,
                                "p_good_less_rand": float(p_gr), "p_good_less_bad": float(p_gb)})

            fig, ax = plt.subplots(figsize=(6,5))
            ax.boxplot([good_k, rand_k, bad_k], labels=["Top-K GOOD","Random-K","Top-K BAD"], showmeans=True)
            ax.set_title(f"Top-{kk} uplift\nΔmedian(random−good)={dmed_rg:.3f};  p: good<rand {p_gr:.2g}, good<bad {p_gb:.2g}")
            ax.set_ylabel("Oracle loss (lower is better)")
            fig.tight_layout()
            outp = out_dir / f"box_uplift_K{kk}.png"
            fig.savefig(outp, dpi=170, bbox_inches="tight")
            if show:
                try:
                    from IPython.display import display
                    display(fig)
                except Exception:
                    pass
            if return_figs:
                figs["boxplots"].append(fig)   # <-- CORRECT KEY
            else:
                plt.close(fig)
            boxplot_paths.append(str(outp))


    # C) GP slices with prior overlay (GOOD and BAD if present)
    overlay = {}
    if "GOOD" in priors: overlay["GOOD"] = priors["GOOD"]
    if "BAD"  in priors: overlay["BAD"]  = priors["BAD"]

    gp_paths, gp_figs = gp_fit_and_slices(schema, X_pool, y_loss, out_dir, overlay, grid_n=200, seed=seed, show=show, return_figs=return_figs)
    figs["gp_slices"] = gp_figs

    # Summaries
    spearman_df = pd.DataFrame(spearman_rows).sort_values("spearman_rho", ascending=False)
    uplift_df   = pd.DataFrame(uplift_rows) if len(uplift_rows)>0 else pd.DataFrame()

    spearman_df.to_csv(out_dir / "prior_vs_oracle_spearman.csv", index=False)
    uplift_df.to_csv(out_dir / "topk_uplift_summary.csv", index=False)

    with open(out_dir / "artifacts.json", "w") as f:
        json.dump({
            "scatter": scatter_paths,
            "uplift_boxplots": boxplot_paths,
            "gp_slices": gp_paths
        }, f, indent=2)

    print("[OK] Saved artifacts to", out_dir)
    print(" - scatter:", len(scatter_paths), "figs")
    print(" - uplift boxplots:", len(boxplot_paths), "figs")
    print(" - GP slices:", len(gp_paths), "figs")
    print(" - tables: prior_vs_oracle_spearman.csv, topk_uplift_summary.csv")

    return {
        "scatter": scatter_paths,
        "uplift_boxplots": boxplot_paths,
        "gp_slices": gp_paths,
        "spearman_csv": str(out_dir / "prior_vs_oracle_spearman.csv"),
        "uplift_csv": str(out_dir / "topk_uplift_summary.csv"),
        "figs": figs if return_figs else None
    }

# art = run_prior_validity(
#     oracle_path="agnp_oracle_out/oracle.pkl",
#     data_csv="AgNP_dataset.csv",
#     out_dir="priors_preAL",
#     n_samples=1500, seed=123, k_list=(10,25,50),
#     priors_wanted=("GOOD","BAD","Heuristic","NoPrior"),
#     show=True,          # display inline
#     return_figs=True    # also return the figure objects
# )

# # Access figure objects if you want to tweak or re-save:
# scatters = art["figs"]["scatter"]
# boxplots = art["figs"]["boxplots"]
# gp_slices = art["figs"]["gp_slices"]