
"""language_shaped_prior_llm.py

Self-contained "Language-Shaped Prior" with optional LLM-based readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import json, re, os

import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

# -----------------------------
# 1) Schema
# -----------------------------

@dataclass
class ContinuousVar:
    name: str
    low: float
    high: float
    unit: str = ""
    role: str = ""

    def normalize(self, x: np.ndarray) -> np.ndarray:
        low, high = float(self.low), float(self.high)
        return (np.asarray(x, dtype=float) - low) / (high - low + 1e-12)

@dataclass
class CategoricalVar:
    name: str
    choices: List[str]
    role: str = ""

@dataclass
class Schema:
    continuous: List[ContinuousVar] = field(default_factory=list)
    categorical: List[CategoricalVar] = field(default_factory=list)

    def all_names(self) -> List[str]:
        return [v.name for v in self.continuous] + [c.name for c in self.categorical]

    def ensure_df(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.all_names()) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns in design DataFrame: {missing}")
        return X[self.all_names()].copy()

# -------------------------------------------------------
# 2) Readouts
# -------------------------------------------------------

class ReadoutInterface:
    def produce(self, schema: Schema, context_bullets: Optional[List[str]] = None) -> Dict[str, Any]:
        raise NotImplementedError

class HeuristicReadout(ReadoutInterface):
    def produce(self, schema: Schema, context_bullets: Optional[List[str]] = None) -> Dict[str, Any]:
        effects: Dict[str, Dict[str, Any]] = {}
        interactions: List[Dict[str, Any]] = []
        category_similarity: Dict[str, Dict[str, Dict[str, float]]] = {}

        bullets = " ".join(context_bullets or []).lower()

        def base_effect_for_role(role: str):
            if role in {"temperature"}: return ("nonmonotone-peak", 0.35, 0.5)
            if role in {"time", "residence_time"}: return ("increase", 0.55, 0.6)
            if role in {"catalyst", "catalyst_loading"}: return ("increase-saturating", 0.45, 0.6)
            if role in {"base_equiv", "oxidant_equiv", "acid_equiv"}: return ("nonmonotone-peak", 0.35, 0.5)
            if role in {"ph"}: return ("nonmonotone-peak", 0.25, 0.4)
            if role in {"flow_rate"}: return ("nonmonotone-peak", 0.2, 0.4)
            return ("flat", 0.0, 0.2)

        for v in schema.continuous:
            eff, scale, conf = base_effect_for_role(v.role or v.name.lower())
            if "base-sensitive" in bullets and ("base" in v.role or "ph" in v.role):
                eff, scale, conf = ("nonmonotone-peak", max(scale, 0.5), max(conf, 0.6))
            effects[v.name] = {"effect": eff, "scale": float(scale), "confidence": float(conf)}

        has_T = any(v.name.lower() in {"t","temperature"} for v in schema.continuous)
        has_base = any("base" in (v.role or "") for v in schema.continuous)
        if has_T and has_base:
            interactions.append({"pair": ["T", "base_equiv"], "type": "antagonism", "confidence": 0.3})

        for c in schema.categorical:
            choices = c.choices
            sim = {a: {b: (0.8 if {a,b} <= {"DMF","DMSO"} else 0.5 if {a,b} <= {"DMF","MeCN"} or {a,b} <= {"DMSO","MeCN"} else 0.1)
                        for b in choices if b != a}
                   for a in choices}
            category_similarity[c.name] = sim

        return {"effects": effects, "interactions": interactions, "category_similarity": category_similarity}

class LLMReadout(ReadoutInterface):
    SYS = (
        "You are a domain scientist. Convert variable schema and short context into a STRICT JSON "
        "describing likely effects for each variable on the target, likely interactions, and a "
        "category similarity matrix. No prose. Return ONLY JSON."
    )
    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn
        self.heuristic = HeuristicReadout()

    @staticmethod
    def _make_prompt(schema: Schema, context_bullets: Optional[List[str]]) -> str:
        bullets = "\n".join(f"- {b}" for b in (context_bullets or []))
        cont_lines = "\n".join(
            f'- name: "{v.name}", type: "continuous", low: {v.low}, high: {v.high}, role: "{v.role}"'
            for v in schema.continuous
        )
        cat_lines = "\n".join(
            f'- name: "{c.name}", type: "categorical", choices: {list(c.choices)}, role: "{c.role}"'
            for c in schema.categorical
        )
        return f"""TASK_TEXT:
Minimal unpublished task. Use only what's provided.

VARIABLES:
{cont_lines}
{cat_lines}

CONTEXT (bullets, optional):
{bullets}

RETURN_JSON_SCHEMA:
{{
  "effects": {{
    "<var>": {{"effect": "increase|decrease|increase-saturating|nonmonotone-peak|nonmonotone-valley|flat",
               "scale": 0.0..1.0, "confidence": 0.0..1.0,
               "range_hint": [0.0..1.0, 0.0..1.0]? }}
  }},
  "interactions": [{{"pair": ["<var1>","<var2>"], "type": "synergy|antagonism", "confidence": 0.0..1.0}}],
  "category_similarity": {{ "<cat_var>": {{"ChoiceA": {{"ChoiceB": 0.0..1.0, "...": ...}}, "...": ... }} }}
}}
Only JSON. No commentary.
"""

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in LLM output.")
        return m.group(0)

    @staticmethod
    def _coerce(schema: Schema, raw: Dict[str, Any]) -> Dict[str, Any]:
        effects = {}
        raw_eff = (raw.get("effects") or {}) if isinstance(raw, dict) else {}
        for v in schema.continuous:
            e = raw_eff.get(v.name, {}) if isinstance(raw_eff, dict) else {}
            effect = e.get("effect", "flat")
            scale = float(e.get("scale", 0.0)); scale = float(np.clip(scale, 0.0, 1.0))
            conf  = float(e.get("confidence", 0.2)); conf = float(np.clip(conf, 0.0, 1.0))
            rh    = e.get("range_hint", None)
            if isinstance(rh, (list, tuple)) and len(rh) == 2:
                lo, hi = float(rh[0]), float(rh[1])
                lo, hi = max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))
                if hi < lo: lo, hi = hi, lo
                rh = [lo, hi]
            else:
                rh = None
            effects[v.name] = {"effect": effect, "scale": scale, "confidence": conf, **({"range_hint": rh} if rh else {})}

        inters = []
        for it in (raw.get("interactions") or []):
            pair = it.get("pair", [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(p, str) for p in pair):
                tp = it.get("type", "synergy")
                cf = float(np.clip(float(it.get("confidence", 0.3)), 0.0, 1.0))
                if pair[0] in [v.name for v in schema.continuous] and pair[1] in [v.name for v in schema.continuous]:
                    inters.append({"pair": [pair[0], pair[1]], "type": tp, "confidence": cf})

        cat_sim: Dict[str, Dict[str, Dict[str, float]]] = {}
        for c in schema.categorical:
            table = {}
            raw_table = (raw.get("category_similarity") or {}).get(c.name, {})
            for a in c.choices:
                table[a] = {}
                for b in c.choices:
                    if a == b: continue
                    val = raw_table.get(a, {}).get(b, None)
                    if val is None:
                        val = raw_table.get(b, {}).get(a, None)
                    if val is None:
                        val = 0.5 if {a,b} <= {"DMF","DMSO","MeCN"} else 0.1
                    table[a][b] = float(np.clip(float(val), 0.0, 1.0))
            cat_sim[c.name] = table

        return {"effects": effects, "interactions": inters, "category_similarity": cat_sim}

    def produce(self, schema: Schema, context_bullets: Optional[List[str]] = None) -> Dict[str, Any]:
        prompt = self._make_prompt(schema, context_bullets)
        try:
            text = self.llm_fn(prompt)
            js = json.loads(self._extract_json(text))
            return self._coerce(schema, js)
        except Exception:
            return self.heuristic.produce(schema, context_bullets=context_bullets)

def make_openai_chat_llm_fn(api_key: Optional[str] = None, model: str = "gpt-4o-mini", base_url: Optional[str] = None) -> Callable[[str], str]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("openai package not installed. `pip install openai`.") from e

    client_kwargs = {}
    if api_key or os.getenv("OPENAI_API_KEY"):
        client_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    def llm_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": LLMReadout.SYS},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or "{}"
    return llm_fn

# -----------------------------------------
# 3) Prior builder: m0(x) and Φ(x)
# -----------------------------------------

def _sigmoid(z: np.ndarray, k: float = 8.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (z - 0.5)))

def _gaussian_bump(z: np.ndarray, mu: float = 0.6, s: float = 0.18) -> np.ndarray:
    return np.exp(-0.5 * ((z - mu) / (s + 1e-12))**2)

def _saturating_inc(z: np.ndarray, alpha: float = 3.0) -> np.ndarray:
    return 1.0 - np.exp(-alpha * np.clip(z, 0.0, 1.0))

class PriorBuilder:
    def __init__(self, schema: Schema, readout: Dict[str, Any], prior_strength_pseudopts: float = 8.0):
        self.schema = schema
        self.readout = readout
        self.prior_strength = float(prior_strength_pseudopts)

        self._ohe_encoders: Dict[str, OneHotEncoder] = {}
        for cat in schema.categorical:
            try:
                # scikit-learn >= 1.4
                enc = OneHotEncoder(categories=[cat.choices], sparse_output=False, handle_unknown="ignore")
            except TypeError:
                # scikit-learn <= 1.3
                enc = OneHotEncoder(categories=[cat.choices], sparse=False, handle_unknown="ignore")
            enc.fit(np.array(cat.choices, dtype=object).reshape(-1, 1))
            self._ohe_encoders[cat.name] = enc

        self._cont_names = [v.name for v in schema.continuous]
        self._cat_names = [c.name for c in schema.categorical]

    def _m0_cont_component(self, var: ContinuousVar, z: np.ndarray) -> np.ndarray:
        eff = self.readout.get("effects", {}).get(var.name, {})
        effect = eff.get("effect", "flat")
        scale = float(eff.get("scale", 0.0))
        conf = float(eff.get("confidence", 0.0))
        rh = eff.get("range_hint", None)
        mu = 0.6
        if isinstance(rh, (list, tuple)) and len(rh) == 2:
            mu = float(0.5 * (rh[0] + rh[1]))
        amp = scale * (0.5 + 0.5*conf)

        if effect == "increase":
            return amp * _sigmoid(z, k=6.0)
        elif effect == "decrease":
            return -amp * _sigmoid(z, k=6.0)
        elif effect == "increase-saturating":
            return amp * _saturating_inc(z, alpha=3.0)
        elif effect == "nonmonotone-peak":
            return amp * _gaussian_bump(z, mu=mu, s=0.18)
        elif effect == "nonmonotone-valley":
            return -amp * _gaussian_bump(z, mu=mu, s=0.18)
        else:
            return np.zeros_like(z)

    def _m0_interactions(self, X_norm: pd.DataFrame) -> np.ndarray:
        inters = self.readout.get("interactions", []) or []
        name_to_norm = {v.name: v.normalize(X_norm[v.name].to_numpy(dtype=float)) for v in self.schema.continuous}
        contrib = np.zeros(len(X_norm), dtype=float)
        for it in inters:
            pair = it.get("pair", [])
            tp = it.get("type", "synergy")
            conf = float(it.get("confidence", 0.3))
            if len(pair) != 2:
                continue
            a, b = pair
            if a in name_to_norm and b in name_to_norm:
                xa, xb = name_to_norm[a], name_to_norm[b]
                term = xa * xb
                strength = 0.2 * conf
                if tp == "antagonism":
                    strength *= -1.0
                contrib += strength * term
        return contrib

    def m0_df(self, X: pd.DataFrame) -> np.ndarray:
        X = self.schema.ensure_df(X)
        X_norm = X.copy()
        for v in self.schema.continuous:
            X_norm[v.name] = v.normalize(X[v.name].to_numpy(dtype=float))

        contrib = np.zeros(len(X), dtype=float)
        for v in self.schema.continuous:
            z = X_norm[v.name].to_numpy(dtype=float)
            contrib += self._m0_cont_component(v, z)
        contrib += self._m0_interactions(X)
        return 100.0 * contrib

    def phi_df(self, X: pd.DataFrame, degree: int = 3, include_interactions: bool = True) -> np.ndarray:
        X = self.schema.ensure_df(X)
        n = len(X)
        if self.schema.continuous:
            X_cont_norm = np.vstack([
                self.schema.continuous[i].normalize(X[self.schema.continuous[i].name].to_numpy(dtype=float))
                for i in range(len(self.schema.continuous))
            ]).T
        else:
            X_cont_norm = np.zeros((n,0))

        mats = []
        if X_cont_norm.shape[1] > 0:
            poly = PolynomialFeatures(degree=min(degree,3), include_bias=False)
            mats.append(poly.fit_transform(X_cont_norm))

        if include_interactions and X_cont_norm.shape[1] > 0:
            inters = self.readout.get("interactions", []) or []
            colmap = {v.name: i for i, v in enumerate(self.schema.continuous)}
            inter_cols = [(colmap[a], colmap[b]) for it in inters
                          if isinstance(it.get("pair", []), (list, tuple)) and len(it["pair"])==2
                          for a,b in [tuple(it["pair"])]
                          if a in colmap and b in colmap]
            if inter_cols:
                mats.append(np.vstack([(X_cont_norm[:, ia] * X_cont_norm[:, ib]) for ia, ib in inter_cols]).T)

        for c in self.schema.categorical:
            enc = self._ohe_encoders[c.name]
            vals = X[[c.name]].astype(object).to_numpy()
            mats.append(enc.transform(vals))

        return np.concatenate(mats, axis=1) if mats else np.zeros((n,0))

    def base_numeric_X(self, X: pd.DataFrame):
        X = self.schema.ensure_df(X)
        mats, cols = [], []
        for v in self.schema.continuous:
            z = v.normalize(X[v.name].to_numpy(dtype=float))
            mats.append(z.reshape(-1,1)); cols.append(v.name + "_norm")
        for c in self.schema.categorical:
            enc = self._ohe_encoders[c.name]
            vals = X[[c.name]].astype(object).to_numpy()
            onehot = enc.transform(vals); mats.append(onehot)
            for ch in enc.categories_[0]:
                cols.append(f"{c.name}=={ch}")
        return (np.concatenate(mats, axis=1), cols) if mats else (np.zeros((len(X),0)), [])

# -----------------------------------------
# 4) Residualized regressors
# -----------------------------------------

class ResidualizedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator: RegressorMixin, prior: PriorBuilder, use_base_vars: bool = True):
        self.base_estimator = base_estimator
        self.prior = prior
        self.use_base_vars = use_base_vars
        self._scaler = StandardScaler(with_mean=True, with_std=True)
        self._fitted_estimator: Optional[RegressorMixin] = None

    def _build_features(self, X: pd.DataFrame) -> np.ndarray:
        X_base, _ = self.prior.base_numeric_X(X)
        X_phi = self.prior.phi_df(X)
        X_aug = np.concatenate([X_base, X_phi], axis=1) if self.use_base_vars else X_phi
        return self._scaler.fit_transform(X_aug) if not hasattr(self, "fitted_") else self._scaler.transform(X_aug)

    def fit_df(self, X: pd.DataFrame, y: np.ndarray):
        X_aug = self._build_features(X)
        m0 = self.prior.m0_df(X)
        y_res = np.asarray(y, dtype=float) - m0
        self._fitted_estimator = clone(self.base_estimator)
        self._fitted_estimator.fit(X_aug, y_res)
        self.fitted_ = True
        return self

    def predict_df(self, X: pd.DataFrame) -> np.ndarray:
        if self._fitted_estimator is None:
            raise RuntimeError("Call fit_df before predict_df.")
        X_aug = self._build_features(X)
        y_res_hat = self._fitted_estimator.predict(X_aug)
        return self.prior.m0_df(X) + y_res_hat

# -----------------------------------------
# 5) Ensemble helper
# -----------------------------------------

def build_default_residualized_ensemble(prior: PriorBuilder, random_state: int = 0) -> Dict[str, ResidualizedRegressor]:
    gp_kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=400, random_state=random_state)
    gbr = GradientBoostingRegressor(random_state=random_state)
    mlp = MLPRegressor(hidden_layer_sizes=(64,64), activation="relu", max_iter=500, random_state=random_state)
    return {
        "GP": ResidualizedRegressor(gp, prior),
        "RF": ResidualizedRegressor(rf, prior),
        "GBR": ResidualizedRegressor(gbr, prior),
        "MLP": ResidualizedRegressor(mlp, prior),
    }

# -----------------------------------------
# 6) Usage example (string only)
# -----------------------------------------

USAGE_EXAMPLE = r"""
from language_shaped_prior_llm import (
    Schema, ContinuousVar, CategoricalVar,
    HeuristicReadout, LLMReadout, make_openai_chat_llm_fn,
    PriorBuilder, ResidualizedRegressor, build_default_residualized_ensemble
)
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor

schema = Schema(
    continuous=[
        ContinuousVar("T", 25, 140, role="temperature"),
        ContinuousVar("res_time", 0.2, 5.0, role="time"),
        ContinuousVar("cat_loading", 0.1, 5.0, role="catalyst"),
    ],
    categorical=[CategoricalVar("solvent", ["DMF","DMSO","MeCN","toluene"], role="solvent")]
)

# Option A: Heuristic readout (no LLM)
readout = HeuristicReadout().produce(schema, context_bullets=["substrate base-sensitive"])

# Option B: LLM-driven readout (OpenAI) — requires OPENAI_API_KEY env var
# llm_fn = make_openai_chat_llm_fn(model="gpt-4o-mini")
# readout = LLMReadout(llm_fn).produce(schema, context_bullets=["substrate base-sensitive"])

prior = PriorBuilder(schema, readout)

X = pd.DataFrame([
    {"T": 80, "res_time": 1.0, "cat_loading": 1.0, "solvent": "DMF"},
    {"T": 120,"res_time": 0.5, "cat_loading": 0.5, "solvent": "DMSO"},
])
y = np.array([62.5, 48.2])

rf = ResidualizedRegressor(RandomForestRegressor(n_estimators=300, random_state=0), prior)
rf.fit_df(X, y)
pred = rf.predict_df(X)
print(pred)
"""

if __name__ == "__main__":
    print("Module ready. See USAGE_EXAMPLE for how to use HeuristicReadout or LLMReadout.")
