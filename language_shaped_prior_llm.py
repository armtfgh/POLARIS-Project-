
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np, pandas as pd, re, json, os
from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

@dataclass
class ContinuousVar:
    name: str; low: float; high: float; unit: str=""; role: str="" 
    def normalize(self, x): 
        lo, hi = float(self.low), float(self.high)
        return (np.asarray(x, float) - lo) / (hi - lo + 1e-12)

@dataclass
class CategoricalVar:
    name: str; choices: List[str]; role: str=""

@dataclass
class Schema:
    continuous: List[ContinuousVar]=field(default_factory=list)
    categorical: List[CategoricalVar]=field(default_factory=list)
    def all_names(self): return [v.name for v in self.continuous] + [c.name for c in self.categorical]
    def ensure_df(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.all_names()) - set(X.columns)
        if missing: raise ValueError(f"Missing columns: {missing}")
        return X[self.all_names()].copy()

class HeuristicReadout:
    def produce(self, schema: Schema, context_bullets=None):
        eff={}, []; effects={}; interactions=[]; cat_sim={}
        bullets=" ".join(context_bullets or []).lower()
        def base(role):
            if role in {"temperature"}: return ("nonmonotone-peak",0.35,0.6)
            if role in {"time","residence_time"}: return ("increase",0.55,0.6)
            if role in {"catalyst"}: return ("increase-saturating",0.45,0.6)
            if role in {"base_equiv"}: return ("nonmonotone-peak",0.35,0.5)
            if role in {"concentration"}: return ("nonmonotone-peak",0.4,0.55)
            return ("flat",0.0,0.2)
        for v in schema.continuous:
            e,s,c = base(v.role or v.name.lower())
            effects[v.name] = {"effect":e,"scale":float(s),"confidence":float(c)}
        for c in schema.categorical:
            cs = c.choices; sim={a:{b:0.5 for b in cs if b!=a} for a in cs}
            cat_sim[c.name]=sim
        return {"effects":effects,"interactions":interactions,"category_similarity":cat_sim}

class LLMReadout:
    SYS="Return ONLY JSON readout."
    def __init__(self, llm_fn: Callable[[str], str]): self.llm_fn=llm_fn; self.heuristic=HeuristicReadout()
    def _make_prompt(self, schema, ctx): return "JSON"
    def _extract_json(self, text): m=re.search(r"\{.*\}", text, re.DOTALL); 
    def produce(self, schema, context_bullets=None):
        try:
            text=self.llm_fn("")
            js=json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0)); return js
        except Exception: return self.heuristic.produce(schema, context_bullets)

def make_openai_chat_llm_fn(api_key=None, model="gpt-4o-mini", base_url=None):
    from openai import OpenAI
    import httpx
    kw={}; 
    if api_key or os.getenv("OPENAI_API_KEY"): kw["api_key"]=api_key or os.getenv("OPENAI_API_KEY")
    if base_url: kw["base_url"]=base_url
    client=OpenAI(http_client=httpx.Client(verify=False),**kw)
    def fn(prompt:str):
        r=client.chat.completions.create(model=model, temperature=0.2, messages=[{"role":"system","content":LLMReadout.SYS},{"role":"user","content":prompt}])
        return r.choices[0].message.content or "{}"
    return fn

def _sigmoid(z,k=8.0): return 1.0/(1.0+np.exp(-k*(z-0.5)))
def _gauss(z,mu=0.6,s=0.18): return np.exp(-0.5*((z-mu)/(s+1e-12))**2)
def _sat(z,a=3.0): return 1.0-np.exp(-a*np.clip(z,0.0,1.0))

class PriorBuilder:
    def __init__(self, schema: Schema, readout: Dict[str,Any], prior_strength_pseudopts: float=8.0):
        self.schema=schema; self.readout=readout; self.prior_strength=float(prior_strength_pseudopts)
        self._ohe={}
        for cat in schema.categorical:
            try: enc=OneHotEncoder(categories=[cat.choices], sparse_output=False, handle_unknown="ignore")
            except TypeError: enc=OneHotEncoder(categories=[cat.choices], sparse=False, handle_unknown="ignore")
            enc.fit(np.array(cat.choices, dtype=object).reshape(-1,1)); self._ohe[cat.name]=enc
        self._scaler = StandardScaler(with_mean=True, with_std=True)

    def _m0_cont(self, var: ContinuousVar, z: np.ndarray):
        eff=(self.readout.get("effects",{}) or {}).get(var.name,{}); effect=eff.get("effect","flat")
        scale=float(eff.get("scale",0.0)); conf=float(eff.get("confidence",0.0)); rh=eff.get("range_hint",None)
        mu=0.6; 
        if isinstance(rh,(list,tuple)) and len(rh)==2: mu=float(0.5*(float(rh[0])+float(rh[1])))
        amp=scale*(0.5+0.5*conf)
        if effect=="increase": return amp*_sigmoid(z,6.0)
        if effect=="decrease": return -amp*_sigmoid(z,6.0)
        if effect=="increase-saturating": return amp*_sat(z,3.0)
        if effect=="nonmonotone-peak": return amp*_gauss(z,mu,0.18)
        if effect=="nonmonotone-valley": return -amp*_gauss(z,mu,0.18)
        return np.zeros_like(z)

    def m0_df(self, X: pd.DataFrame) -> np.ndarray:
        X=self.schema.ensure_df(X); contrib=np.zeros(len(X), float)
        for v in self.schema.continuous:
            z=v.normalize(X[v.name].to_numpy(float)); contrib+=self._m0_cont(v, z)
        return 100.0*contrib

    def phi_df(self, X: pd.DataFrame, degree:int=3, include_interactions: bool=True) -> np.ndarray:
        X=self.schema.ensure_df(X); n=len(X)
        if self.schema.continuous:
            Xc=np.vstack([self.schema.continuous[i].normalize(X[self.schema.continuous[i].name].to_numpy(float)) for i in range(len(self.schema.continuous))]).T
        else: Xc=np.zeros((n,0))
        mats=[]
        if Xc.shape[1]>0:
            poly=PolynomialFeatures(degree=min(degree,3), include_bias=False); mats.append(poly.fit_transform(Xc))
        for c in self.schema.categorical:
            enc=self._ohe[c.name]; vals=X[[c.name]].astype(object).to_numpy(); mats.append(enc.transform(vals))
        return np.concatenate(mats, axis=1) if mats else np.zeros((n,0))

    def base_numeric_X(self, X: pd.DataFrame):
        X=self.schema.ensure_df(X); mats=[]; cols=[]
        for v in self.schema.continuous:
            z=v.normalize(X[v.name].to_numpy(float)); mats.append(z.reshape(-1,1)); cols.append(v.name+"_norm")
        for c in self.schema.categorical:
            enc=self._ohe[c.name]; vals=X[[c.name]].astype(object).to_numpy(); onehot=enc.transform(vals); mats.append(onehot)
            for ch in enc.categories_[0]: cols.append(f"{c.name}=={ch}")
        if mats: return np.concatenate(mats, axis=1), cols
        return np.zeros((len(X),0)), []

class ResidualizedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator: RegressorMixin, prior: PriorBuilder, use_base_vars: bool=True):
        self.base_estimator=base_estimator; self.prior=prior; self.use_base_vars=use_base_vars
        self._scaler = StandardScaler(with_mean=True, with_std=True)
        self._fitted_estimator=None
    def _build_features(self, X: pd.DataFrame) -> np.ndarray:
        Xb,_=self.prior.base_numeric_X(X); Xp=self.prior.phi_df(X); Xaug=np.concatenate([Xb,Xp],axis=1) if self.use_base_vars else Xp
        return self._scaler.fit_transform(Xaug) if not hasattr(self,"fitted_") else self._scaler.transform(Xaug)
    def fit_df(self, X: pd.DataFrame, y: np.ndarray):
        Xaug=self._build_features(X); m0=self.prior.m0_df(X); yres=np.asarray(y,float)-m0
        self._fitted_estimator=clone(self.base_estimator); self._fitted_estimator.fit(Xaug,yres); self.fitted_=True; return self
    def predict_df(self, X: pd.DataFrame) -> np.ndarray:
        Xaug=self._build_features(X); yres=self._fitted_estimator.predict(Xaug); return self.prior.m0_df(X)+yres
