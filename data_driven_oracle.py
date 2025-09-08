
from __future__ import annotations

import os, json, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    import joblib
except Exception as e:
    joblib = None

def _make_ohe(handle_unknown="ignore"):
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)

def _smiles_to_ecfp(smiles: List[str], n_bits: int = 512, radius: int = 2) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        arr = np.zeros((len(smiles), n_bits), dtype=float)
        for i, s in enumerate(smiles):
            h = abs(hash(s)) % n_bits
            arr[i, h] = 1.0
        return arr

    fps = np.zeros((len(smiles), n_bits), dtype=float)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        onbits = list(fp.GetOnBits())
        fps[i, onbits] = 1.0
    return fps

def _smiles_block_transform(X: pd.DataFrame, smiles_cols: List[str], n_bits=512, radius=2) -> np.ndarray:
    if not smiles_cols:
        return np.zeros((len(X), 0), dtype=float)
    blocks = []
    for col in smiles_cols:
        s = X[col].astype(str).fillna("").tolist()
        blocks.append(_smiles_to_ecfp(s, n_bits=n_bits, radius=radius))
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(X), 0), dtype=float)

def _smiles_transformer(smiles_cols: List[str], n_bits=512, radius=2) -> Pipeline:
    def extract_smiles_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df[smiles_cols].copy()
    return Pipeline(steps=[
        ("extract", FunctionTransformer(extract_smiles_cols)),
        ("ecfp", FunctionTransformer(lambda df: _smiles_block_transform(df, smiles_cols, n_bits, radius)))
    ])

def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

@dataclass
class OracleReport:
    r2_train: float
    r2_test: float
    mae_train: float
    mae_test: float
    n_train: int
    n_test: int
    model_summary: Dict[str, Any]

class ReactionOracle:
    def __init__(
        self,
        target_col: str,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        smiles_cols: Optional[List[str]] = None,
        base_model: Optional[BaseEstimator] = None,
        noise_sd: float = 0.0,
        ecfp_bits: int = 512,
        ecfp_radius: int = 2,
    ):
        self.target_col = target_col
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.smiles_cols = smiles_cols or []
        self.noise_sd = float(noise_sd)
        self.ecfp_bits = int(ecfp_bits)
        self.ecfp_radius = int(ecfp_radius)
        self.base_model = base_model or RandomForestRegressor(n_estimators=500, random_state=0)

        transformers = []
        if self.numeric_cols:
            transformers.append(("num", StandardScaler(), self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", _make_ohe(), self.categorical_cols))
        if self.smiles_cols:
            transformers.append(("smi", _smiles_transformer(self.smiles_cols, self.ecfp_bits, self.ecfp_radius), self.smiles_cols))

        self.pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
        self.model = Pipeline(steps=[("pre", self.pre), ("reg", self.base_model)])

    def fit_df(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 0) -> OracleReport:
        assert self.target_col in df.columns, f"Missing target column {self.target_col}"
        cat = _safe_cols(df, self.categorical_cols)
        num = _safe_cols(df, self.numeric_cols)
        smi = _safe_cols(df, self.smiles_cols)
        self.categorical_cols, self.numeric_cols, self.smiles_cols = cat, num, smi

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        X_train = train_df[cat + num + smi]
        y_train = train_df[self.target_col].astype(float).values
        X_test = test_df[cat + num + smi]
        y_test = test_df[self.target_col].astype(float).values

        self.model.fit(X_train, y_train)

        yhat_tr = self.model.predict(X_train)
        yhat_te = self.model.predict(X_test)

        report = OracleReport(
            r2_train=float(r2_score(y_train, yhat_tr)),
            r2_test=float(r2_score(y_test, yhat_te)),
            mae_train=float(mean_absolute_error(y_train, yhat_tr)),
            mae_test=float(mean_absolute_error(y_test, yhat_te)),
            n_train=len(train_df),
            n_test=len(test_df),
            model_summary={
                "estimator": type(self.base_model).__name__,
                "categorical_cols": cat,
                "numeric_cols": num,
                "smiles_cols": smi,
                "noise_sd": self.noise_sd,
            },
        )
        self._fitted_ = True
        return report

    def predict_df(self, X: pd.DataFrame, return_std: bool = False):
        if not hasattr(self, "_fitted_"):
            raise RuntimeError("Oracle is not fitted. Call fit_df first.")
        cols = self.categorical_cols + self.numeric_cols + self.smiles_cols
        mu = self.model.predict(X[cols])
        if self.noise_sd > 0.0:
            mu = mu + np.random.normal(0.0, self.noise_sd, size=len(mu))
        if return_std:
            std = None
            try:
                reg = self.model.named_steps["reg"]
                if hasattr(reg, "estimators_"):
                    preds = np.stack([est.predict(self.model.named_steps["pre"].transform(X[cols])) for est in reg.estimators_], axis=0)
                    std = preds.std(axis=0)
            except Exception:
                std = None
            if std is None:
                std = np.full_like(mu, fill_value=max(self.noise_sd, 1.0), dtype=float)
            return mu, std
        return mu

    def save(self, path: str):
        if joblib is None:
            raise ImportError("joblib is required for save/load. pip install joblib")
        joblib.dump({
            "target_col": self.target_col,
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
            "smiles_cols": self.smiles_cols,
            "noise_sd": self.noise_sd,
            "ecfp_bits": self.ecfp_bits,
            "ecfp_radius": self.ecfp_radius,
            "model": self.model
        }, path)

    @staticmethod
    def load(path: str):
        if joblib is None:
            raise ImportError("joblib is required for save/load. pip install joblib")
        blob = joblib.load(path)
        obj = ReactionOracle(
            target_col=blob["target_col"],
            categorical_cols=blob["categorical_cols"],
            numeric_cols=blob["numeric_cols"],
            smiles_cols=blob["smiles_cols"],
            noise_sd=blob["noise_sd"],
            ecfp_bits=blob["ecfp_bits"],
            ecfp_radius=blob["ecfp_radius"],
        )
        obj.model = blob["model"]
        obj._fitted_ = True
        return obj

def fit_oracle_from_csv(
    csv_path: str,
    target_col: str,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    smiles_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 0,
    noise_sd: float = 0.0,
):
    df = pd.read_csv(csv_path)
    oracle = ReactionOracle(
        target_col=target_col,
        categorical_cols=categorical_cols or [],
        numeric_cols=numeric_cols or [],
        smiles_cols=smiles_cols or [],
        noise_sd=noise_sd,
    )
    report = oracle.fit_df(df, test_size=test_size, random_state=random_state)
    return oracle, report
