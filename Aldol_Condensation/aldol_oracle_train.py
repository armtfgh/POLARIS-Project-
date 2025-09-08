
#!/usr/bin/env python
# aldol_oracle_train.py (patched)
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

try:
    import joblib
except Exception:
    joblib = None

def make_model(kind: str, random_state=0):
    k = kind.lower()
    if k == "rf":
        return RandomForestRegressor(n_estimators=600, random_state=random_state, n_jobs=-1)
    if k == "gbr":
        return GradientBoostingRegressor(random_state=random_state)
    if k == "mlp":
        return MLPRegressor(hidden_layer_sizes=(128,64), activation="relu", early_stopping=True,
                            max_iter=800, random_state=random_state)
    raise ValueError("model must be one of: rf, gbr, mlp")

def train(data_csv: str, out_dir: str, model="rf", test_size=0.2, random_state=0, cv_folds=5):
    df = pd.read_csv(data_csv)
    feat_cols = [c for c in df.columns if c != 'yld']
    assert 'yld' in df.columns and len(feat_cols) >= 2, "Expected 'yld' and at least 2 features"
    df = df.dropna(subset=feat_cols + ['yld']).reset_index(drop=True)
    X = df[feat_cols].copy()
    y = df['yld'].astype(float).values
    pre = ColumnTransformer([("num", StandardScaler(), feat_cols)], remainder="drop")
    est = make_model(model, random_state=random_state)
    pipe = Pipeline([("pre", pre), ("reg", est)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe.fit(Xtr, ytr)
    pred_tr, pred_te = pipe.predict(Xtr), pipe.predict(Xte)
    r2_tr, r2_te = r2_score(ytr, pred_tr), r2_score(yte, pred_te)
    mae_tr, mae_te = mean_absolute_error(ytr, pred_tr), mean_absolute_error(yte, pred_te)

    # CV
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv = []
    for tr, va in kf.split(X):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict(X.iloc[va])
        cv.append({"r2": float(r2_score(y[va], p)), "mae": float(mean_absolute_error(y[va], p))})

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if joblib:
        import joblib as jb
        jb.dump(pipe, out/"oracle.pkl")
    report = {"rows": int(len(df)), "features": feat_cols, "model": model,
              "metrics": {"r2_train": float(r2_tr), "r2_test": float(r2_te),
                          "mae_train": float(mae_tr), "mae_test": float(mae_te), "cv": cv}}
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", default="aldol_oracle_out")
    ap.add_argument("--model", default="rf", choices=["rf","gbr","mlp"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--cv_folds", type=int, default=5)
    args = ap.parse_args()
    train(args.data_csv, args.out_dir, args.model, args.test_size, args.random_state, args.cv_folds)
