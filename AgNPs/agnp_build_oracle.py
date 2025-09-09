
#!/usr/bin/env python
"""
Train an AgNP oracle (loss predictor) from AgNP_dataset.csv and save oracle.pkl

Inputs:
  --data_csv /path/to/AgNP_dataset.csv
  --out_dir  agnp_oracle_out

The CSV is expected to contain columns:
  ['QAgNO3(%)','Qpva(%)','Qtsc(%)','Qseed(%)','Qtot(uL/min)','loss']

We fit a RandomForestRegressor to predict "loss" and persist a dict with:
  - 'model' : fitted estimator
  - 'features' : feature column order
  - 'bounds' : recommended bounds for optimization (dict of tuples)
  - 'scaler' : None (kept for API symmetry, in case you switch to a scaled model)

Usage:
  python agnp_build_oracle.py --data_csv /mnt/data/AgNP_dataset.csv --out_dir agnp_oracle_out
"""
import argparse, os, pickle, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", default="agnp_oracle_out")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    expected = ['QAgNO3(%)','Qpva(%)','Qtsc(%)','Qseed(%)','Qtot(uL/min)','loss']
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.data_csv}. Found: {list(df.columns)}")

    X = df[['QAgNO3(%)','Qpva(%)','Qtsc(%)','Qseed(%)','Qtot(uL/min)']].values
    y = df['loss'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    os.makedirs(args.out_dir, exist_ok=True)
    out_pkl = os.path.join(args.out_dir, "oracle.pkl")
    payload = {
        "model": rf,
        "features": ['QAgNO3(%)','Qpva(%)','Qtsc(%)','Qseed(%)','Qtot(uL/min)'],
        "bounds": {
            'QAgNO3(%)': (0.5, 80.0),
            'Qpva(%)': (10.0, 40.0),
            'Qtsc(%)': (0.5, 80.0),
            'Qseed(%)': (0.5, 80.0),
            'Qtot(uL/min)': (200.0, 1000.0),
        },
        "metrics": {"r2": float(r2), "mae": float(mae)},
        "note": "RandomForestRegressor predicting 'loss'. Lower = better."
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    report = os.path.join(args.out_dir, "training_report.json")
    with open(report, "w") as f:
        json.dump(payload["metrics"], f, indent=2)

    print(f"[OK] Saved oracle to: {out_pkl}")
    print(f"[INFO] Test R2={r2:.3f}, MAE={mae:.4f}")
    print(f"[INFO] Feature order: {payload['features']}")

if __name__ == "__main__":
    main()
