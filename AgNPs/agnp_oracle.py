
#!/usr/bin/env python
"""
AgNP black-box oracle.

API:
  from agnp_oracle import AgNPOracle
  oracle = AgNPOracle("agnp_oracle_out/oracle.pkl")
  y = oracle.evaluate_one({"QAgNO3(%)": 12.0, "Qpva(%)": 20.0, "Qtsc(%)": 5.0, "Qseed(%)": 4.0, "Qtot(uL/min)": 400.0})
  ys = oracle.evaluate_batch(list_of_dicts)

- Returns predicted 'loss' (lower is better).
- Supports soft out-of-bounds penalty via clamp or penalty mode.
"""
import pickle, numpy as np

class AgNPOracle:
    def __init__(self, oracle_pkl, oob_mode="penalize", noise_std=0.0, penalty=0.15):
        with open(oracle_pkl, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.features = payload["features"]
        self.bounds = payload["bounds"]
        self.oob_mode = oob_mode
        self.noise_std = float(noise_std)
        self.penalty = float(penalty)

    def _prepare_X(self, xdicts):
        X = []
        penalties = []
        for d in xdicts:
            row = []
            p = 0.0
            for f in self.features:
                v = float(d[f])
                lo, hi = self.bounds[f]
                if v < lo or v > hi:
                    if self.oob_mode == "clamp":
                        v = min(max(v, lo), hi)
                    elif self.oob_mode == "penalize":
                        p += self.penalty * (abs(v - lo) if v < lo else abs(v - hi)) / (hi - lo)
                row.append(v)
            X.append(row)
            penalties.append(p)
        return np.array(X, dtype=float), np.array(penalties, dtype=float)

    def evaluate_batch(self, xdicts):
        X, penalties = self._prepare_X(xdicts)
        y = self.model.predict(X)
        if self.noise_std > 0:
            y = y + np.random.normal(0, self.noise_std, size=y.shape)
        return (y + penalties).tolist()

    def evaluate_one(self, xdict):
        return self.evaluate_batch([xdict])[0]
