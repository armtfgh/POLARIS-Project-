#!/usr/bin/env python3
"""
Train a neural prior-prediction model on the synthetic meta dataset.

Features:
  - Rich summary statistics, directional bins, percentiles, trend coefficients.
  - Rasterized observation maps (mean + count channels) fed into a small CNN.
  - Multi-task architecture that jointly predicts x1/x2 effect classes,
    interaction class, and continuous parameters (scale/confidence/range/bumps).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE_MAP = {"increase": 0, "decrease": 1, "peak": 2, "valley": 3, "none": 4}
INTER_MAP = {"synergy": 0, "antagonism": 1, "none": 2}
MAX_BUMPS = 2


# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------

def directional_stats(points: np.ndarray, values: np.ndarray) -> List[float]:
    stats = []
    masks = {
        "x1_low": points[:, 0] < 0.3,
        "x1_high": points[:, 0] > 0.7,
        "x2_low": points[:, 1] < 0.3,
        "x2_high": points[:, 1] > 0.7,
    }
    for mask in masks.values():
        stats.append(float(values[mask].mean()) if mask.any() else 0.0)
    return stats


def coarse_heatmap(points: np.ndarray, values: np.ndarray, bins: int) -> List[float]:
    H, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins, weights=values)
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    avg = np.divide(H, counts, out=np.zeros_like(H), where=counts > 0)
    return avg.flatten().tolist()


def rasterize(points: np.ndarray, values: np.ndarray, bins: int = 16) -> np.ndarray:
    sum_grid, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=bins, weights=values)
    count_grid, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    mean_grid = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid), where=count_grid > 0)
    stacked = np.stack([mean_grid, count_grid], axis=0)
    return stacked.astype(np.float32)


def linear_trend(points: np.ndarray, values: np.ndarray) -> List[float]:
    n = len(values)
    if n < 3:
        return [0.0, 0.0, 0.0, 0.0]
    A = np.column_stack([np.ones(n), points[:, 0], points[:, 1], points[:, 0] * points[:, 1]])
    coefs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    return coefs.tolist()


def compute_features(points: List[List[float]], values: List[float], stage_idx: int) -> Tuple[List[float], np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32)
    n = len(vals)
    feats = [float(stage_idx), float(n)]
    if n == 0:
        feats.extend([0.0] * 70)
        raster = np.zeros((2, 16, 16), dtype=np.float32)
        return feats, raster

    feats.extend(pts.mean(axis=0).tolist())
    feats.extend(pts.var(axis=0).tolist())
    feats.append(float(vals.mean()))
    feats.append(float(vals.std()))
    feats.extend(np.percentile(vals, [10, 25, 50, 75, 90]).tolist())

    best = np.argmax(vals)
    worst = np.argmin(vals)
    feats.extend([float(vals[best]), float(pts[best, 0]), float(pts[best, 1])])
    feats.extend([float(vals[worst]), float(pts[worst, 0]), float(pts[worst, 1])])
    feats.append(float(vals[best] - vals.mean()))
    feats.append(float(vals.mean() - vals[worst]))

    feats.extend(directional_stats(pts, vals))
    feats.extend(coarse_heatmap(pts, vals, bins=4))
    feats.extend(coarse_heatmap(pts, vals, bins=6))
    feats.extend(linear_trend(pts, vals))
    corr_x1 = np.corrcoef(pts[:, 0], vals)[0, 1] if n > 2 else 0.0
    corr_x2 = np.corrcoef(pts[:, 1], vals)[0, 1] if n > 2 else 0.0
    feats.extend([float(np.nan_to_num(corr_x1)), float(np.nan_to_num(corr_x2))])

    raster = rasterize(pts, vals, bins=16)
    return feats, raster


# --------------------------------------------------------------------------------------
# Target encoding helpers
# --------------------------------------------------------------------------------------

def encode_effect(prior: Dict, dim: str) -> Tuple[int, List[float]]:
    eff = (prior.get("effects") or {}).get(dim)
    if not eff:
        return TYPE_MAP["none"], [0.0, 0.0, 0.0, 1.0]
    eff_type = TYPE_MAP.get(eff.get("effect", "none"), TYPE_MAP["none"])
    scale = float(eff.get("scale", 0.0))
    conf = float(eff.get("confidence", 0.0))
    rh = eff.get("range_hint", [0.0, 1.0])
    lo, hi = (float(rh[0]), float(rh[1])) if isinstance(rh, (list, tuple)) and len(rh) == 2 else (0.0, 1.0)
    return eff_type, [scale, conf, lo, hi]


def encode_interaction(prior: Dict) -> Tuple[int, List[float]]:
    inters = prior.get("interactions") or []
    if not inters:
        return INTER_MAP["none"], [0.0, 0.0]
    inter = inters[0]
    itype = INTER_MAP.get(inter.get("type", "none"), INTER_MAP["none"])
    return itype, [float(inter.get("scale", 0.0)), float(inter.get("confidence", 0.0))]


def encode_bumps(prior: Dict, max_bumps: int = MAX_BUMPS) -> List[float]:
    bumps = prior.get("bumps") or []
    out: List[float] = []
    for b in bumps[:max_bumps]:
        mu = b.get("mu", [0.5, 0.5])
        sigma = b.get("sigma", [0.1, 0.1])
        out.extend([float(mu[0]), float(mu[1]), float(sigma[0]), float(sigma[1]), float(b.get("amp", 0.0))])
    while len(out) < max_bumps * 5:
        out.extend([0.0, 0.0, 0.1, 0.1, 0.0])
    return out


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

@dataclass
class SampleRecord:
    features: List[float]
    raster: np.ndarray
    eff_x1: int
    eff_x2: int
    inter: int
    continuous: List[float]


def load_records(dataset_dir: Path) -> List[SampleRecord]:
    with open(dataset_dir / "meta_dataset.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    records: List[SampleRecord] = []
    for rec in raw:
        feats, raster = compute_features(rec["sample_points"], rec["sample_values"], rec["stage_idx"])
        prior = rec["prior"]
        eff_x1_type, eff_x1_cont = encode_effect(prior, "x1")
        eff_x2_type, eff_x2_cont = encode_effect(prior, "x2")
        inter_type, inter_cont = encode_interaction(prior)
        bumps = encode_bumps(prior)
        cont = eff_x1_cont + eff_x2_cont + inter_cont + bumps
        records.append(
            SampleRecord(
                features=feats,
                raster=raster,
                eff_x1=eff_x1_type,
                eff_x2=eff_x2_type,
                inter=inter_type,
                continuous=cont,
            )
        )
    return records


class PriorDataset(Dataset):
    def __init__(self, records: List[SampleRecord], scaler: StandardScaler | None = None, augment_swap: bool = False):
        self.records = records
        self.augment_swap = augment_swap
        feats = np.array([r.features for r in records], dtype=np.float32)
        self.scaler = scaler or StandardScaler().fit(feats)
        self.features = self.scaler.transform(feats).astype(np.float32)
        self.rasters = np.stack([r.raster for r in records]).astype(np.float32)
        self.eff_x1 = np.array([r.eff_x1 for r in records], dtype=np.int64)
        self.eff_x2 = np.array([r.eff_x2 for r in records], dtype=np.int64)
        self.inter = np.array([r.inter for r in records], dtype=np.int64)
        self.cont = np.stack([r.continuous for r in records]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        feat = self.features[idx]
        raster = self.rasters[idx]
        eff_x1 = self.eff_x1[idx]
        eff_x2 = self.eff_x2[idx]
        inter = self.inter[idx]
        cont = self.cont[idx]

        if self.augment_swap and np.random.rand() < 0.5:
            raster = np.ascontiguousarray(raster[:, :, ::-1])  # horizontal flip
        else:
            raster = np.ascontiguousarray(raster)
        sample = {
            "features": torch.from_numpy(feat),
            "raster": torch.from_numpy(raster),
            "eff_x1": torch.tensor(eff_x1, dtype=torch.long),
            "eff_x2": torch.tensor(eff_x2, dtype=torch.long),
            "inter": torch.tensor(inter, dtype=torch.long),
            "cont": torch.from_numpy(cont),
        }
        return sample


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class PriorNet(nn.Module):
    def __init__(self, feat_dim: int, cont_dim: int, raster_bins: int = 16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        cnn_out = 64 * 2 * 2
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        hidden_dim = cnn_out + 128
        self.shared = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU())
        self.head_x1 = nn.Linear(256, len(TYPE_MAP))
        self.head_x2 = nn.Linear(256, len(TYPE_MAP))
        self.head_inter = nn.Linear(256, len(INTER_MAP))
        self.head_cont = nn.Linear(256, cont_dim)

    def forward(self, features: torch.Tensor, rasters: torch.Tensor):
        cnn_out = self.cnn(rasters).view(rasters.size(0), -1)
        feat_out = self.feat_net(features)
        h = torch.cat([cnn_out, feat_out], dim=1)
        h = self.shared(h)
        return {
            "x1": self.head_x1(h),
            "x2": self.head_x2(h),
            "inter": self.head_inter(h),
            "cont": self.head_cont(h),
        }


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------

def train_model(
    dataset_dir: Path,
    model_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    test_split: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    records = load_records(dataset_dir)
    idx = np.arange(len(records))
    np.random.shuffle(idx)
    split = int(len(idx) * (1.0 - test_split))
    train_idx, val_idx = idx[:split], idx[split:]
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    scaler = StandardScaler().fit(np.array([r.features for r in train_records]))
    train_ds = PriorDataset(train_records, scaler=scaler, augment_swap=True)
    val_ds = PriorDataset(val_records, scaler=scaler, augment_swap=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    feat_dim = train_ds.features.shape[1]
    cont_dim = train_ds.cont.shape[1]
    model = PriorNet(feat_dim=feat_dim, cont_dim=cont_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    w_cls = 1.0
    w_reg = 0.5

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            feats = batch["features"].to(DEVICE)
            rasters = batch["raster"].to(DEVICE)
            outputs = model(feats, rasters)
            loss_cls = (
                criterion_cls(outputs["x1"], batch["eff_x1"].to(DEVICE))
                + criterion_cls(outputs["x2"], batch["eff_x2"].to(DEVICE))
                + criterion_cls(outputs["inter"], batch["inter"].to(DEVICE))
            )
            loss_reg = criterion_reg(outputs["cont"], batch["cont"].to(DEVICE))
            loss = w_cls * loss_cls + w_reg * loss_reg
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_ds)

        model.eval()
        correct_x1 = correct_x2 = correct_inter = total = 0
        reg_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(DEVICE)
                rasters = batch["raster"].to(DEVICE)
                outputs = model(feats, rasters)
                pred_x1 = outputs["x1"].argmax(dim=1).cpu()
                pred_x2 = outputs["x2"].argmax(dim=1).cpu()
                pred_inter = outputs["inter"].argmax(dim=1).cpu()
                correct_x1 += (pred_x1 == batch["eff_x1"]).sum().item()
                correct_x2 += (pred_x2 == batch["eff_x2"]).sum().item()
                correct_inter += (pred_inter == batch["inter"]).sum().item()
                reg_loss += (
                    criterion_reg(outputs["cont"], batch["cont"].to(DEVICE)).item() * feats.size(0)
                )
                total += feats.size(0)
        print(
            f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} | "
            f"val_acc_x1={correct_x1/total:.3f} | val_acc_x2={correct_x2/total:.3f} | "
            f"val_acc_inter={correct_inter/total:.3f} | val_reg_loss={reg_loss/total:.4f}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "prior_net.pt")
    np.save(model_dir / "scaler_mean.npy", train_ds.scaler.mean_)
    np.save(model_dir / "scaler_scale.npy", train_ds.scaler.scale_)
    print(f"[train_on_meta] Saved model and scaler to {model_dir}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural prior model on meta dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("meta_dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("prior_net_model"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        test_split=args.test_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
