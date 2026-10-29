"""Train the baseline CNN on one fold split and report validation/test metrics.

Fold roles come from norm_stats.json (written by preprocess/normalize.py).

Usage (from the repo root):
  python -m train.train --data-dir /cluster/scratch/$USER/copper_dataset \
      --out-root /cluster/scratch/$USER/copper_runs --run-name baseline
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from preprocess.normalize import normalize_tiles
from train.model import CopperCNN


def load_normalized(path, stats, chunk=5000):
    raw = np.load(path, mmap_mode="r")
    out = np.empty(raw.shape, dtype=np.float32)
    for i in range(0, len(raw), chunk):
        out[i:i + chunk] = normalize_tiles(np.asarray(raw[i:i + chunk]), stats)
    return out


def augment(x):
    k = int(torch.randint(0, 4, (1,)))
    if k:
        x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


@torch.no_grad()
def predict(model, x, device, batch_size=1024):
    model.eval()
    logits, logs = [], []
    for i in range(0, len(x), batch_size):
        b_logit, b_log = model(x[i:i + batch_size].to(device))
        logits.append(b_logit.cpu())
        logs.append(b_log.cpu())
    return torch.cat(logits), torch.cat(logs).numpy()


def score(y_bin, y_log, p_bin, pred_log):
    return {
        "roc_auc_bin": float(roc_auc_score(y_bin, p_bin)),
        "pr_auc_bin": float(average_precision_score(y_bin, p_bin)),
        "roc_auc_log": float(roc_auc_score(y_bin, pred_log)),
        "pr_auc_log": float(average_precision_score(y_bin, pred_log)),
        "spearman_log": float(spearmanr(pred_log, y_log).correlation),
        "pos_rate": float(np.mean(y_bin)),
        "n": int(len(y_bin)),
    }


def main():
    user = os.environ.get("USER", "user")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=f"/cluster/scratch/{user}/copper_dataset")
    p.add_argument("--out-root", default=f"/cluster/scratch/{user}/copper_runs")
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--w-bin", type=float, default=1.0)
    p.add_argument("--w-log", type=float, default=0.5)
    p.add_argument("--buffer-km", type=float, default=3.2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "norm_stats.json")) as f:
        stats = json.load(f)
    cells = pd.read_csv(os.path.join(args.data_dir, "fold_cells.csv"))
    tiles = load_normalized(os.path.join(args.data_dir, "tiles.npy"), stats)

    folds = cells["fold"].values
    val_idx = np.flatnonzero(folds == stats["val_fold"])
    test_idx = np.flatnonzero(folds == stats["test_fold"])
    train_idx = np.flatnonzero(np.isin(folds, stats["train_folds"]))
    xy = cells[["x", "y"]].values
    near = cKDTree(xy[np.concatenate([val_idx, test_idx])]).query(xy[train_idx], distance_upper_bound=args.buffer_km * 1000)[0]
    n_before = len(train_idx)
    train_idx = train_idx[~np.isfinite(near)]
    print(f"device {device}; train {len(train_idx)} (dropped {n_before - len(train_idx)} within {args.buffer_km} km "
          f"of val/test), val {len(val_idx)}, test {len(test_idx)}")

    y_bin = cells["y_bin"].values.astype(np.float32)
    y_log = np.clip(cells["y_log"].values, 1, 5).astype(np.float32)
    x_all = torch.from_numpy(tiles)
    x_train = x_all[train_idx]
    yb_train = torch.from_numpy(y_bin[train_idx])
    yl_train = torch.from_numpy(y_log[train_idx])

    n_pos = float(yb_train.sum())
    pos_weight = torch.tensor((len(train_idx) - n_pos) / n_pos, device=device)
    print(f"train positives {int(n_pos)} ({n_pos / len(train_idx):.1%}), pos_weight {pos_weight.item():.1f}")

    model = CopperCNN(in_channels=tiles.shape[1]).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    select_key = "pr_auc_bin" if args.w_bin > 0 else "pr_auc_log"

    history, best, best_epoch, best_state = [], -np.inf, -1, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train_idx))
        sums = {"bce": 0.0, "mse": 0.0}
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            xb = augment(x_train[b]).to(device)
            logit, pred_log = model(xb)
            l_bce = bce(logit, yb_train[b].to(device))
            l_mse = mse(pred_log, yl_train[b].to(device))
            loss = args.w_bin * l_bce + args.w_log * l_mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums["bce"] += l_bce.item() * len(b)
            sums["mse"] += l_mse.item() * len(b)

        logit_val, log_val = predict(model, x_all[val_idx], device)
        val = score(y_bin[val_idx], y_log[val_idx], torch.sigmoid(logit_val).numpy(), log_val)
        row = {"epoch": epoch, "train_bce": sums["bce"] / len(perm), "train_mse": sums["mse"] / len(perm),
               "val_bce": float(bce(logit_val.to(device), torch.from_numpy(y_bin[val_idx]).to(device)).item()),
               "val_mse": float(np.mean((log_val - y_log[val_idx]) ** 2)),
               **{f"val_{k}": v for k, v in val.items() if k not in ("n", "pos_rate")},
               "seconds": time.time() - t0}
        history.append(row)
        improved = val[select_key] > best
        if improved:
            best, best_epoch = val[select_key], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  train bce {row['train_bce']:.3f} mse {row['train_mse']:.3f} | "
              f"val bce {row['val_bce']:.3f} mse {row['val_mse']:.3f} | val roc {val['roc_auc_bin']:.3f} "
              f"pr {val['pr_auc_bin']:.3f} | log roc {val['roc_auc_log']:.3f} spearman {val['spearman_log']:.3f} "
              f"| {row['seconds']:.0f}s{'  *' if improved else ''}")
        if epoch - best_epoch >= args.patience:
            print(f"Early stop: no {select_key} improvement for {args.patience} epochs")
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "args": vars(args), "norm_stats": stats}, os.path.join(out_dir, "best.pt"))

    results = {"best_epoch": best_epoch, "select_key": select_key, "args": vars(args)}
    for name, idx in [("val", val_idx), ("test", test_idx)]:
        logit, pred_log = predict(model, x_all[idx], device)
        p_bin = torch.sigmoid(logit).numpy()
        results[name] = score(y_bin[idx], y_log[idx], p_bin, pred_log)
        if name == "test":
            preds = cells.iloc[idx][["row", "col", "x", "y", "cu_ppm", "y_bin", "y_log"]].copy()
            preds["p_bin"], preds["pred_log"] = p_bin, pred_log
            preds.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best epoch {best_epoch} ({select_key} {best:.3f})")
    for name in ("val", "test"):
        r = results[name]
        print(f"  {name:4s}: bin ROC-AUC {r['roc_auc_bin']:.3f}, PR-AUC {r['pr_auc_bin']:.3f} "
              f"(random = {r['pos_rate']:.3f}) | log ROC-AUC {r['roc_auc_log']:.3f}, PR-AUC {r['pr_auc_log']:.3f}, "
              f"Spearman {r['spearman_log']:.3f} | n={r['n']}")
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()
