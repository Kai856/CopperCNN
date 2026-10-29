"""Train the CNN on one fold split and report validation/test metrics.

Fold roles come from norm_stats.json (written by preprocess/normalize.py).
Optional inputs: --rock (rock.npy categorical tiles), --depth (depth_m column),
--crop N (centre-crop tiles, e.g. 64 -> 32).

Usage (from the repo root):
  python -m train.train --data-dir /cluster/scratch/$USER/copper_dataset64 \
      --out-root /cluster/scratch/$USER/copper_runs --run-name wide64_rock --rock
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


def load_normalized(path, stats, crop=None, chunk=5000):
    raw = np.load(path, mmap_mode="r")
    size = raw.shape[-1] if crop is None else crop
    s = (raw.shape[-1] - size) // 2
    out = np.empty(raw.shape[:2] + (size, size), dtype=np.float32)
    for i in range(0, len(raw), chunk):
        out[i:i + chunk] = normalize_tiles(np.asarray(raw[i:i + chunk, :, s:s + size, s:s + size]), stats)
    return out


def load_rock(path, crop=None):
    raw = np.load(path, mmap_mode="r")
    size = raw.shape[-1] if crop is None else crop
    s = (raw.shape[-1] - size) // 2
    return np.array(raw[:, :, s:s + size, s:s + size])


def augment(*tiles):
    k = int(torch.randint(0, 4, (1,)))
    flip = torch.rand(1).item() < 0.5
    out = []
    for t in tiles:
        if t is not None:
            if k:
                t = torch.rot90(t, k, dims=(2, 3))
            if flip:
                t = torch.flip(t, dims=(3,))
        out.append(t)
    return out


class Inputs:
    """Holds every model input for all cells; batch(idx) returns them on the device."""

    def __init__(self, x, rock, extra, device):
        self.x, self.rock, self.extra, self.device = x, rock, extra, device

    def batch(self, idx, train=False):
        x = self.x[idx]
        rock = self.rock[idx].long() if self.rock is not None else None
        if train:
            x, rock = augment(x, rock)
        extra = self.extra[idx].to(self.device) if self.extra is not None else None
        return x.to(self.device), rock.to(self.device) if rock is not None else None, extra


@torch.no_grad()
def predict(model, inputs, idx, batch_size=1024):
    model.eval()
    logits, logs = [], []
    for i in range(0, len(idx), batch_size):
        b_logit, b_log = model(*inputs.batch(idx[i:i + batch_size]))
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
    p.add_argument("--data-dir", default=f"/cluster/scratch/{user}/copper_dataset64")
    p.add_argument("--out-root", default=f"/cluster/scratch/{user}/copper_runs")
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--rock", action="store_true", help="add rock unit embeddings from rock.npy")
    p.add_argument("--depth", action="store_true", help="add log10 hole depth (depth_m) as an input")
    p.add_argument("--crop", type=int, default=None, help="centre-crop tiles to this size")
    p.add_argument("--res", type=float, default=100.0, help="grid cell size in metres")
    p.add_argument("--buffer-km", type=float, default=None, help="default: tile width")
    p.add_argument("--w-bin", type=float, default=1.0)
    p.add_argument("--w-log", type=float, default=0.5)
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
    tiles = load_normalized(os.path.join(args.data_dir, "tiles.npy"), stats, args.crop)
    tile_size = tiles.shape[-1]
    buffer_km = args.buffer_km if args.buffer_km is not None else tile_size * args.res / 1000

    folds = cells["fold"].values
    val_idx = np.flatnonzero(folds == stats["val_fold"])
    test_idx = np.flatnonzero(folds == stats["test_fold"])
    train_idx = np.flatnonzero(np.isin(folds, stats["train_folds"]))
    xy = cells[["x", "y"]].values
    near = cKDTree(xy[np.concatenate([val_idx, test_idx])]).query(xy[train_idx], distance_upper_bound=buffer_km * 1000)[0]
    n_before = len(train_idx)
    train_idx = train_idx[~np.isfinite(near)]
    print(f"device {device}; tiles {tile_size}x{tile_size}; train {len(train_idx)} (dropped {n_before - len(train_idx)} "
          f"within {buffer_km:g} km of val/test), val {len(val_idx)}, test {len(test_idx)}")

    rock, rock_sizes = None, ()
    if args.rock:
        with open(os.path.join(args.data_dir, "rock_layers.json")) as f:
            rock_sizes = tuple(json.load(f)["n_codes"])
        rock = torch.from_numpy(load_rock(os.path.join(args.data_dir, "rock.npy"), args.crop))
        print(f"rock layers: {len(rock_sizes)}, codes per layer {rock_sizes}")

    extra, depth_stats = None, None
    if args.depth:
        log_depth = np.log10(np.clip(cells["depth_m"].values, 1, None))
        missing = ~np.isfinite(log_depth)
        train_vals = log_depth[train_idx][~missing[train_idx]]
        depth_stats = {"median": float(np.median(train_vals)), "mean": float(train_vals.mean()), "std": float(train_vals.std())}
        log_depth = np.where(missing, depth_stats["median"], log_depth)
        extra = torch.from_numpy(((log_depth - depth_stats["mean"]) / depth_stats["std"]).astype(np.float32)[:, None])
        print(f"depth: {missing.sum()} cells missing ({missing.mean():.1%}) -> train median "
              f"{10 ** depth_stats['median']:.0f} m")

    inputs = Inputs(torch.from_numpy(tiles), rock, extra, device)
    y_bin = cells["y_bin"].values.astype(np.float32)
    y_log = np.clip(cells["y_log"].values, 1, 5).astype(np.float32)
    yb_all, yl_all = torch.from_numpy(y_bin), torch.from_numpy(y_log)

    n_pos = float(y_bin[train_idx].sum())
    pos_weight = torch.tensor((len(train_idx) - n_pos) / n_pos, device=device)
    print(f"train positives {int(n_pos)} ({n_pos / len(train_idx):.1%}), pos_weight {pos_weight.item():.1f}")

    model_kwargs = {"in_channels": tiles.shape[1], "rock_sizes": rock_sizes, "n_extra": 1 if args.depth else 0}
    model = CopperCNN(**model_kwargs).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    select_key = "pr_auc_bin" if args.w_bin > 0 else "pr_auc_log"

    history, best, best_epoch, best_state = [], -np.inf, -1, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = train_idx[torch.randperm(len(train_idx)).numpy()]
        sums = {"bce": 0.0, "mse": 0.0}
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            logit, pred_log = model(*inputs.batch(b, train=True))
            l_bce = bce(logit, yb_all[b].to(device))
            l_mse = mse(pred_log, yl_all[b].to(device))
            loss = args.w_bin * l_bce + args.w_log * l_mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums["bce"] += l_bce.item() * len(b)
            sums["mse"] += l_mse.item() * len(b)

        logit_val, log_val = predict(model, inputs, val_idx)
        val = score(y_bin[val_idx], y_log[val_idx], torch.sigmoid(logit_val).numpy(), log_val)
        row = {"epoch": epoch, "train_bce": sums["bce"] / len(perm), "train_mse": sums["mse"] / len(perm),
               "val_bce": float(bce(logit_val.to(device), yb_all[val_idx].to(device)).item()),
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
    torch.save({"state_dict": best_state, "model_kwargs": model_kwargs, "args": vars(args),
                "norm_stats": stats, "depth_stats": depth_stats}, os.path.join(out_dir, "best.pt"))

    results = {"best_epoch": best_epoch, "select_key": select_key, "args": vars(args)}
    for name, idx in [("val", val_idx), ("test", test_idx)]:
        logit, pred_log = predict(model, inputs, idx)
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
