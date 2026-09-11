import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_log(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def to_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="outputs/train_log.csv")
    p.add_argument("-o", "--output", default="outputs/plots/learning_curve.png")
    p.add_argument("--title", default="PPO learns to press a RESET button")
    p.add_argument("--subtitle", default="")
    args = p.parse_args()

    rows = read_log(args.log)
    it = np.array([int(r["iter"]) for r in rows])
    steps = np.array([int(r["steps"]) for r in rows]) / 1000.0
    ret = np.array([to_float(r["mean_return"]) for r in rows])
    worker_succ = np.array([to_float(r["worker_success"]) for r in rows])
    eval_succ = np.array([to_float(r["eval_success"]) for r in rows])
    entropy = np.array([to_float(r["entropy"]) for r in rows])
    vf = np.array([to_float(r["value_loss"]) for r in rows])
    sps = np.array([to_float(r["sps"]) for r in rows])

    per_iter_steps = steps[1] - steps[0] if len(steps) > 1 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(sps > 0, per_iter_steps / (sps / 1000.0), 0.0)
    elapsed_min = np.cumsum(dt) / 60.0
    total_min = float(elapsed_min[-1]) if len(elapsed_min) else 0.0

    solved_idx = np.where(eval_succ >= 0.95)[0]
    solved_step = steps[solved_idx[0]] if len(solved_idx) else None
    solved_min = elapsed_min[solved_idx[0]] if len(solved_idx) else None

    def rolling(x, w=25):
        out = np.full_like(x, np.nan)
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            vals = x[lo : i + 1]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                out[i] = vals.mean()
        return out

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    fig.suptitle(args.title, fontsize=17, fontweight="bold", y=0.98)
    subtitle = args.subtitle or (
        f"ManiSkill3 / SAPIEN, Franka arm, 2x128 MLP, PPO from scratch, "
        f"8 CPU workers on M1 Max - {total_min:.0f} min wall-clock"
    )
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=11, color="#444444")

    ax = axes[0, 0]
    ax.plot(steps, rolling(ret), color="#1f77b4", lw=2, label="episode return (smoothed)")
    ax.plot(steps, ret, color="#1f77b4", alpha=0.15, lw=0.8)
    ax.set_title("Episode return")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("return")
    if solved_step is not None:
        ax.axvline(solved_step, color="#2ca02c", ls="--", lw=1.2, alpha=0.8)

    ax = axes[0, 1]
    ax.plot(steps, worker_succ, color="#ff7f0e", alpha=0.25, lw=0.8)
    ax.plot(steps, rolling(worker_succ, 10), color="#ff7f0e", lw=2, label="train success (smoothed)")
    mask = ~np.isnan(eval_succ)
    ax.plot(steps[mask], eval_succ[mask], "o-", color="#2ca02c", ms=5, lw=1.5, label="eval success (20 episodes)")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Button press success rate")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("success rate")
    ax.legend(loc="lower right", fontsize=9)
    if solved_step is not None and solved_min is not None:
        ax.annotate(
            f"solved in {solved_min:.1f} min",
            xy=(solved_step, 1.0),
            xytext=(solved_step + 0.25 * steps[-1], 0.55),
            arrowprops=dict(arrowstyle="->", color="#2ca02c"),
            color="#2ca02c",
            fontweight="bold",
        )

    ax = axes[1, 0]
    ax.plot(steps, entropy, color="#9467bd", lw=2)
    ax.set_title("Policy entropy (exploration)")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("entropy (nats)")

    ax = axes[1, 1]
    ax.plot(steps, np.maximum(vf, 1e-6), color="#d62728", lw=2)
    ax.set_yscale("log")
    ax.set_title("Value loss")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("MSE (log scale)")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out} ({total_min:.1f} min total)")


if __name__ == "__main__":
    main()
