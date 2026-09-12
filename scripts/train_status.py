import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", default="outputs/seed_0/train_log.csv")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.log)))
    if not rows:
        print("no rows yet")
        return
    steps = int(rows[-1]["steps"])
    sps = [float(r["sps"]) for r in rows if float(r["sps"]) > 0]
    avg_sps = sum(sps) / len(sps) if sps else 0.0
    elapsed = sum(
        (int(rows[i]["steps"]) - int(rows[i - 1]["steps"])) / float(rows[i]["sps"])
        for i in range(1, len(rows))
        if float(rows[i]["sps"]) > 0
    )
    remaining = (args.total_steps - steps) / avg_sps if avg_sps else 0.0
    evals = [r for r in rows if r["eval_success"] not in ("nan", "")]
    last_eval = evals[-1] if evals else None
    print(
        f"iter={rows[-1]['iter']:>4}  steps={steps/1000:6.0f}k/{args.total_steps//1000}k  "
        f"elapsed={elapsed/60:4.1f}min  avg={avg_sps:5.0f} steps/s  eta={remaining/60:4.1f}min"
    )
    print(
        f"train return={rows[-1]['mean_return']:>7}  train success={rows[-1]['worker_success']:>5}  "
        + (
            f"last eval success={last_eval['eval_success']} @ {int(last_eval['steps'])//1000}k steps"
            if last_eval
            else "no eval yet"
        )
    )


if __name__ == "__main__":
    main()
