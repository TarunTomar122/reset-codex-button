import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from rl.policy import GaussianPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="outputs/demos/order_demos.npz")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", default="outputs/bc/checkpoints/best.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.demos)
    obs = torch.from_numpy(data["obs"])
    actions = torch.from_numpy(data["actions"])
    n = obs.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_obs, train_act = obs[train_idx], actions[train_idx]
    val_obs, val_act = obs[val_idx], actions[val_idx]

    policy = GaussianPolicy(obs.shape[1], actions.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    best_val = float("inf")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        policy.train()
        perm = torch.randperm(train_obs.shape[0])
        losses = []
        for start in range(0, train_obs.shape[0], args.batch):
            mb = perm[start : start + args.batch]
            mean, _ = policy(train_obs[mb])
            loss = nn.functional.mse_loss(mean, train_act[mb])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == args.epochs:
            policy.eval()
            with torch.no_grad():
                val_mean, _ = policy(val_obs)
                val_loss = nn.functional.mse_loss(val_mean, val_act).item()
            print(
                f"epoch {epoch:4d} train_mse={np.mean(losses):.5f} val_mse={val_loss:.5f}",
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "obs_dim": obs.shape[1],
                        "act_dim": actions.shape[1],
                        "iteration": 0,
                        "steps": 0,
                        "label": f"Behavior cloning from {n} oracle transitions",
                        "args": vars(args),
                    },
                    out,
                )
    print(f"saved best val_mse={best_val:.5f} to {out}", flush=True)


if __name__ == "__main__":
    main()
