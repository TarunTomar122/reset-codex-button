import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import reset_env  # noqa: F401
from rl.policy import GaussianPolicy


def flat_obs(obs):
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/seed_0/checkpoints/latest.pt")
    p.add_argument("--env-id", default="ResetButton-v2")
    p.add_argument("--noises", default="0.02,0.05,0.1,0.2,0.3,0.5")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=3000)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    print(f"checkpoint: iteration {ckpt['iteration']} ({ckpt['steps']} steps)")
    print(f"{'qpos_noise':>10} | {'success':>8} | {'mean steps':>10} | {'mean return':>11}")
    for noise in [float(x) for x in args.noises.split(",")]:
        env = gym.make(
            args.env_id,
            obs_mode="state",
            num_envs=1,
            sim_backend="cpu",
            render_mode=None,
            robot_init_qpos_noise=noise,
        )
        successes = 0
        steps_list = []
        returns = []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            steps = 0
            ep_return = 0.0
            info = {}
            while not done and steps < 100:
                with torch.no_grad():
                    mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                    action = mean.clamp(-1.0, 1.0).numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward[0])
                steps += 1
                done = bool(terminated[0]) or bool(truncated[0])
            successes += int(bool(info["success"][0]))
            steps_list.append(steps)
            returns.append(ep_return)
        print(
            f"{noise:>10.2f} | {successes}/{args.episodes:<6} | {np.mean(steps_list):>10.1f} | {np.mean(returns):>11.2f}"
        )
        env.close()


if __name__ == "__main__":
    main()
