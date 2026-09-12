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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="ResetButton-v3")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--qpos-noise", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    env = gym.make(
        args.env_id,
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode=None,
        robot_init_qpos_noise=args.qpos_noise,
    )
    successes = 0
    steps_list = []
    jerks = []
    mags = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        prev_action = np.zeros(4, dtype=np.float32)
        info = {}
        while not done and steps < 100:
            with torch.no_grad():
                mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                action = mean.clamp(-1.0, 1.0).numpy().reshape(-1)
            jerks.append(float(((action - prev_action) ** 2).sum()))
            mags.append(float((action**2).sum()))
            prev_action = action
            obs, _, terminated, truncated, info = env.step(action.reshape(1, -1))
            steps += 1
            done = bool(terminated[0]) or bool(truncated[0])
        successes += int(bool(info["success"][0]))
        steps_list.append(steps)
    print(
        f"{args.checkpoint} ({ckpt['steps']} steps) env={args.env_id} noise={args.qpos_noise}: "
        f"success={successes}/{args.episodes} steps={np.mean(steps_list):.1f} "
        f"jerk={np.mean(jerks):.4f} action_mag={np.mean(mags):.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
