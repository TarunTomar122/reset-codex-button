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
    p.add_argument("--checkpoint", default="outputs/order_0/checkpoints/best.pt")
    p.add_argument("--env-id", default="ResetButton-v6")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--qpos-noise", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"checkpoint iter={ckpt['iteration']} steps={ckpt['steps']}", flush=True)

    env = gym.make(
        args.env_id,
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode=None,
        robot_init_qpos_noise=args.qpos_noise,
    )
    order_ok = red_first = blue_only = no_blue = 0
    ok_steps = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        info = {}
        truncated = [False]
        while not done and steps < 200:
            with torch.no_grad():
                mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                action = mean.clamp(-1.0, 1.0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated[0]) or bool(truncated[0])
        if bool(info.get("order_success", torch.tensor([False]))[0]):
            order_ok += 1
            ok_steps.append(steps)
        elif bool(info.get("red_first", torch.tensor([False]))[0]):
            red_first += 1
        elif bool(truncated[0]) and bool(info.get("blue_done", torch.tensor([False]))[0]):
            blue_only += 1
        else:
            no_blue += 1
    n = args.episodes
    print(
        f"noise={args.qpos_noise} episodes={n}: "
        f"order_success={order_ok}/{n} ({order_ok/n:.0%}) "
        f"red_first={red_first} blue_then_timeout={blue_only} never_blue={no_blue}"
    )
    if ok_steps:
        print(f"mean steps when successful: {np.mean(ok_steps):.1f}")
    env.close()


if __name__ == "__main__":
    main()
