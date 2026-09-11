import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import reset_env  # noqa: F401
from rl.policy import GaussianPolicy


def flat_obs(obs):
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def to_numpy_frame(render_out):
    if hasattr(render_out, "detach"):
        render_out = render_out.detach().cpu().numpy()
    arr = np.asarray(render_out)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--record-episodes", type=int, default=2)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("-o", "--output", default="outputs/videos/trained_press.mp4")
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(
        f"loaded iteration {ckpt['iteration']} ({ckpt['steps']} steps)",
        flush=True,
    )

    env = gym.make(
        "ResetButton-v1",
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode="rgb_array",
    )
    successes = 0
    lengths = []
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        info = {}
        while not done:
            with torch.no_grad():
                mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                action = mean.clamp(-1.0, 1.0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated[0]) or bool(truncated[0])
            if ep < args.record_episodes:
                frames.append(to_numpy_frame(env.render()))
        successes += int(bool(info["success"][0]))
        lengths.append(steps)
        print(f"episode {ep}: success={bool(info['success'][0])} steps={steps}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    iio.imwrite(args.output, frames, fps=args.fps)
    print(
        f"success {successes}/{args.episodes} "
        f"mean_steps={np.mean(lengths):.1f} wrote {args.output}: {len(frames)} frames",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
