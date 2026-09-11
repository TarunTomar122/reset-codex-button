import argparse
import os
import time

os.environ.setdefault("VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")

import gymnasium as gym
import imageio.v3 as iio
import mani_skill.envs
import numpy as np


def to_numpy_frame(render_out):
    if hasattr(render_out, "detach"):
        render_out = render_out.detach().cpu().numpy()
    arr = np.asarray(render_out)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--env-id", default="Reach-v1")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("-o", "--output", default="outputs/videos/random_action.mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-mode", default="state")
    args = p.parse_args()

    env = gym.make(args.env_id, obs_mode=args.obs_mode, render_mode="rgb_array", num_envs=1)
    env.reset(seed=args.seed)

    frames = []
    t0 = time.time()
    for _ in range(args.steps):
        action = env.action_space.sample()
        env.step(action)
        frames.append(to_numpy_frame(env.render()))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    iio.imwrite(args.output, frames, fps=args.fps)
    print(f"wrote {args.output}: {len(frames)} frames in {time.time() - t0:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
