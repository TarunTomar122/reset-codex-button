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

import mani_skill.envs
import reset_env


def to_numpy_frame(render_out):
    if hasattr(render_out, "detach"):
        render_out = render_out.detach().cpu().numpy()
    arr = np.asarray(render_out)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("-o", "--output", default="outputs/videos/oracle_press.mp4")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make(
        "ResetButton-v1",
        obs_mode="state",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
    )
    env.reset(seed=args.seed)
    u = env.unwrapped

    hover_height = 0.05
    phase = "idle"
    idle_steps = 10
    frames = []
    episode_return = 0.0

    for t in range(args.steps):
        tcp = u.agent.tcp_pose.p[0].numpy()
        button_top = u._button_top()[0].numpy()
        depression = float(u._depression()[0])
        action = np.zeros(4, dtype=np.float32)
        action[3] = -1.0

        if phase == "idle":
            if t >= idle_steps:
                phase = "approach"
        elif phase == "approach":
            delta = button_top + np.array([0.0, 0.0, hover_height]) - tcp
            action[:3] = np.clip(delta, -0.005, 0.005) / 0.1
            if np.linalg.norm(delta[:2]) < 0.012 and abs(delta[2]) < 0.012:
                phase = "press"
        elif phase == "press":
            action[2] = -0.08
        else:
            action[2] = -0.2

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward[0])
        if phase == "press" and bool(info["success"][0]):
            phase = "done"
        frames.append(to_numpy_frame(env.render()))

        if t % 5 == 0 or bool(info["success"][0]):
            print(
                f"t={t:3d} phase={phase:8s} "
                f"dist={np.linalg.norm(button_top - tcp):.4f} "
                f"depression={depression:.4f} "
                f"reward={float(reward[0]):+.3f} "
                f"success={bool(info['success'][0])}"
            )

        if bool(terminated[0]) or bool(truncated[0]):
            print("terminated" if bool(terminated[0]) else "truncated")
            break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    iio.imwrite(args.output, frames, fps=args.fps)
    print(f"wrote {args.output}: {len(frames)} frames")
    print(f"episode return: {episode_return:+.2f}")
    env.close()


if __name__ == "__main__":
    main()
