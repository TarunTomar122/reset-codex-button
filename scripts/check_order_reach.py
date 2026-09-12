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

import mani_skill.envs  # noqa: F401
import reset_env  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make(
        "ResetButton-v6",
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode=None,
    )
    successes = 0
    red_firsts = 0
    timeouts = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        u = env.unwrapped
        blue_xy = u.button.pose.p[0, :2].numpy()
        red_xy = u.button_red.pose.p[0, :2].numpy()
        action = np.zeros(4, dtype=np.float32)
        action[3] = -1.0
        info = {}
        for step in range(args.max_steps):
            tcp = u.agent.tcp_pose.p[0].numpy()
            blue_done = bool(u._blue_pressed[0])
            top = (
                u._button_top()[0].numpy()
                if not blue_done
                else u._button_top_red()[0].numpy()
            )
            delta = top + np.array([0.0, 0.0, 0.05]) - tcp
            horiz = np.linalg.norm(delta[:2])
            if horiz < 0.015:
                action[:3] = 0.0
                action[2] = -0.08
            elif np.linalg.norm(delta) > 0.03:
                action[:3] = np.clip(delta * 0.8, -0.03, 0.03) / 0.1
            else:
                action[:3] = np.clip(delta, -0.005, 0.005) / 0.1
            _, _, terminated, truncated, info = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                break
        order_success = bool(info["order_success"][0]) if "order_success" in info else False
        red_first = bool(info["red_first"][0]) if "red_first" in info else False
        blue_done = bool(info["blue_done"][0]) if "blue_done" in info else False
        successes += int(order_success)
        red_firsts += int(red_first)
        timed_out = bool(truncated[0])
        timeouts += int(timed_out)
        tag = "ORDER-OK" if order_success else ("RED-FIRST" if red_first else ("BLUE-ONLY" if blue_done else "FAIL"))
        print(
            f"ep {ep:2d} blue=({blue_xy[0]:+.3f},{blue_xy[1]:+.3f}) "
            f"red=({red_xy[0]:+.3f},{red_xy[1]:+.3f}) {tag} steps={step + 1}",
            flush=True,
        )
    print(
        f"oracle: order_success={successes}/{args.episodes} "
        f"red_first={red_firsts} timeouts={timeouts}"
    )
    env.close()


if __name__ == "__main__":
    main()
