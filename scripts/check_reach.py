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
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make(
        "ResetButton-v2",
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode=None,
    )
    successes = 0
    failures = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        u = env.unwrapped
        button_xy = u.button.pose.p[0, :2].numpy()
        phase = "approach"
        action = np.zeros(4, dtype=np.float32)
        action[3] = -1.0
        steps = 0
        info = {}
        for _ in range(args.max_steps):
            tcp = u.agent.tcp_pose.p[0].numpy()
            top = u._button_top()[0].numpy()
            action[:3] = 0.0
            if phase == "approach":
                delta = top + np.array([0.0, 0.0, 0.05]) - tcp
                action[:3] = np.clip(delta, -0.005, 0.005) / 0.1
                if np.linalg.norm(delta[:2]) < 0.012 and abs(delta[2]) < 0.012:
                    phase = "press"
            elif phase == "press":
                action[2] = -0.08
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            if bool(terminated[0]) or bool(truncated[0]):
                break
        success = bool(info["success"][0]) if "success" in info else False
        successes += int(success)
        tag = "ok" if success else "FAIL"
        print(
            f"ep {ep:2d} button=({button_xy[0]:+.3f},{button_xy[1]:+.3f}) "
            f"{tag} steps={steps}",
            flush=True,
        )
        if not success:
            failures.append(button_xy)

    print(f"oracle success: {successes}/{args.episodes}")
    if failures:
        f = np.array(failures)
        print(
            f"failures x range [{f[:,0].min():+.3f},{f[:,0].max():+.3f}] "
            f"y range [{f[:,1].min():+.3f},{f[:,1].max():+.3f}]"
        )
    env.close()


if __name__ == "__main__":
    main()
