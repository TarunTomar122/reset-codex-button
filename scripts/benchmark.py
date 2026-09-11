import argparse
import os
import time

os.environ.setdefault("VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")

import gymnasium as gym
import mani_skill.envs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--env-id", default="Reach-v1")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--obs-mode", default="state")
    p.add_argument("--sim-backend", default="cpu")
    args = p.parse_args()

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        render_mode=None,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
    )
    env.reset(seed=0)

    action = env.action_space.sample()
    t0 = time.time()
    for _ in range(args.steps):
        env.step(action)
    dt = time.time() - t0

    print(
        f"env={args.env_id} num_envs={args.num_envs} obs={args.obs_mode} "
        f"steps={args.steps} time={dt:.2f}s throughput={args.steps / dt:.1f} steps/sec"
    )
    env.close()


if __name__ == "__main__":
    main()
