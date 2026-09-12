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
from rl.oracle import flat_obs, oracle_action
from rl.policy import GaussianPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--env-id", default="ResetButton-v13")
    p.add_argument("--demos", default="outputs/demos/order_demos.npz")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--qpos-noise", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("-o", "--output", default="outputs/demos/order_demos_dagger1.npz")
    args = p.parse_args()

    ckpt = torch.load(args.policy, weights_only=False)
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
    new_obs = []
    new_act = []
    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        info = {}
        for _ in range(args.max_steps):
            new_obs.append(flat_obs(obs))
            new_act.append(oracle_action(u).copy())
            with torch.no_grad():
                mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                action = mean.numpy().reshape(-1)
                action = action + np.random.normal(0, args.noise, size=action.shape)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
            obs, _, terminated, truncated, info = env.step(action.reshape(1, -1))
            if bool(terminated[0]) or bool(truncated[0]):
                break
        successes += int(bool(info.get("order_success", info.get("success"))[0]))

    old = np.load(args.demos)
    obs_all = np.concatenate([old["obs"], np.asarray(new_obs, dtype=np.float32)])
    act_all = np.concatenate([old["actions"], np.asarray(new_act, dtype=np.float32)])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=obs_all, actions=act_all)
    print(
        f"policy episodes={args.episodes} success={successes}/{args.episodes} "
        f"new_transitions={len(new_act)} total_transitions={len(act_all)} -> {out}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
