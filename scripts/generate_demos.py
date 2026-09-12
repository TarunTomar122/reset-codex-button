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
from rl.oracle import flat_obs, oracle_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="ResetButton-v13")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qpos-noise", type=float, default=0.3)
    p.add_argument("-o", "--output", default="outputs/demos/order_demos.npz")
    args = p.parse_args()

    env = gym.make(
        args.env_id,
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode=None,
        robot_init_qpos_noise=args.qpos_noise,
    )
    obs_list = []
    act_list = []
    successes = 0
    lengths = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        info = {}
        ep_obs = []
        ep_act = []
        for _ in range(args.max_steps):
            action = oracle_action(u)
            ep_obs.append(flat_obs(obs))
            ep_act.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                break
        success = bool(info.get("order_success", info.get("success"))[0])
        successes += int(success)
        if success:
            obs_list.extend(ep_obs)
            act_list.extend(ep_act)
            lengths.append(len(ep_act))
    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=obs_arr, actions=act_arr)
    print(
        f"episodes={args.episodes} success={successes}/{args.episodes} "
        f"kept_transitions={len(act_arr)} mean_len={np.mean(lengths):.1f} "
        f"obs_dim={obs_arr.shape[1]} act_dim={act_arr.shape[1]} -> {out}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
