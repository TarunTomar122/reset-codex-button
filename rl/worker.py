import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault(
    "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
)

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

import reset_env  # noqa: F401
from rl.policy import GaussianPolicy


def flat_obs(obs):
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class EnvWorker:
    def __init__(self, env_id, seed):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env = gym.make(
            env_id,
            obs_mode="state",
            num_envs=1,
            sim_backend="cpu",
            render_mode=None,
        )
        obs, _ = self.env.reset(seed=seed)
        self.obs = flat_obs(obs)
        self.obs_dim = self.obs.shape[0]
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.policy = GaussianPolicy(self.obs_dim, self.act_dim)

    def rollout(self, state_dict, steps):
        self.policy.load_state_dict(state_dict)
        obs_buf = np.zeros((steps, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((steps, self.act_dim), dtype=np.float32)
        logp_buf = np.zeros(steps, dtype=np.float32)
        rew_buf = np.zeros(steps, dtype=np.float32)
        done_buf = np.zeros(steps, dtype=np.float32)
        ep_returns = []
        ep_successes = 0
        ep_count = 0
        ep_return = 0.0
        with torch.no_grad():
            for t in range(steps):
                obs_t = torch.from_numpy(self.obs).unsqueeze(0)
                dist = self.policy.dist(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                action_env = action.clamp(-1.0, 1.0).numpy().reshape(1, -1)
                next_obs, reward, terminated, truncated, info = self.env.step(action_env)
                success = bool(info["success"][0]) if "success" in info else False
                done = bool(terminated[0]) or bool(truncated[0])
                obs_buf[t] = self.obs
                act_buf[t] = action.numpy().reshape(-1)
                logp_buf[t] = float(logp[0])
                rew_buf[t] = float(reward[0])
                done_buf[t] = float(done)
                ep_return += float(reward[0])
                if done:
                    ep_returns.append(ep_return)
                    ep_return = 0.0
                    ep_successes += int(success)
                    ep_count += 1
                    next_obs, _ = self.env.reset()
                self.obs = flat_obs(next_obs)
        return {
            "obs": obs_buf,
            "actions": act_buf,
            "logp": logp_buf,
            "rewards": rew_buf,
            "dones": done_buf,
            "final_obs": self.obs.copy(),
            "ep_returns": ep_returns,
            "ep_successes": ep_successes,
            "ep_count": ep_count,
        }


def worker_main(conn, env_id, seed):
    worker = EnvWorker(env_id, seed)
    while True:
        msg = conn.recv()
        if msg["cmd"] == "rollout":
            conn.send(worker.rollout(msg["state_dict"], msg["steps"]))
        elif msg["cmd"] == "close":
            break
