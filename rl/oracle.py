import numpy as np


def oracle_action(u):
    tcp = u.agent.tcp_pose.p[0].numpy()
    blue_done = bool(u._blue_pressed[0]) if hasattr(u, "_blue_pressed") else False
    top = (
        u._button_top_red()[0].numpy()
        if blue_done
        else u._button_top()[0].numpy()
    )
    delta = top + np.array([0.0, 0.0, 0.05]) - tcp
    horiz = np.linalg.norm(delta[:2])
    action = np.zeros(4, dtype=np.float32)
    action[3] = -1.0
    if horiz < 0.015:
        action[2] = -0.08
    elif np.linalg.norm(delta) > 0.03:
        action[:3] = np.clip(delta * 0.8, -0.03, 0.03) / 0.1
    else:
        action[:3] = np.clip(delta, -0.005, 0.005) / 0.1
    return action


def flat_obs(obs):
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32).reshape(-1)
