import argparse
import csv
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ.setdefault(
    "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
)

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

import reset_env  # noqa: F401
from rl.policy import GaussianPolicy, ValueNet
from rl.worker import worker_main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="ResetButton-v1")
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--max-mem-gb", type=float, default=20.0)
    p.add_argument("--qpos-noise", type=float, default=0.02)
    p.add_argument("--stop-at-success", type=float, default=1.01)
    p.add_argument("--snapshot-iters", default="")
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def flat_obs(obs):
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def total_rss_gb(pids):
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
        capture_output=True,
        text=True,
    ).stdout
    kb = sum(int(line.strip()) for line in out.splitlines() if line.strip())
    return kb / (1024 * 1024)


def save_checkpoint(path, policy, value, iteration, steps, args, obs_dim, act_dim):
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "iteration": iteration,
            "steps": steps,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "args": vars(args),
        },
        path,
    )


def evaluate(env, policy, episodes):
    policy.eval()
    successes = 0
    red_firsts = 0
    blue_onlys = 0
    returns = []
    lengths = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        info = {}
        truncated = [False]
        while not done:
            with torch.no_grad():
                mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                action = mean.clamp(-1.0, 1.0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward[0])
            steps += 1
            done = bool(terminated[0]) or bool(truncated[0])
        if "order_success" in info:
            successes += int(bool(info["order_success"][0]))
            red_firsts += int(bool(info["red_first"][0]))
            blue_onlys += int(
                bool(truncated[0])
                and bool(info["blue_done"][0])
                and not bool(info["red_now"][0])
            )
        else:
            successes += int(bool(info["success"][0]))
        returns.append(ep_return)
        lengths.append(steps)
    policy.train()
    return (
        successes / episodes,
        float(np.mean(returns)),
        float(np.mean(lengths)),
        red_firsts / episodes,
        blue_onlys / episodes,
    )


def compute_gae(batches, value, args):
    obs_list, act_list, logp_list, adv_list, ret_list = [], [], [], [], []
    for b in batches:
        obs = torch.from_numpy(b["obs"])
        final_obs = torch.from_numpy(b["final_obs"]).unsqueeze(0)
        with torch.no_grad():
            values = value(obs)
            final_value = value(final_obs)[0]
        v_next = torch.cat([values[1:], final_value.unsqueeze(0)])
        nonterminal = 1.0 - torch.from_numpy(b["dones"])
        rewards = torch.from_numpy(b["rewards"])
        delta = rewards + args.gamma * v_next * nonterminal - values
        adv = torch.zeros_like(delta)
        gae = 0.0
        for t in reversed(range(len(delta))):
            gae = float(delta[t] + args.gamma * args.lam * nonterminal[t] * gae)
            adv[t] = gae
        obs_list.append(obs)
        act_list.append(torch.from_numpy(b["actions"]))
        logp_list.append(torch.from_numpy(b["logp"]))
        adv_list.append(adv)
        ret_list.append(adv + values)
    return (
        torch.cat(obs_list),
        torch.cat(act_list),
        torch.cat(logp_list),
        torch.cat(adv_list),
        torch.cat(ret_list),
    )


def main():
    args = parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    snapshot_iters = {
        int(x) for x in args.snapshot_iters.split(",") if x.strip().isdigit()
    }

    eval_env = gym.make(
        args.env_id,
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode=None,
        robot_init_qpos_noise=args.qpos_noise,
    )
    obs, _ = eval_env.reset(seed=args.seed)
    obs_dim = flat_obs(obs).shape[0]
    act_dim = eval_env.action_space.shape[0]
    print(f"obs_dim={obs_dim} act_dim={act_dim} workers={args.num_workers}", flush=True)

    policy = GaussianPolicy(obs_dim, act_dim)
    value = ValueNet(obs_dim)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, weights_only=False)
        policy.load_state_dict(init["policy"])
        print(f"initialized policy from {args.init_checkpoint}", flush=True)
    params = list(policy.parameters()) + list(value.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, eps=1e-5)

    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for i in range(args.num_workers):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=worker_main,
            args=(child_conn, args.env_id, args.seed * 1000 + i, args.qpos_noise),
            daemon=True,
        )
        proc.start()
        conns.append(parent_conn)
        procs.append(proc)
        child_conn.close()
    pids = [p.pid for p in procs]

    steps_per_iter = args.rollout_steps * args.num_workers
    iterations = max(1, args.total_steps // steps_per_iter)
    print(f"iterations={iterations} steps_per_iter={steps_per_iter}", flush=True)

    log_file = open(out / "train_log.csv", "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(
        [
            "iter",
            "steps",
            "mean_return",
            "worker_success",
            "worker_red_first",
            "worker_blue_only",
            "eval_success",
            "eval_red_first",
            "eval_blue_only",
            "eval_return",
            "eval_length",
            "policy_loss",
            "value_loss",
            "entropy",
            "clip_frac",
            "rss_gb",
            "sps",
        ]
    )
    log_file.flush()

    total_steps = 0
    best_success = -1.0
    try:
        for it in range(1, iterations + 1):
            t_iter = time.time()
            weights = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            for conn in conns:
                conn.send(
                    {
                        "cmd": "rollout",
                        "state_dict": weights,
                        "steps": args.rollout_steps,
                    }
                )
            batches = [conn.recv() for conn in conns]
            total_steps += steps_per_iter

            obs_t, act_t, logp_t, adv_t, ret_t = compute_gae(batches, value, args)
            n = obs_t.shape[0]
            adv_mean, adv_std = adv_t.mean(), adv_t.std(unbiased=False)

            policy_losses, value_losses, entropies, clip_fracs = [], [], [], []
            for _ in range(args.epochs):
                perm = torch.randperm(n)
                for start in range(0, n, args.minibatch):
                    mb = perm[start : start + args.minibatch]
                    mean, log_std = policy(obs_t[mb])
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    logp = dist.log_prob(act_t[mb]).sum(-1)
                    ratio = (logp - logp_t[mb]).exp()
                    adv_mb = (adv_t[mb] - adv_mean) / (adv_std + 1e-8)
                    pg_loss = -torch.min(
                        ratio * adv_mb,
                        ratio.clamp(1 - args.clip, 1 + args.clip) * adv_mb,
                    ).mean()
                    v_loss = 0.5 * (value(obs_t[mb]) - ret_t[mb]).pow(2).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 0.5)
                    optimizer.step()
                    with torch.no_grad():
                        policy_losses.append(pg_loss.item())
                        value_losses.append(v_loss.item())
                        entropies.append(entropy.item())
                        clip_fracs.append(
                            ((ratio - 1.0).abs() > args.clip).float().mean().item()
                        )

            ep_returns = [r for b in batches for r in b["ep_returns"]]
            ep_count = sum(b["ep_count"] for b in batches)
            ep_successes = sum(b["ep_successes"] for b in batches)
            ep_red_first = sum(b.get("ep_red_first", 0) for b in batches)
            ep_blue_only = sum(b.get("ep_blue_only", 0) for b in batches)
            mean_return = float(np.mean(ep_returns)) if ep_returns else float("nan")
            worker_success = ep_successes / ep_count if ep_count else float("nan")
            worker_red_first = ep_red_first / ep_count if ep_count else float("nan")
            worker_blue_only = ep_blue_only / ep_count if ep_count else float("nan")

            eval_success = eval_return = eval_length = float("nan")
            eval_red_first = eval_blue_only = float("nan")
            stop_now = False
            if it % args.eval_every == 0 or it == 1:
                (
                    eval_success,
                    eval_return,
                    eval_length,
                    eval_red_first,
                    eval_blue_only,
                ) = evaluate(eval_env, policy, args.eval_episodes)
                if eval_success > best_success:
                    best_success = eval_success
                    save_checkpoint(
                        out / "checkpoints" / "best.pt",
                        policy,
                        value,
                        it,
                        total_steps,
                        args,
                        obs_dim,
                        act_dim,
                    )
                if eval_success >= args.stop_at_success:
                    print(
                        f"reached eval success {eval_success:.2f}, stopping early",
                        flush=True,
                    )
                    stop_now = True

            rss = total_rss_gb(pids + [os.getpid()])
            sps = steps_per_iter / (time.time() - t_iter)
            save_checkpoint(
                out / "checkpoints" / "latest.pt",
                policy,
                value,
                it,
                total_steps,
                args,
                obs_dim,
                act_dim,
            )
            if it in snapshot_iters:
                save_checkpoint(
                    out / "snapshots" / f"iter_{it:04d}.pt",
                    policy,
                    value,
                    it,
                    total_steps,
                    args,
                    obs_dim,
                    act_dim,
                )
            logger.writerow(
                [
                    it,
                    total_steps,
                    f"{mean_return:.2f}",
                    f"{worker_success:.3f}",
                    f"{worker_red_first:.3f}",
                    f"{worker_blue_only:.3f}",
                    f"{eval_success:.3f}",
                    f"{eval_red_first:.3f}",
                    f"{eval_blue_only:.3f}",
                    f"{eval_return:.2f}",
                    f"{eval_length:.1f}",
                    f"{np.mean(policy_losses):.4f}",
                    f"{np.mean(value_losses):.4f}",
                    f"{np.mean(entropies):.3f}",
                    f"{np.mean(clip_fracs):.3f}",
                    f"{rss:.2f}",
                    f"{sps:.0f}",
                ]
            )
            log_file.flush()

            if it % 10 == 0 or it == 1 or it % args.eval_every == 0:
                print(
                    f"it={it:5d} steps={total_steps:8d} "
                    f"return={mean_return:7.2f} worker_succ={worker_success:5.2f} "
                    f"red_first={worker_red_first:5.2f} eval_succ={eval_success:5.2f} "
                    f"ent={np.mean(entropies):6.3f} "
                    f"vf={np.mean(value_losses):7.3f} rss={rss:4.1f}GB sps={sps:5.0f}",
                    flush=True,
                )

            if rss > args.max_mem_gb:
                print(
                    f"memory {rss:.1f}GB exceeded cap {args.max_mem_gb}GB, stopping",
                    flush=True,
                )
                break
            if stop_now:
                break
    except KeyboardInterrupt:
        print("interrupted, saving checkpoint", flush=True)
        save_checkpoint(
            out / "checkpoints" / "latest.pt",
            policy,
            value,
            0,
            total_steps,
            args,
            obs_dim,
            act_dim,
        )
    finally:
        for conn in conns:
            try:
                conn.send({"cmd": "close"})
            except (BrokenPipeError, OSError):
                pass
        for proc in procs:
            proc.join(timeout=10)
        eval_env.close()
        log_file.close()
        print(f"done after {total_steps} steps", flush=True)


if __name__ == "__main__":
    main()
