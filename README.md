# reset-codex-button

Train a simulated Franka arm to press a big spring-loaded **RESET** button so Tibo can finally rest.

Learning project: RL from scratch — custom ManiSkill env, PPO with a small MLP, domain randomization, demo video ("Tibo has been automated").

## Stack

- [ManiSkill 3](https://maniskill.ai) / SAPIEN (CPU simulation on macOS via MoltenVK)
- PyTorch (CPU / MPS)
- Custom PPO implementation

## Setup (macOS)

```sh
brew install uv ffmpeg molten-vk
uv sync
source scripts/vulkan_env.sh
```

Reference docs: [ManiSkill macOS install](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html)

## Sanity checks

```sh
source scripts/vulkan_env.sh
uv run python scripts/record_demo.py -e PickCube-v1
uv run python scripts/benchmark.py -e PickCube-v1 --num-envs 1
uv run python scripts/play_env.py --steps 100 --fps 15
```

Videos land in `outputs/videos/`.

## The task: `ResetButton-v1`

A Franka arm presses a big spring-loaded button mounted on a pedestal. Defined in `reset_env/reset_button.py`.

| | |
|---|---|
| Scene | Panda + table; button on a pedestal, cap on a prismatic joint, spring drive (1000 N/m), 2 cm travel |
| Action | `pd_ee_delta_pos`, 4D `[dx, dy, dz, gripper]`, ±0.1 m per step |
| Observation | state vector (proprioception) + `tcp_to_button` (3) + `button_depression` (1) |
| Reward | `5*(prev_dist - dist)` + `2*(dep - prev_dep)` + `20` on success − `5e-4*||a||^2` |
| Success | depression ≥ 60% of travel (1.2 cm), then episode terminates |
| Episode | 100 steps at 20 Hz |

Validated with a scripted oracle (`scripts/play_env.py`): reaches the button, presses, and triggers success (+20 reward) in ~30 control steps.

Design traps encountered (worth remembering):

- The Panda's fixed-orientation Cartesian IK has a workspace floor (~z=0.074 m near the base) — the button must sit above it. Hence the pedestal.
- The cap must have clearance above the base at rest, or contact blocks it instead of the spring.
- With the cap too high, simply closing the gripper pinches the cap and "wins" without moving the arm — a reward hack.
- Spring stiffness must be strong enough to hold the cap's weight (sag ≪ travel) but weak enough for the arm to press.

## Measured on M1 Max (32 GB, CPU sim, state obs)

| Setup | Throughput |
|---|---|
| 1 env process, no render | ~990 steps/sec |
| 8 env processes, no render | **~6,300 steps/sec aggregate** |
| sim + offscreen render (512x512) | ~166 fps |

Implication: 2M PPO steps ≈ 5 min of raw simulation, ≈ 15-30 min wall-clock with updates.

## Roadmap

1. ~~Sanity render + throughput benchmark on Mac~~
2. ~~Custom `ResetButton-v1` env (fixed button, shaped reward)~~
3. PPO from scratch, 2x128 MLP
4. Button position randomization
5. Record early-vs-trained demo
