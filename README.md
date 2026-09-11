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
uv run python scripts/record_demo.py -e Reach-v1
uv run python scripts/benchmark.py -e Reach-v1 --num-envs 4
```

Videos land in `outputs/videos/`.

## Measured on M1 Max (32 GB, CPU sim, state obs)

| Setup | Throughput |
|---|---|
| 1 env process, no render | ~990 steps/sec |
| 8 env processes, no render | **~6,300 steps/sec aggregate** |
| sim + offscreen render (512x512) | ~166 fps |

Implication: 2M PPO steps ≈ 5 min of raw simulation, ≈ 15-30 min wall-clock with updates.

## Roadmap

1. Sanity render + throughput benchmark on Mac
2. Custom `ResetButton-v1` env (fixed button, shaped reward)
3. PPO from scratch, 2x128 MLP
4. Button position randomization
5. Record early-vs-trained demo
