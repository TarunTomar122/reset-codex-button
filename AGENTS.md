# AGENTS.md

## Communication style (ADHD-friendly, from ayghri/i-have-adhd)

1. Lead with the next action.
2. Number multi-step tasks.
3. End with one concrete next step.
4. Suppress tangents.
5. Restate state every turn.
6. Specific time estimates (minutes, not "a bit").
7. Make wins visible.
8. Matter-of-fact errors.
9. Cap lists to 5 items.
10. No preamble. No recap. No closers.

## Project notes

- macOS CPU-only ManiSkill/SAPIEN setup. Always `source scripts/vulkan_env.sh` before running Python.
- Validate envs with `scripts/check_reach.py` (oracle) before training.
- Training: `train.py` (PPO, 8 workers). Run with `uv run python` or `.venv/bin/python`.
- Keep responses short. No filler.
