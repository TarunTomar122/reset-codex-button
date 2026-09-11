import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mani_skill.envs  # noqa: F401
import reset_env  # noqa: F401
from rl.policy import GaussianPolicy

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def flat_obs(obs):
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def to_numpy_frame(render_out):
    if hasattr(render_out, "detach"):
        render_out = render_out.detach().cpu().numpy()
    arr = np.asarray(render_out)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def label_frame(frame, text_top, text_bottom, ok=None):
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(22)
    small = load_font(18)
    draw.rectangle([(0, 0), (img.width, 40)], fill=(0, 0, 0, 150))
    draw.text((10, 8), text_top, font=font, fill=(255, 255, 255, 255))
    if text_bottom:
        color = (120, 230, 120, 255) if ok else (255, 150, 150, 255)
        draw.rectangle(
            [(0, img.height - 34), (img.width, img.height)], fill=(0, 0, 0, 150)
        )
        draw.text((10, img.height - 30), text_bottom, font=small, fill=color)
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def text_card(width, height, lines, fg=(255, 255, 255), bg=(18, 18, 22)):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    y = height // 2 - len(lines) * 24
    for text, size in lines:
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(
            ((width - (bbox[2] - bbox[0])) // 2, y),
            text,
            font=font,
            fill=fg,
        )
        y += (bbox[3] - bbox[1]) + 26
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", default="outputs/snapshots")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("-o", "--output", default="outputs/videos/training_progression.mp4")
    args = p.parse_args()

    paths = sorted(Path(args.snapshot_dir).glob("iter_*.pt"))
    if not paths:
        raise SystemExit(f"no snapshots in {args.snapshot_dir}")
    print(f"{len(paths)} snapshots", flush=True)

    env = gym.make(
        "ResetButton-v1",
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode="rgb_array",
    )
    frames = []
    frames += [
        text_card(
            512,
            512,
            [
                ("Teaching a robot arm to press RESET", 26),
                ("PPO from scratch - ManiSkill3 - M1 Max CPU", 18),
            ],
        )
    ] * int(1.2 * args.fps)

    last_iter = 0
    for path in paths:
        ckpt = torch.load(path, weights_only=False)
        policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        it = ckpt["iteration"]
        last_iter = it
        obs, _ = env.reset(seed=args.seed)
        done = False
        steps = 0
        info = {}
        episode_frames = []
        while not done and steps < args.max_steps:
            with torch.no_grad():
                mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                action = mean.clamp(-1.0, 1.0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated[0]) or bool(truncated[0])
            episode_frames.append(to_numpy_frame(env.render()))
        success = bool(info["success"][0]) if "success" in info else False
        status = "PRESSED" if success else "not pressed"
        for frame in episode_frames[:: args.stride]:
            frames.append(
                label_frame(
                    frame,
                    f"PPO iteration {it}  -  {ckpt['steps'] // 1000}k steps",
                    f"button {status}  ({steps} sim steps)",
                    ok=success,
                )
            )
        print(f"iter {it}: success={success} steps={steps}", flush=True)

    frames += [
        text_card(
            512,
            512,
            [
                ("Tibo has been automated.", 30),
                (f"PPO converged in ~{last_iter} iterations", 18),
            ],
        )
    ] * int(1.8 * args.fps)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=args.fps)
    print(f"wrote {out}: {len(frames)} frames ({len(frames) / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
