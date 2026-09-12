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


def label_frame(frame, top, bottom, color):
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(22)
    small = load_font(18)
    draw.rectangle([(0, 0), (img.width, 40)], fill=(0, 0, 0, 150))
    draw.text((10, 8), top, font=font, fill=(255, 255, 255, 255))
    draw.rectangle([(0, img.height - 34), (img.width, img.height)], fill=(0, 0, 0, 150))
    draw.text((10, img.height - 30), bottom, font=small, fill=color)
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def text_card(width, height, lines, fg=(255, 255, 255), bg=(18, 18, 22)):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    y = height // 2 - len(lines) * 24
    for text, size in lines:
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) // 2, y), text, font=font, fill=fg)
        y += (bbox[3] - bbox[1]) + 26
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/seed_0/checkpoints/latest.pt")
    p.add_argument("--env-id", default="ResetButton-v2")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--hold", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=25)
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--qpos-noise", type=float, default=0.02)
    p.add_argument("--order", action="store_true")
    p.add_argument("-o", "--output", default="outputs/videos/v2_policy.mp4")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"loaded iteration {ckpt['iteration']} ({ckpt['steps']} steps)", flush=True)

    env = gym.make(
        args.env_id,
        obs_mode="state",
        num_envs=1,
        sim_backend="cpu",
        render_mode="rgb_array",
        robot_init_qpos_noise=args.qpos_noise,
    )
    frames = [
        text_card(
            512,
            512,
            [
                ("Trained policy - random button positions", 24),
                ("PPO from scratch - 431k steps - 4 minutes on Mac", 17),
            ],
        )
    ] * int(1.5 * args.fps)
    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        xy = u.button.pose.p[0, :2].numpy()
        red_xy = (
            u.button_red.pose.p[0, :2].numpy() if args.order else None
        )
        done = False
        steps = 0
        info = {}
        ep_frames = []
        while not done:
            with torch.no_grad():
                mean, _ = policy(torch.from_numpy(flat_obs(obs)).unsqueeze(0))
                action = mean.clamp(-1.0, 1.0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated[0]) or bool(truncated[0])
            stage = ""
            if args.order:
                stage = "stage: to RED" if bool(u._blue_pressed[0]) else "stage: to BLUE"
            ep_frames.append((to_numpy_frame(env.render()), stage))
        if args.order:
            order_ok = bool(info.get("order_success", torch.tensor([False]))[0])
            red_first = bool(info.get("red_first", torch.tensor([False]))[0])
            blue_done = bool(info.get("blue_done", torch.tensor([False]))[0])
            if order_ok:
                tag, success = "ORDER OK (blue then red)", True
            elif red_first:
                tag, success = "WRONG ORDER (red first)", False
            elif blue_done and bool(truncated[0]):
                tag, success = "blue only, timeout", False
            else:
                tag, success = "no press", False
        else:
            success = bool(info["success"][0]) if "success" in info else False
            tag = "PRESSED" if success else "failed"
        successes += int(success)
        color = (120, 230, 120, 255) if success else (255, 150, 150, 255)
        stride = max(1, int(np.ceil(len(ep_frames) / args.max_frames)))
        speed_note = f"  ({stride}x)" if stride > 1 else ""
        labeled = None
        for frame, stage in ep_frames[::stride]:
            if args.order:
                top = (
                    f"blue({xy[0]:+.2f},{xy[1]:+.2f})  "
                    f"red({red_xy[0]:+.2f},{red_xy[1]:+.2f})  ep {ep + 1}"
                )
                bottom = f"{stage}   {tag}  in {steps} steps{speed_note}"
            else:
                top = f"button at ({xy[0]:+.2f}, {xy[1]:+.2f})   episode {ep + 1}"
                bottom = f"{tag}  in {steps} sim steps{speed_note}"
            labeled = label_frame(frame, top, bottom, color)
            frames.extend([labeled] * args.repeat)
        frames.extend([labeled] * args.hold)
        print(
            f"ep {ep}: blue=({xy[0]:+.3f},{xy[1]:+.3f}) "
            + (f"red=({red_xy[0]:+.3f},{red_xy[1]:+.3f}) " if args.order else "")
            + f"{tag} steps={steps}",
            flush=True,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=args.fps)
    print(
        f"success {successes}/{args.episodes} wrote {out}: {len(frames)} frames "
        f"({len(frames) / args.fps:.1f}s)"
    )
    env.close()


if __name__ == "__main__":
    main()
