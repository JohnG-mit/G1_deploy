import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def extract_index(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return -1
    return int(matches[-1])


def collect_images(input_dir: Path):
    image_paths = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    image_paths.sort(key=lambda p: (extract_index(p), p.name))
    return image_paths


def _load_rgb_frames(image_paths):
    first = Image.open(image_paths[0]).convert("RGB")
    width, height = first.size
    frames = [np.asarray(first, dtype=np.float32) / 255.0]

    for path in image_paths[1:]:
        img = Image.open(path).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), resample=Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)

    return np.stack(frames, axis=0)


def sample_indices(total, stride):
    if total <= 0:
        return []
    if stride <= 1:
        return list(range(total))

    picked = list(range(0, total, stride))
    if picked[-1] != total - 1:
        picked.append(total - 1)
    return picked


def overlay_by_index(image_paths, output_path: Path, min_alpha=0.08, bg_threshold=0.05, stride=5):
    if not image_paths:
        raise ValueError("No images found to overlay.")

    all_frames = _load_rgb_frames(image_paths)
    all_total = all_frames.shape[0]
    picked_idx = sample_indices(all_total, int(stride))
    frames = all_frames[picked_idx]
    total = frames.shape[0]

    # Use the temporal median as static background estimate.
    background = np.median(frames, axis=0)
    out = background.copy()

    if total == 1:
        alphas = np.array([1.0], dtype=np.float32)
    else:
        alphas = np.linspace(float(min_alpha), 1.0, total, dtype=np.float32)

    for i in range(total):
        frame = frames[i]
        alpha = float(alphas[i])

        # Foreground mask from background subtraction.
        diff = np.max(np.abs(frame - background), axis=2)
        mask = diff > float(bg_threshold)
        if not np.any(mask):
            continue

        out_mask = out[mask]
        frame_mask = frame[mask]
        out[mask] = (1.0 - alpha) * out_mask + alpha * frame_mask

    # Force the last frame foreground fully clear.
    final_frame = frames[-1]
    final_diff = np.max(np.abs(final_frame - background), axis=2)
    final_mask = final_diff > float(bg_threshold)
    out[final_mask] = final_frame[final_mask]

    out_img = Image.fromarray(np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)


def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "imgs"
    default_output = script_dir / "overlay_result.png"

    parser = argparse.ArgumentParser(
        description="Overlay images in index order with alpha increasing to 1 for the last frame."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Input image folder (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output image path (default: {default_output})",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.08,
        help="Minimum alpha for the first frame (default: 0.08).",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.05,
        help="Background subtraction threshold in [0,1] (default: 0.05).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Use one frame every N frames for overlay; always keep the last frame (default: 5).",
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")

    if not (0.0 <= args.min_alpha <= 1.0):
        raise ValueError("--min-alpha must be within [0, 1].")
    if not (0.0 <= args.bg_threshold <= 1.0):
        raise ValueError("--bg-threshold must be within [0, 1].")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")

    image_paths = collect_images(args.input_dir)
    picked_idx = sample_indices(len(image_paths), args.stride)
    overlay_by_index(
        image_paths,
        args.output,
        min_alpha=args.min_alpha,
        bg_threshold=args.bg_threshold,
        stride=args.stride,
    )

    print(f"[Overlay] Input images: {len(image_paths)}")
    print(f"[Overlay] Used indices: {picked_idx}")
    print(f"[Overlay] Output saved to: {args.output}")


if __name__ == "__main__":
    main()
