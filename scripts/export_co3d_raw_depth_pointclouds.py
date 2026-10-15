#!/usr/bin/env python3
"""Export per-frame RGB point clouds from a CO3Dv2 sequence's raw depth maps."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_ROOT = Path("/mnt/data/wangzhixing/data/pi3_data/co3d_v2_mini")
DEFAULT_SEQUENCE = "396_49631_97799"
DEFAULT_CATEGORY = "microwave"


def camera_intrinsics(anno: dict, image_height: int, image_width: int) -> np.ndarray:
    """Match opencv_from_cameras_projection() in datasets/co3dv2_dataset.py."""
    scale = min(image_width, image_height) / 2.0
    center = np.array([image_width, image_height], dtype=np.float64) / 2.0
    focal = np.asarray(anno["focal_length"], dtype=np.float64) * scale
    principal = center - np.asarray(anno["principal_point"], dtype=np.float64) * scale
    return np.array(
        [[focal[0], 0.0, principal[0]],
         [0.0, focal[1], principal[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def decode_depth(path: Path) -> np.ndarray:
    # CO3Dv2 geometric depth PNG stores float16 bits in uint16 pixels; this is
    # the same decoding used by CO3DV2Dataset._get_views().
    encoded = np.asarray(Image.open(path), dtype=np.uint16)
    depth = encoded.view(np.float16).astype(np.float32)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def export_frame(
    root: Path,
    anno: dict,
    output_path: Path,
    stride: int,
    use_object_mask: bool,
    min_depth: float,
    max_depth: float,
) -> int:
    relative_image = Path(anno["filepath"])
    image_path = root / relative_image
    depth_path = root / relative_image.as_posix().replace("/images/", "/depths/")
    depth_path = Path(f"{depth_path}.geometric.png")

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    depth = decode_depth(depth_path)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"Image/depth size mismatch for {image_path}: {rgb.shape[:2]} vs {depth.shape}"
        )

    valid = np.isfinite(depth) & (depth > min_depth)
    if max_depth > 0:
        valid &= depth <= max_depth

    if use_object_mask:
        mask_path = Path(
            str(image_path).replace("/images/", "/masks/")
        ).with_suffix(".png")
        if mask_path.exists():
            object_mask = np.asarray(Image.open(mask_path).convert("L"))
            if object_mask.shape != depth.shape:
                raise ValueError(f"Image/mask size mismatch for {image_path}")
            valid &= object_mask.astype(np.float32) / 255.0 > 0.1
        else:
            raise FileNotFoundError(f"Object mask not found: {mask_path}")

    if stride > 1:
        sampled = np.zeros_like(valid)
        sampled[::stride, ::stride] = True
        valid &= sampled

    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    k = camera_intrinsics(anno, *depth.shape)
    xyz = np.column_stack((
        (u - k[0, 2]) * z / k[0, 0],
        (v - k[1, 2]) * z / k[1, 1],
        z,
    )).astype(np.float32)
    colors = rgb[v, u]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply(output_path, xyz, colors)
    return len(xyz)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/wangzhixing/tmp/co3d_depth_rgb/microwave-396_49631_97799"),
    )
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth pixel in each axis.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 exports every annotated frame.")
    parser.add_argument("--include-background", action="store_true", help="Do not apply the CO3D object mask.")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=0.0, help="0 disables the upper depth limit.")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    annotation_file = args.data_root / f"{args.category}_{args.split}.jgz"
    with gzip.open(annotation_file, "rt", encoding="utf-8") as stream:
        annotations = json.load(stream)
    if args.sequence not in annotations:
        raise KeyError(
            f"Sequence {args.sequence!r} is not in {annotation_file}; "
            f"available splits should be checked (found {len(annotations)} sequences)."
        )

    frames = annotations[args.sequence]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    total_points = 0
    for index, anno in enumerate(tqdm(frames, desc="Exporting CO3Dv2 frames")):
        frame_name = Path(anno["filepath"]).stem
        output_path = args.output_dir / f"{frame_name}_depth_rgb.ply"
        total_points += export_frame(
            root=args.data_root,
            anno=anno,
            output_path=output_path,
            stride=args.stride,
            use_object_mask=not args.include_background,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

    print(f"Exported {len(frames)} frames and {total_points:,} points to {args.output_dir}")
    print("Coordinates: per-frame camera coordinates; colors: original RGB uint8.")


if __name__ == "__main__":
    main()