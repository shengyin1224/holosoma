from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def rotation_x_matrix(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cos_r = np.cos(radians)
    sin_r = np.sin(radians)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_r, -sin_r],
            [0.0, sin_r, cos_r],
        ]
    )
    return transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate an OBJ by roll (x-axis) and save.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/models/low_chair/low_chair_sample.obj",
        help="Input OBJ path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/models/low_chair/low_chair_sample_roll90.obj",
        help="Output OBJ path.",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=90.0,
        help="Roll angle in degrees (rotation about x-axis).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh = trimesh.load(args.input, force="mesh")
    mesh.apply_transform(rotation_x_matrix(args.degrees))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(f"Saved rotated OBJ to: {args.output}")


if __name__ == "__main__":
    main()
