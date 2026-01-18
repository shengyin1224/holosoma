from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if "body_positions" not in data:
        raise KeyError("Missing key 'body_positions' in npz file.")
    if "movement" not in data:
        raise KeyError("Missing key 'movement' in npz file.")

    body_positions = data["body_positions"]
    movement = data["movement"]

    if body_positions.ndim != 3 or body_positions.shape[1:] != (15, 3):
        raise ValueError(
            f"body_positions must have shape (T, 15, 3), got {body_positions.shape}."
        )
    if movement.ndim != 2 or movement.shape[1] != 7:
        raise ValueError(f"movement must have shape (T, 7), got {movement.shape}.")
    if movement.shape[0] != body_positions.shape[0]:
        raise ValueError(
            "movement and body_positions must have the same frame count; "
            f"got {movement.shape[0]} vs {body_positions.shape[0]}."
        )

    return body_positions, movement


def build_interMimic_tensor(body_positions: np.ndarray, movement: np.ndarray) -> torch.Tensor:
    num_frames = body_positions.shape[0]
    human_joints_flat = body_positions.reshape(num_frames, -1)

    # movement order: [qw, qx, qy, qz, x, y, z]
    # stored order: [x, y, z, qx, qy, qz, qw] for load_intermimic_data compatibility
    object_data_flat = movement[:, [4, 5, 6, 1, 2, 3, 0]]

    final_data = np.concatenate([human_joints_flat, object_data_flat], axis=1)
    expected_dim = 15 * 3 + 7
    if final_data.shape[1] != expected_dim:
        raise ValueError(
            f"Final data should have {expected_dim} dims, got {final_data.shape[1]}."
        )

    return torch.from_numpy(final_data).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate InterMimic .pt from npz (body_positions + movement)."
    )
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=base_dir / "move_chair.npz",
        help="Path to input npz file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=base_dir / "move_chair_from_npz.pt",
        help="Path to output .pt file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading npz: {args.npz_path}")
    body_positions, movement = load_npz(args.npz_path)
    print(f"body_positions shape: {body_positions.shape}")
    print(f"movement shape: {movement.shape}")

    final_tensor = build_interMimic_tensor(body_positions, movement)
    print(f"final tensor shape: {final_tensor.shape}")
    print(f"final tensor dtype: {final_tensor.dtype}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_tensor, args.output_path)
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
