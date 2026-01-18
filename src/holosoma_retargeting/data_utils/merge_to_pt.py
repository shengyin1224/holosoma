import torch
import pickle
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_joints_path", type=str, required=True)
    parser.add_argument("--object_movement_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # 加载机器人关节数据 (来自 Blender 提取的 body_positions)
    print(f"加载机器人关节数据: {args.robot_joints_path}")
    with open(args.robot_joints_path, 'rb') as f:
        robot_data = pickle.load(f)
    body_positions = robot_data['body_positions']

    # 加载物体运动数据 (来自 FK 提取的 object_movement)
    print(f"加载物体运动数据: {args.object_movement_path}")
    with open(args.object_movement_path, 'rb') as f:
        object_data = pickle.load(f)

    obj_pos = object_data['body_pos_w'].squeeze(1)  # (T, 3)
    obj_quat = object_data['body_quat_w'].squeeze(1)  # (T, 4)

    # 合并物体的四元数和位置: [qw, qx, qy, qz, x, y, z]
    object_data_flat = np.concatenate([obj_quat, obj_pos], axis=1)  # (T, 7)

    save_data = {
        "human_joints": body_positions,
        "object_poses": object_data_flat,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(save_data, args.output_path)
    print(f"成功合并并保存到: {args.output_path}")

if __name__ == "__main__":
    main()
