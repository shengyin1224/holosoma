import numpy as np
import os

def prepare_isaac_data(input_path, output_path):
    print(f"[*] 正在从 {input_path} 提取 IsaacGym 数据...")
    data = np.load(input_path)
    
    # 1. 基础信息
    num_frames = data['body_pos_w'].shape[0]
    
    # 2. Root (Pelvis) 数据 - 在 body 列表的索引 0
    # 注意：IsaacGym 使用 xyzw 顺序，MuJoCo 使用 wxyz
    root_pos = data['body_pos_w'][:, 0, :]
    root_quat_wxyz = data['body_quat_w'][:, 0, :]
    root_quat_xyzw = np.roll(root_quat_wxyz, -1, axis=-1)
    
    root_lin_vel = data['body_lin_vel_w'][:, 0, :]
    root_ang_vel = data['body_ang_vel_w'][:, 0, :]
    
    # 3. Head 数据 (使用刚才在转换脚本中新增保存的精确位置)
    if 'head_pos_w' in data:
        head_pos = data['head_pos_w']
        head_quat_wxyz = data['head_quat_w']
        head_quat_xyzw = np.roll(head_quat_wxyz, -1, axis=-1)
    else:
        # 兼容旧数据的兜底方案：使用 Torso (索引 9)
        print("[!] 警告: 未找到精确头部数据，使用 Torso 偏移作为近似")
        head_pos = data['body_pos_w'][:, 9, :] + np.array([0, 0, 0.1])
        head_quat_wxyz = data['body_quat_w'][:, 9, :]
        head_quat_xyzw = np.roll(head_quat_wxyz, -1, axis=-1)

    # 4. 打包保存
    isaac_ready_data = {
        "root_pos": root_pos,
        "root_quat": root_quat_xyzw,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "head_pos": head_pos,
        "head_quat": head_quat_xyzw,
        "fps": data['fps']
    }
    
    np.savez(output_path, **isaac_ready_data)
    print(f"[OK] 提取完成！")
    print(f"     - 帧数: {num_frames}")
    print(f"     - FPS: {data['fps'][0]}")
    print(f"     - 保存路径: {output_path}")
    print("-" * 40)
    print("前 3 帧预览 (Root Pos):")
    for i in range(min(3, num_frames)):
        print(f"  Frame {i}: {root_pos[i]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="isaac_final_data.npz")
    args = parser.parse_args()
    
    prepare_isaac_data(args.input, args.output)
