import torch
import pickle
import numpy as np
import os
import json
import sys
import subprocess
import argparse  # 新增：用于解析命令行参数
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ----------------------------- 环境自动切换配置 -----------------------------
MANIPTRANS_PYTHON = "/home/ubuntu/miniconda3/envs/maniptrans/bin/python3"

def run_in_maniptrans():
    """检测并自动切换到 maniptrans 环境运行"""
    if sys.executable != MANIPTRANS_PYTHON:
        if not os.path.exists(MANIPTRANS_PYTHON):
            print(f"错误: 找不到 maniptrans 环境路径: {MANIPTRANS_PYTHON}")
            sys.exit(1)
            
        print(f"[*] 当前环境为 {sys.executable}")
        print(f"[*] 正在自动切换到 maniptrans 环境执行脚本...")
        
        try:
            # sys.argv 包含了所有命令行参数，会被原样传给子进程
            result = subprocess.run([MANIPTRANS_PYTHON] + sys.argv)
            sys.exit(result.returncode)
        except Exception as e:
            print(f"切换环境执行失败: {e}")
            sys.exit(1)

# ----------------------------- ManiTrans Paths -----------------------------
MANIPTRANS_DIR = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS"
DATASET_PATH = os.path.join(MANIPTRANS_DIR, "main/dataset")
if DATASET_PATH not in sys.path:
    sys.path.insert(0, DATASET_PATH)

# ----------------------------- Configuration -----------------------------
RAW_DATA_PATH = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data/humoto/carry_side_table_with_both_hands_walk_around-536/carry_side_table_with_both_hands_walk_around-536.pkl"
MODEL_JSON_PATH = os.path.join(DATASET_PATH, "human_model/human_model_mixamo_bone_zup.json")
OUTPUT_DIR = "/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/carry_side_table"
TASK_NAME = "carry_side_table_536"

# ================= 运动学映射 =================
JOINT_MAPPING = {
    "pelvis": "mixamorig:Hips",
    "left_hip": "mixamorig:LeftUpLeg",
    "right_hip": "mixamorig:RightUpLeg",
    "left_knee": "mixamorig:LeftLeg",
    "right_knee": "mixamorig:RightLeg",
    "left_shoulder": "mixamorig:LeftArm",      
    "right_shoulder": "mixamorig:RightArm",    
    "left_elbow": "mixamorig:LeftForeArm",    
    "right_elbow": "mixamorig:RightForeArm",  
    "left_ankle": "mixamorig:LeftFoot",
    "right_ankle": "mixamorig:RightFoot",
    "left_foot": "mixamorig:LeftToeBase",
    "right_foot": "mixamorig:RightToeBase",
    "left_wrist": "mixamorig:LeftHand",
    "right_wrist": "mixamorig:RightHand"
}

HOLOSOMA_JOINTS = [
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "left_wrist", "right_wrist"
]

def main():
    # 1. 环境切换检测
    run_in_maniptrans()

    # 2. 解析参数 (在环境切换后解析)
    parser = argparse.ArgumentParser(description="Convert Humoto data to Holosoma .pt format")
    parser.add_argument("--obj_name", type=str, default=None, help="Name of the object to extract from pkl")
    parser.add_argument("--raw_data_path", type=str, default=RAW_DATA_PATH, help="Path to input .pkl file")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save output files")
    parser.add_argument("--task_name", type=str, default=TASK_NAME, help="Task name for output files")
    args = parser.parse_args()

    # 3. 延迟导入平台库
    from utils.load_humoto import load_one_humoto_sequence
    from human_model.human_model import HumanModelDifferentiable
    from utils.rotation_helper import quaternion_to_matrix_torch

    device = "cpu" 
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"加载数据 (Z-up): {args.raw_data_path}")
    raw_data = load_one_humoto_sequence(
        data_path=os.path.dirname(args.raw_data_path), 
        include_text=False,
        y_up=False, 
        object_model=False,
        pose_params=True
    )
    
    print(f"初始化 HumanModel: {MODEL_JSON_PATH}")
    human_model = HumanModelDifferentiable(
        character_data_path=MODEL_JSON_PATH, 
        device=device
    )
    
    # 4. 骨骼处理 (FK)
    pose_params_dict = raw_data['armature_pose_params']
    object_pose_params_dict = raw_data['object_pose_params']
    
    num_frames_orig = next(iter(pose_params_dict.values())).shape[0]
    print(f"原始帧数: {num_frames_orig}，正在插值到 60fps...")
    
    # --- 插值逻辑 ---
    target_fps = 60.0
    orig_fps = 30.0 # 强制假设原始数据是 30fps
    sampling_step = orig_fps / target_fps # 0.5
    
    num_frames = int((num_frames_orig - 1) / sampling_step) + 1
    
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation as R, Slerp
    
    def interpolate_7d(data, num_frames):
        # data: (T, 7) [qw, qx, qy, qz, x, y, z]
        T = data.shape[0]
        x_orig = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, num_frames)
        
        # 插值位置 (x, y, z)
        pos_interp = interp1d(x_orig, data[:, 4:], axis=0, kind='linear')(x_new)
        
        # 插值旋转 (qw, qx, qy, qz)
        # Slerp 需要 [qx, qy, qz, qw] 格式
        rots = R.from_quat(data[:, [1, 2, 3, 0]])
        slerp = Slerp(x_orig, rots)
        rot_interp = slerp(x_new).as_quat() # 返回 [qx, qy, qz, qw]
        # 转回 [qw, qx, qy, qz]
        rot_interp = rot_interp[:, [3, 0, 1, 2]]
        
        return np.concatenate([rot_interp, pos_interp], axis=1)

    # 对所有人体骨骼进行插值
    new_pose_params_dict = {}
    for k, v in pose_params_dict.items():
        new_pose_params_dict[k] = interpolate_7d(v, num_frames)
    pose_params_dict = new_pose_params_dict
    
    # 对所有物体进行插值
    new_object_pose_params_dict = {}
    for k, v in object_pose_params_dict.items():
        new_object_pose_params_dict[k] = interpolate_7d(v, num_frames)
    raw_data['object_pose_params'] = new_object_pose_params_dict
    
    print(f"插值完成，新帧数: {num_frames}")

    pose_params_tensor_dict = {}
    for k, v in pose_params_dict.items():
        t = torch.tensor(v, dtype=torch.float32, device=device)
        mat = quaternion_to_matrix_torch(t)
        pose_params_tensor_dict[k] = mat

    with torch.no_grad():
        joint_positions_dict = human_model.compute_joint_positions(pose_params_tensor_dict)
    
    all_human_joints = []
    for i in range(num_frames):
        frame_joints = []
        for j_name in HOLOSOMA_JOINTS:
            mxm_name = JOINT_MAPPING[j_name]
            pos = joint_positions_dict[mxm_name][i].cpu().numpy()
            frame_joints.append(pos)
        all_human_joints.append(frame_joints)
        
    human_joints = np.array(all_human_joints) 
    
    # 5. 物体提取逻辑 (支持检索)
    available_objects = list(raw_data['object_pose_params'].keys())
    print(f"文件内包含的物体: {available_objects}")

    if args.obj_name:
        if args.obj_name in raw_data['object_pose_params']:
            obj_name = args.obj_name
        else:
            print(f"错误: 找不到物体 '{args.obj_name}'。请从以下列表中选择: {available_objects}")
            sys.exit(1)
    else:
        obj_name = available_objects[0]
        print(f"未指定 --obj_name，默认提取第一个物体: {obj_name}")

    print(f"正在提取物体位姿: {obj_name}")
    object_poses = raw_data['object_pose_params'][obj_name] # (T, 7) [qw, qx, qy, qz, x, y, z]
    
    # 6. 保存 object_movement.pkl
    obj_movement_data = {
        "body_pos_w": object_poses[:, 4:7][:, np.newaxis, :],  # [T, 1, 3]
        "body_quat_w": object_poses[:, 0:4][:, np.newaxis, :]  # [T, 1, 4]
    }
    obj_movement_pkl_path = os.path.join(args.output_dir, f"object_movement_{args.task_name}.pkl")
    with open(obj_movement_pkl_path, 'wb') as f:
        pickle.dump(obj_movement_data, f)
    print(f"成功保存物体运动数据到: {obj_movement_pkl_path}")

    # 7. 保存最终 .pt 文件
    save_data = {
        "human_joints": human_joints,
        "object_poses": object_poses
    }
    
    output_path = os.path.join(args.output_dir, f"{args.task_name}.pt")
    torch.save(save_data, output_path)
    print(f"成功保存处理后的数据到: {output_path}")
    print(f"human_joints shape: {human_joints.shape}")
    print(f"object_poses shape: {object_poses.shape}")

if __name__ == "__main__":
    main()
