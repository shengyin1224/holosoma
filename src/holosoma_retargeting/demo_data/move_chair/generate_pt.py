import torch
import pickle
import numpy as np

# 定义文件路径
robot_joints_path = '/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/move_chair/output_smplx_body_kpts_world_723_omniretargeting.pkl'
object_movement_path = '/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/move_chair/object_movement_723.pkl'
output_path = '/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/move_chair/move_chair.pt'

# 加载机器人关节数据
print("加载机器人关节数据...")
with open(robot_joints_path, 'rb') as f:
    robot_data = pickle.load(f)

# 提取body_positions: shape应该是(T, 15, 3)
body_positions = robot_data['body_positions']
print(f"body_positions shape: {body_positions.shape}")

# 加载物体运动数据
print("\n加载物体运动数据...")
with open(object_movement_path, 'rb') as f:
    object_data = pickle.load(f)

# 提取位置和四元数
body_pos_w = object_data['body_pos_w']  # shape: (T, 1, 3)
body_quat_w = object_data['body_quat_w']  # shape: (T, 1, 4), 顺序: qw, qx, qy, qz

print(f"body_pos_w shape: {body_pos_w.shape}")
print(f"body_quat_w shape: {body_quat_w.shape}")

# 将位置和四元数reshape成(T, 3)和(T, 4)
obj_pos = body_pos_w.squeeze(1)  # (T, 3)
obj_quat = body_quat_w.squeeze(1)  # (T, 4)

print(f"\n物体位置shape: {obj_pos.shape}")
print(f"物体四元数shape: {obj_quat.shape}")

# 合并物体的四元数和位置: [qw, qx, qy, qz, x, y, z]
object_data_flat = np.concatenate([obj_quat, obj_pos], axis=1)  # (T, 7)
print(f"物体数据展平后shape: {object_data_flat.shape}")

save_data = {
    "human_joints": body_positions,  # (T, 15, 3)
    "object_poses": object_data_flat,  # (T, 7) [qw, qx, qy, qz, x, y, z]
}

# 保存为.pt文件
torch.save(save_data, output_path)
print(f"\n成功保存到: {output_path}")

# 验证保存的文件
print("\n验证保存的文件...")
# 模拟在 hsretargeting 环境下运行，可能需要 mock numpy._core
import sys
try:
    import numpy._core
except ImportError:
    sys.modules['numpy._core'] = np
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

loaded_data = torch.load(output_path, map_location='cpu')
print(f"加载后的keys: {list(loaded_data.keys())}")
print(f"human_joints shape: {loaded_data['human_joints'].shape}")
print(f"object_poses shape: {loaded_data['object_poses'].shape}")
