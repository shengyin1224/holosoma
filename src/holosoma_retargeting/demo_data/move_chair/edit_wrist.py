import torch
import h5py
import numpy as np

# 定义文件路径
old_pt_path = '/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/move_chair/move_chair_old.pt'
hdf5_path = '/home/magic-4090/yueruchen/rollouts.hdf5'
output_path = '/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/move_chair/move_chair.pt'

# 加载原始的move_chair_old.pt
print("加载原始文件...")
loaded_data = torch.load(old_pt_path, weights_only=False)

print(f"原始数据是字典,包含keys: {loaded_data.keys()}")
human_joints = loaded_data['human_joints']  # (274, 15, 3)
object_poses = loaded_data['object_poses']  # (274, 7)

print(f"human_joints shape: {human_joints.shape}")
print(f"object_poses shape: {object_poses.shape}")

# 从HDF5文件中读取手腕位置
print("\n从HDF5文件读取手腕数据...")
with h5py.File(hdf5_path, "r") as f:
    g = f["rollouts/successful/rollout_0"]
    
    wrist_rh_xyz = g["state_rh"][:, 0:3]   # (T, 3) 右手手腕
    wrist_lh_xyz = g["state_lh"][:, 0:3]   # (T, 3) 左手手腕
    
    # 交换x和y坐标 (原本是x,y,z -> 改为y,x,z)
    wrist_rh_xyz = wrist_rh_xyz[:, [1, 0, 2]]
    wrist_lh_xyz = wrist_lh_xyz[:, [1, 0, 2]]
    
    dof_rh = g["q_rh"][:]  # (T, 12)
    dof_lh = g["q_lh"][:]  # (T, 12)

print(f"右手手腕 XYZ shape: {wrist_rh_xyz.shape}")
print(f"左手手腕 XYZ shape: {wrist_lh_xyz.shape}")
print(f"右手自由度 shape: {dof_rh.shape}")
print(f"左手自由度 shape: {dof_lh.shape}")

# 检查帧数是否匹配
num_frames_old = human_joints.shape[0]
num_frames_hdf5 = wrist_lh_xyz.shape[0]
print(f"\n原始文件帧数: {num_frames_old}")
print(f"HDF5文件帧数: {num_frames_hdf5}")

# 如果帧数不同,对HDF5数据进行降采样
if num_frames_old != num_frames_hdf5:
    print(f"帧数不匹配! 将HDF5数据从{num_frames_hdf5}帧降采样到{num_frames_old}帧")
    
    # 创建降采样索引
    indices = np.linspace(0, num_frames_hdf5 - 1, num_frames_old).astype(int)
    
    # 对手腕数据进行降采样
    wrist_lh_xyz = wrist_lh_xyz[indices]
    wrist_rh_xyz = wrist_rh_xyz[indices]
    
    print(f"降采样后左手手腕 shape: {wrist_lh_xyz.shape}")
    print(f"降采样后右手手腕 shape: {wrist_rh_xyz.shape}")

num_frames = num_frames_old

# 更新第14个关节(左手手腕): 索引13
# 更新第15个关节(右手手腕): 索引14
print("\n更新手腕位置...")
print(f"第14个关节(左手手腕)原始值(前3帧):\n{human_joints[:3, 13, :]}")
print(f"第15个关节(右手手腕)原始值(前3帧):\n{human_joints[:3, 14, :]}")

# 替换数据
human_joints[:, 13, :] = wrist_lh_xyz  # 左手手腕,索引13
human_joints[:, 14, :] = wrist_rh_xyz  # 右手手腕,索引14

print(f"\n第14个关节(左手手腕)更新后值(前3帧):\n{human_joints[:3, 13, :]}")
print(f"第15个关节(右手手腕)更新后值(前3帧):\n{human_joints[:3, 14, :]}")

# 将human_joints展平为(274, 45)
human_joints_flat = human_joints.reshape(num_frames, -1)
print(f"\nhuman_joints展平后 shape: {human_joints_flat.shape}")

# 合并human_joints和object_poses
final_data = np.concatenate([human_joints_flat, object_poses], axis=1)
print(f"最终数据 shape: {final_data.shape}")  # 应该是 (274, 52)

# 转换回torch tensor
final_tensor = torch.from_numpy(final_data).float()
print(f"\n最终tensor shape: {final_tensor.shape}")
print(f"最终tensor dtype: {final_tensor.dtype}")

# 保存为新的.pt文件
torch.save(final_tensor, output_path)
print(f"\n成功保存到: {output_path}")

# 验证保存的文件
print("\n验证保存的文件...")
loaded_tensor = torch.load(output_path, weights_only=False)
print(f"加载后的tensor shape: {loaded_tensor.shape}")
print(f"左手手腕位置(前3帧):\n{loaded_tensor[:3, 39:42]}")
print(f"右手手腕位置(前3帧):\n{loaded_tensor[:3, 42:45]}")

# 显示完整的数据范围统计
print("\n数据统计:")
print(f"左手手腕 - min: {loaded_tensor[:, 39:42].min():.4f}, max: {loaded_tensor[:, 39:42].max():.4f}")
print(f"右手手腕 - min: {loaded_tensor[:, 42:45].min():.4f}, max: {loaded_tensor[:, 42:45].max():.4f}")