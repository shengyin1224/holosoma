import numpy as np
import mujoco
import os

# --- 配置路径 ---
NPZ_PATH = "demo_results/g1/object_interaction/omomo/baking_with_spatula_mixing_bowl_and_scooping_to_tray-244_original.npz"
XML_PATH = "models/g1/g1_29dof_w_mixing_bowl.xml" # 或者是您生成的带物体的 XML

# 1. 加载数据
data = np.load(NPZ_PATH)
qpos = data['qpos']  # (T, 43)
fps = float(data['fps'])
dt = 1.0 / fps
num_frames = qpos.shape[0]

# 2. 加载模型用于 FK
model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data = mujoco.MjData(model)

# 结果存储
extracted_data = {
    "root_pos": qpos[:, 0:3],
    "root_quat": qpos[:, 3:7], # [qw, qx, qy, qz]
    "head_pos": [],
    "head_quat": [],
    "root_lin_vel": [],
    "root_ang_vel": []
}

# 3. 提取头部和计算速度
for i in range(num_frames):
    # --- 计算头部 FK ---
    # 注意：这里的 qpos 顺序需要和 XML 匹配
    # G1 29DOF: root(7) + joints(29) + object(7) = 43
    mj_data.qpos[:43] = qpos[i] 
    mujoco.mj_forward(model, mj_data)
    
    # 假设头部的 geom 名称是 'head_link'
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_link")
    if head_id != -1:
        extracted_data["head_pos"].append(mj_data.geom_xpos[head_id].copy())
        # 将 xmat (3x3) 转换为 quat (4,)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mj_data.geom_xmat[head_id])
        extracted_data["head_quat"].append(quat.copy())
    else:
        # 如果找不到 head_link geom，尝试找 torso_link body
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        extracted_data["head_pos"].append(mj_data.xpos[torso_id].copy())
        extracted_data["head_quat"].append(mj_data.xquat[torso_id].copy())

# 转换为 numpy
extracted_data["head_pos"] = np.array(extracted_data["head_pos"])
extracted_data["head_quat"] = np.array(extracted_data["head_quat"])

# 4. 计算 Root 速度 (一阶差分)
pos = extracted_data["root_pos"]
lin_vel = np.zeros_like(pos)
lin_vel[1:] = (pos[1:] - pos[:-1]) / dt
extracted_data["root_lin_vel"] = lin_vel

# 计算 Root 角速度 (从四元数变化率提取)
def quat_to_angvel(q, dt):
    ang_vel = np.zeros((len(q), 3))
    for i in range(1, len(q)):
        q_curr = q[i-1]
        q_next = q[i]
        
        # 计算相对旋转: q_diff = q_next * conj(q_curr)
        q_curr_conj = q_curr.copy()
        q_curr_conj[1:] *= -1 # Conjugate: [w, x, y, z] -> [w, -x, -y, -z]
        q_diff = np.zeros(4)
        mujoco.mju_mulQuat(q_diff, q_next, q_curr_conj)
        
        # 转换为角速度
        v = np.zeros(3)
        mujoco.mju_quat2Vel(v, q_diff, dt)
        ang_vel[i] = v
    return ang_vel

extracted_data["root_ang_vel"] = quat_to_angvel(extracted_data["root_quat"], dt)

# 打印信息确认
print(f"提取完成！")
print(f"Root Pos Shape: {extracted_data['root_pos'].shape}")
print(f"Head Pos Shape: {extracted_data['head_pos'].shape}")

# --- 新增：打印前十帧数据 ---
print("\n" + "="*50)
print("前 10 帧数据预览:")
print("="*50)
for i in range(min(10, num_frames)):
    print(f"Frame {i}:")
    print(f"  Root Pos:     {extracted_data['root_pos'][i]}")
    print(f"  Root Quat:    {extracted_data['root_quat'][i]}")
    print(f"  Root Lin Vel: {extracted_data['root_lin_vel'][i]}")
    print(f"  Root Ang Vel: {extracted_data['root_ang_vel'][i]}")
    print(f"  Head Pos:     {extracted_data['head_pos'][i]}")
    print("-" * 20)
print("="*50)
