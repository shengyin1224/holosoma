import bpy
import numpy as np
import pickle
import os
import sys
import subprocess

# ----------------------------- 环境自动切换配置 -----------------------------
# 目标环境 (humoto) 的 Python 解释器绝对路径
HUMOTO_PYTHON = "/home/ubuntu/miniconda3/envs/humoto/bin/python3"

def run_in_humoto():
    """检测并自动切换到 humoto 环境运行"""
    if sys.executable != HUMOTO_PYTHON:
        if not os.path.exists(HUMOTO_PYTHON):
            print(f"警告: 找不到 humoto 环境路径: {HUMOTO_PYTHON}，将尝试直接运行...")
            return
            
        print(f"[*] 当前环境为 {sys.executable}")
        print(f"[*] 正在自动切换到 humoto 环境执行 Blender 提取脚本...")
        
        try:
            # 传递所有环境变量和参数
            result = subprocess.run([HUMOTO_PYTHON] + sys.argv, env=os.environ)
            sys.exit(result.returncode)
        except Exception as e:
            print(f"切换环境执行失败: {e}")
            sys.exit(1)

# 如果不是由 run_pipeline 指定 Python 运行，则尝试自动切换
if __name__ == "__main__" and "BLENDER_EXTRACT" not in os.environ:
    run_in_humoto()

# ==== 路径设置 (优先从环境变量读取) ====
FBX_PATH = os.environ.get("FBX_PATH", r"/home/ubuntu/DATA1/shengyin/humanoid/HUMOTO/humoto/carry_side_table_with_both_hands_walk_around-536/carry_side_table_with_both_hands_walk_around-536.fbx")
OUTPUT_PKL = os.environ.get("OUTPUT_PKL", r"/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/carry_side_table/output_smplx_kpts_world_723_finger_tips.pkl")
OUTPUT_BODY_PKL = os.environ.get("OUTPUT_BODY_PKL", r"/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/carry_side_table/output_smplx_body_kpts_world_723_omniretargeting.pkl")

# ==== SMPL_X 关节顺序（加上 pelvis） ====
SMPLX_BODY_ORDER = [
    "pelvis",
    "left_hip", "right_hip",
    "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle",
    "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "jaw", "left_eye_smplx", "right_eye_smplx",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]

# ==== 简化 body 输出顺序 ====
SMPLX_BODY_POS_ORDER = [
    "pelvis",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_ankle", "right_ankle",
    "left_foot", "right_foot",
    "left_wrist", "right_wrist",
]

# ==== 需要额外存的“指尖”关节（SMPL_X 名称下的末节骨骼） ====
FINGERTIP_SMPLX = [
    "left_thumb3", "left_index3", "left_middle3", "left_ring3", "left_pinky3",
    "right_thumb3", "right_index3", "right_middle3", "right_ring3", "right_pinky3",
]

FINGERTIP_NAME = [
    "left_thumb_tips", "left_index_tips", "left_middle_tips", "left_ring_tips", "left_pinky_tips",
    "right_thumb_tips", "right_index_tips", "right_middle_tips", "right_ring_tips", "right_pinky_tips",
]

# ==== Mixamo -> SMPL_X 映射 ====
MIXAMO_TO_SMPLX = {
    "mixamorig:Hips": "pelvis",
    "mixamorig:LeftUpLeg": "left_hip",
    "mixamorig:RightUpLeg": "right_hip",
    "mixamorig:Spine": "spine1",
    "mixamorig:LeftLeg": "left_knee",
    "mixamorig:RightLeg": "right_knee",
    "mixamorig:Spine1": "spine2",
    "mixamorig:LeftFoot": "left_ankle",
    "mixamorig:RightFoot": "right_ankle",
    "mixamorig:Spine2": "spine3",
    "mixamorig:LeftToeBase": "left_foot",
    "mixamorig:RightToeBase": "right_foot",
    "mixamorig:Neck": "neck",
    "mixamorig:LeftShoulder": "left_collar",
    "mixamorig:RightShoulder": "right_collar",
    "mixamorig:Head": "head",
    "mixamorig:LeftArm": "left_shoulder",
    "mixamorig:RightArm": "right_shoulder",
    "mixamorig:LeftForeArm": "left_elbow",
    "mixamorig:RightForeArm": "right_elbow",
    "mixamorig:LeftHand": "left_wrist",
    "mixamorig:RightHand": "right_wrist",
    "mixamorig:LeftHandIndex1": "left_index1",
    "mixamorig:LeftHandIndex2": "left_index2",
    "mixamorig:LeftHandIndex3": "left_index3",
    "mixamorig:LeftHandMiddle1": "left_middle1",
    "mixamorig:LeftHandMiddle2": "left_middle2",
    "mixamorig:LeftHandMiddle3": "left_middle3",
    "mixamorig:LeftHandPinky1": "left_pinky1",
    "mixamorig:LeftHandPinky2": "left_pinky2",
    "mixamorig:LeftHandPinky3": "left_pinky3",
    "mixamorig:LeftHandRing1": "left_ring1",
    "mixamorig:LeftHandRing2": "left_ring2",
    "mixamorig:LeftHandRing3": "left_ring3",
    "mixamorig:LeftHandThumb1": "left_thumb1",
    "mixamorig:LeftHandThumb2": "left_thumb2",
    "mixamorig:LeftHandThumb3": "left_thumb3",
    "mixamorig:RightHandIndex1": "right_index1",
    "mixamorig:RightHandIndex2": "right_index2",
    "mixamorig:RightHandIndex3": "right_index3",
    "mixamorig:RightHandMiddle1": "right_middle1",
    "mixamorig:RightHandMiddle2": "right_middle2",
    "mixamorig:RightHandMiddle3": "right_middle3",
    "mixamorig:RightHandPinky1": "right_pinky1",
    "mixamorig:RightHandPinky2": "right_pinky2",
    "mixamorig:RightHandPinky3": "right_pinky3",
    "mixamorig:RightHandRing1": "right_ring1",
    "mixamorig:RightHandRing2": "right_ring2",
    "mixamorig:RightHandRing3": "right_ring3",
    "mixamorig:RightHandThumb1": "right_thumb1",
    "mixamorig:RightHandThumb2": "right_thumb2",
    "mixamorig:RightHandThumb3": "right_thumb3",
}

# 反向映射：SMPL_X 名 -> Mixamo 名
SMPLX_TO_MIXAMO = {}
for mix_name, smplx_name in MIXAMO_TO_SMPLX.items():
    SMPLX_TO_MIXAMO.setdefault(smplx_name, mix_name)

# 指尖索引表（按 FINGERTIP_SMPLX 顺序）
FINGERTIP_INDEX = {name: i for i, name in enumerate(FINGERTIP_SMPLX)}

# ========== 工具函数：速度计算 ==========
def _quat_normalize(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def _quat_conjugate(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def _quat_to_axis_angle(q):
    q = _quat_normalize(q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w*w, 0.0))  # sin(angle/2)
    axis = np.zeros(q.shape[:-1] + (3,), dtype=np.float32)
    small = s < 1e-8
    axis[~small, 0] = q[..., 1][~small] / s[~small]
    axis[~small, 1] = q[..., 2][~small] / s[~small]
    axis[~small, 2] = q[..., 3][~small] / s[~small]
    axis[small, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axis, angle

def compute_linear_velocity(pos, dt):
    vel = np.zeros_like(pos, dtype=np.float32)
    if pos.shape[0] <= 1:
        return vel
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel

def compute_angular_velocity_from_quat(q, dt):
    # q: (T, J, 4) wxyz, world orientation -> omega_world: (T, J, 3)
    T = q.shape[0]
    omega = np.zeros((T, q.shape[1], 3), dtype=np.float32)
    if T <= 1:
        return omega

    qn = _quat_normalize(q.astype(np.float32))

    # fix sign flips to keep continuity
    dots = np.sum(qn[1:] * qn[:-1], axis=-1)  # (T-1, J)
    flip = dots < 0.0
    qn[1:][flip] *= -1.0

    # forward diff for first
    dq_f = _quat_multiply(qn[1], _quat_conjugate(qn[0]))
    axis, angle = _quat_to_axis_angle(dq_f)
    omega[0] = axis * (angle[..., None] / dt)

    # backward diff for last
    dq_b = _quat_multiply(qn[-1], _quat_conjugate(qn[-2]))
    axis, angle = _quat_to_axis_angle(dq_b)
    omega[-1] = axis * (angle[..., None] / dt)

    # central diff for middle
    if T > 2:
        dq_c = _quat_multiply(qn[2:], _quat_conjugate(qn[:-2]))  # (T-2, J, 4)
        axis, angle = _quat_to_axis_angle(dq_c)
        omega[1:-1] = axis * (angle[..., None] / (2.0 * dt))

    return omega

# ==== 清空场景，导入 FBX ====
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX_PATH)

# 找骨架
armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
if not armatures:
    raise RuntimeError("No ARMATURE object found.")
arm = armatures[0]

mixamorig_bones = [b.name for b in arm.data.bones if "mixamorig" in b.name]
print("mixamorig bones in FBX:")
for name in sorted(mixamorig_bones):
    print(name)

if arm.animation_data is None or arm.animation_data.action is None:
    raise RuntimeError("Armature has no animation data.")
action = arm.animation_data.action

frame_start = int(action.frame_range[0])
frame_end = int(action.frame_range[1])
num_frames = frame_end - frame_start + 1
print(f"Frame range: {frame_start} ~ {frame_end}, num_frames = {num_frames}")

scene = bpy.context.scene
print("scene.render.fps_base: ",scene.render.fps_base)
fps = scene.render.fps / scene.render.fps_base
# 强制使用 60fps 进行采样。即便 Blender 显示 24fps，我们也按照数据实际的 30fps 来处理
orig_fps = 30.0 
target_fps = 60.0
sampling_step = orig_fps / target_fps # 30/60 = 0.5
# 更新为目标 FPS
fps = target_fps
dt = 1.0 / fps
print(f"Blender Scene FPS = {scene.render.fps / scene.render.fps_base}, Assumed Source FPS = {orig_fps}, Target FPS = {fps}, sampling_step = {sampling_step}")

num_joints = len(SMPLX_BODY_ORDER)
num_fingertips = len(FINGERTIP_SMPLX)

# 计算总帧数
num_frames = int((frame_end - frame_start) / sampling_step) + 1
joint_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
joint_orientations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)

# 额外：指尖位置（pb.tail），以及指尖末节骨的朝向（来自 joint_orientations 里切片）
fingertip_positions = np.zeros((num_frames, num_fingertips, 3), dtype=np.float32)
# 额外：头顶位置（head 骨骼的 pb.tail）
head_top_positions = np.zeros((num_frames, 3), dtype=np.float32)

pose_bones = arm.pose.bones

for fi in range(num_frames):
    current_frame = frame_start + fi * sampling_step
    # 拆分整数帧和亚帧，Blender 支持 subframe 采样
    frame_int = int(current_frame)
    frame_sub = current_frame - frame_int
    scene.frame_set(frame_int, subframe=frame_sub)
    M_world_arm = arm.matrix_world

    for j_idx, smplx_name in enumerate(SMPLX_BODY_ORDER):
        mix_name = SMPLX_TO_MIXAMO.get(smplx_name, None)
        if mix_name is None:
            continue
        if mix_name not in pose_bones:
            continue

        pb = pose_bones[mix_name]
        M_bone_world = M_world_arm @ pb.matrix

        t = M_bone_world.to_translation()
        q = M_bone_world.to_quaternion()

        joint_positions[fi, j_idx, :] = [t.x, t.y, t.z]
        joint_orientations[fi, j_idx, :] = [q.w, q.x, q.y, q.z]

        # 记录指尖位置：末节骨用 pb.tail
        if smplx_name in FINGERTIP_INDEX:
            tip_idx = FINGERTIP_INDEX[smplx_name]
            tip_world = M_world_arm @ pb.tail
            fingertip_positions[fi, tip_idx, :] = [tip_world.x, tip_world.y, tip_world.z]
        if smplx_name == "head":
            head_top_world = M_world_arm @ pb.tail
            head_top_positions[fi, :] = [head_top_world.x, head_top_world.y, head_top_world.z]

# ========== 线速度 / 角速度（全 joints） ==========
joint_linear_velocities = compute_linear_velocity(joint_positions, dt)                 # (T, J, 3)
joint_angular_velocities = compute_angular_velocity_from_quat(joint_orientations, dt) # (T, J, 3)

# ========== 指尖线速度（pb.tail） ==========
fingertip_linear_velocities = compute_linear_velocity(fingertip_positions, dt)        # (T, 10, 3)

# ========== 指尖末节骨的朝向 + 角速度（从 joints 切片，和 fingertip_names 对齐） ==========
SMPLX_NAME_TO_INDEX = {name: i for i, name in enumerate(SMPLX_BODY_ORDER)}
fingertip_joint_indices = np.array([SMPLX_NAME_TO_INDEX[n] for n in FINGERTIP_SMPLX], dtype=np.int64)

fingertip_orientations = joint_orientations[:, fingertip_joint_indices, :]              # (T,10,4)
fingertip_angular_velocities = joint_angular_velocities[:, fingertip_joint_indices, :]  # (T,10,3)

body_joint_indices = np.array([SMPLX_NAME_TO_INDEX[n] for n in SMPLX_BODY_POS_ORDER], dtype=np.int64)
body_positions = joint_positions[:, body_joint_indices, :]  # (T, body_num, 3)

# ========== 身高（头顶到脚底直线距离） ==========
left_foot_idx = SMPLX_NAME_TO_INDEX.get("left_foot", None)
right_foot_idx = SMPLX_NAME_TO_INDEX.get("right_foot", None)
if left_foot_idx is not None and right_foot_idx is not None:
    head_pos = head_top_positions                   # (T, 3)
    left_foot_pos = joint_positions[:, left_foot_idx, :]
    right_foot_pos = joint_positions[:, right_foot_idx, :]
    # 选择更低的脚作为“脚底”
    use_left = left_foot_pos[:, 2] <= right_foot_pos[:, 2]
    foot_pos = np.where(use_left[:, None], left_foot_pos, right_foot_pos)
    height_per_frame = np.linalg.norm(head_pos - foot_pos, axis=1)
    print("Height (head to foot) per frame:", height_per_frame)
    print(
        "Height stats: mean={:.4f}, min={:.4f}, max={:.4f}".format(
            float(np.mean(height_per_frame)),
            float(np.min(height_per_frame)),
            float(np.max(height_per_frame)),
        )
    )
else:
    print("Height not computed: missing head/foot joints.")

print("Done, saving to:", OUTPUT_PKL)
out = {
    "joint_positions": joint_positions,                 # (T, J, 3)
    "joint_orientations": joint_orientations,           # (T, J, 4) wxyz
    "joint_names": np.array(SMPLX_BODY_ORDER),

    "joint_linear_velocities": joint_linear_velocities,     # (T, J, 3) world, m_per_s
    "joint_angular_velocities": joint_angular_velocities,   # (T, J, 3) world, rad_per_s

    "fingertip_positions": fingertip_positions,             # (T, 10, 3) world (pb.tail)
    "fingertip_names": np.array(FINGERTIP_NAME),
    "fingertip_linear_velocities": fingertip_linear_velocities,  # (T, 10, 3) world

    "fingertip_orientations": fingertip_orientations,              # (T, 10, 4) wxyz (末节骨朝向)
    "fingertip_angular_velocities": fingertip_angular_velocities,  # (T, 10, 3) world, rad_per_s

    "body_positions": body_positions,                  # (T, body_num, 3)
    "body_names": np.array(SMPLX_BODY_POS_ORDER),

    "fps": float(fps),
    "dt": float(dt),
    "frame_start": frame_start,
    "frame_end": frame_end,
}

os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(out, f)

print("Saved:", OUTPUT_PKL)

body_out = {
    "body_positions": body_positions,
    "body_names": np.array(SMPLX_BODY_POS_ORDER),
    "fps": float(fps),
    "dt": float(dt),
    "frame_start": frame_start,
    "frame_end": frame_end,
}

os.makedirs(os.path.dirname(OUTPUT_BODY_PKL), exist_ok=True)
with open(OUTPUT_BODY_PKL, "wb") as f:
    pickle.dump(body_out, f)

print("Saved:", OUTPUT_BODY_PKL)

def save_rollout_npz(hdf5_path, output_npz_path, rollout_group="rollouts/successful/rollout_0"):
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        g = f[rollout_group]

        wrist_rh_xyz = g["state_rh"][:, 0:3]   # (T, 3)
        wrist_lh_xyz = g["state_lh"][:, 0:3]   # (T, 3)
        dof_rh = g["q_rh"][:]                  # (T, 12)
        dof_lh = g["q_lh"][:]                  # (T, 12)

    np.savez(
        output_npz_path,
        wrist_rh_xyz=wrist_rh_xyz,
        wrist_lh_xyz=wrist_lh_xyz,
        dof_rh=dof_rh,
        dof_lh=dof_lh,
    )

    print("Wrist Right Hand XYZ:", wrist_rh_xyz.shape)
    print("Wrist Left Hand XYZ:", wrist_lh_xyz.shape)
    print("Degrees of Freedom Right Hand:", dof_rh.shape)
    print("Degrees of Freedom Left Hand:", dof_lh.shape)
    print("Saved NPZ:", output_npz_path)