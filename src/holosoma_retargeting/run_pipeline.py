import os
import shutil
import subprocess
import argparse
from pathlib import Path

# ----------------------------- 基础路径配置 -----------------------------
BASE_DIR = "/home/ubuntu/DATA1/shengyin/humanoid"
HUMOTO_FBX_ROOT = os.path.join(BASE_DIR, "HUMOTO/humoto")
HUMOTO_PKL_ROOT = os.path.join(BASE_DIR, "HUMOTO/humoto_3")
HOLOSOMA_ROOT = os.path.join(BASE_DIR, "holosoma/src/holosoma_retargeting")
MODEL_DEST_ROOT = os.path.join(HOLOSOMA_ROOT, "models")
DATA_UTILS = os.path.join(HOLOSOMA_ROOT, "data_utils")

# 环境路径
MANIPTRANS_PYTHON = "/home/ubuntu/miniconda3/envs/maniptrans/bin/python3"
HUMOTO_PYTHON = "/home/ubuntu/miniconda3/envs/humoto/bin/python3"

def run_cmd(cmd, env=None, cwd=None):
    print(f"[*] 执行命令: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=not isinstance(cmd, list), env=env, cwd=cwd)
    if result.returncode != 0:
        print(f"[!] 命令执行失败，返回码: {result.returncode}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Holosoma 全自动数据处理流水线")
    parser.add_argument("--task_name", type=str, required=True, help="任务名称，例如 transfer_vase_...-798")
    parser.add_argument("--obj_name", type=str, required=True, help="物品名称，例如 vase")
    args = parser.parse_args()

    task_name = args.task_name
    obj_name = args.obj_name

    # 1. 确定路径
    fbx_path = os.path.join(HUMOTO_FBX_ROOT, task_name, f"{task_name}.fbx")
    raw_pkl_path = os.path.join(HUMOTO_PKL_ROOT, task_name, f"{task_name}.pkl")
    urdf_src = os.path.join(HUMOTO_PKL_ROOT, task_name, f"{obj_name}.urdf")
    obj_src = os.path.join(HUMOTO_PKL_ROOT, task_name, f"{obj_name}_visual.obj")
    
    output_dir = os.path.join(HOLOSOMA_ROOT, "demo_data", obj_name)
    os.makedirs(output_dir, exist_ok=True)

    # 2. 复制并重命名模型文件 (统一为 {obj_name}.obj)
    model_dest_dir = os.path.join(MODEL_DEST_ROOT, obj_name)
    os.makedirs(model_dest_dir, exist_ok=True)
    print(f"[*] 正在复制模型文件到 {model_dest_dir}...")
    if os.path.exists(urdf_src):
        shutil.copy(urdf_src, os.path.join(model_dest_dir, f"{obj_name}.urdf"))
    
    # 优先找 _visual.obj, 找不到再找 .obj
    obj_copied = False
    if os.path.exists(obj_src):
        shutil.copy(obj_src, os.path.join(model_dest_dir, f"{obj_name}.obj"))
        # 同时复制一份带 _visual 的，防止 URDF 找不到
        shutil.copy(obj_src, os.path.join(model_dest_dir, f"{obj_name}_visual.obj"))
        obj_copied = True
    else:
        alt_obj_src = os.path.join(HUMOTO_PKL_ROOT, task_name, f"{obj_name}.obj")
        if os.path.exists(alt_obj_src):
            shutil.copy(alt_obj_src, os.path.join(model_dest_dir, f"{obj_name}.obj"))
            obj_copied = True
    
    # 尝试复制 _collision.obj
    collision_src = os.path.join(HUMOTO_PKL_ROOT, task_name, f"{obj_name}_collision.obj")
    if os.path.exists(collision_src):
        shutil.copy(collision_src, os.path.join(model_dest_dir, f"{obj_name}_collision.obj"))
    
    if not obj_copied:
        print(f"[!] 警告: 找不到模型文件 {obj_name}_visual.obj 或 {obj_name}.obj")

    # --- 新增：自动生成机器人 + 物体的 MuJoCo XML ---
    print(f"[*] 正在生成机器人模型 XML: models/g1/g1_29dof_w_{obj_name}.xml...")
    xml_template_path = os.path.join(HOLOSOMA_ROOT, "models/g1/g1_29dof_w_side_table.xml")
    new_xml_path = os.path.join(HOLOSOMA_ROOT, f"models/g1/g1_29dof_w_{obj_name}.xml")
    
    if os.path.exists(xml_template_path):
        with open(xml_template_path, 'r') as f:
            xml_content = f.read()
        
        # 替换 side_table 相关的名称和路径
        xml_content = xml_content.replace("side_table", obj_name)
        # 确保路径指向新的模型文件夹
        new_obj_path = os.path.join(MODEL_DEST_ROOT, obj_name, f"{obj_name}.obj")
        # 这是一个简单的替换，假设模板里路径格式固定
        
        with open(new_xml_path, 'w') as f:
            f.write(xml_content)
    else:
        print(f"[!] 警告: 找不到 XML 模板 {xml_template_path}, 可能会导致后续步骤失败。")

    # 3. 运行 generate_pkl.py (在 humoto 环境下运行)
    print("\n[Step 1/4] 运行 Blender 提取人体世界坐标 (humoto 环境)...")
    body_pkl_path = os.path.join(output_dir, f"body_joints_{task_name}.pkl")
    temp_env = os.environ.copy()
    temp_env["FBX_PATH"] = fbx_path
    temp_env["OUTPUT_PKL"] = os.path.join(output_dir, "temp_full.pkl")
    temp_env["OUTPUT_BODY_PKL"] = body_pkl_path
    temp_env["BLENDER_EXTRACT"] = "1"
    
    run_cmd([HUMOTO_PYTHON, os.path.join(DATA_UTILS, "generate_pkl.py")], env=temp_env)

    # 4. 运行 convert_humoto_interaction.py
    print("\n[Step 2/4] 运行 FK 脚本提取物体运动数据...")
    run_cmd([
        "python3", os.path.join(DATA_UTILS, "convert_humoto_interaction.py"),
        "--obj_name", obj_name,
        "--raw_data_path", raw_pkl_path,
        "--output_dir", output_dir,
        "--task_name", task_name
    ])

    # 5. 运行 merge_to_pt.py
    print("\n[Step 3/4] 合并人体和物体数据生成 .pt 文件...")
    final_pt_path = os.path.join(output_dir, f"{task_name}.pt")
    run_cmd([
        "python3", os.path.join(DATA_UTILS, "merge_to_pt.py"),
        "--robot_joints_path", body_pkl_path,
        "--object_movement_path", os.path.join(output_dir, f"object_movement_{task_name}.pkl"),
        "--output_path", final_pt_path
    ])

    # 6. 运行 robot_retarget.py
    print("\n[Step 4/4] 开始机器人重定向 (Retargeting)...")
    run_cmd([
        "python3", "examples/robot_retarget.py",
        "--task-type", "object_interaction",
        "--task-name", task_name,
        "--data-path", f"demo_data/{obj_name}",
        "--data-format", "humoto",
        "--task-config.object-name", obj_name
    ], cwd=HOLOSOMA_ROOT)

    # 7. 打印后续指令
    print("\n" + "="*60)
    print("[*] 流水线执行完毕！")
    print("[*] 请手动运行以下命令进行可视化：")
    print(f"\npython viser_player.py \\")
    print(f"    --robot_urdf {HOLOSOMA_ROOT}/models/g1/g1_29dof.urdf \\")
    print(f"    --object_urdf models/{obj_name}/{obj_name}.urdf \\")
    print(f"    --qpos_npz demo_results/g1/object_interaction/omomo/{task_name}_original.npz")
    print("="*60)

if __name__ == "__main__":
    main()
