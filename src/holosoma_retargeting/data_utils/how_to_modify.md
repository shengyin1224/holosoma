/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/data_utils/convert_humoto_interaction.py 目前这里全是你自己写的FK，虽然公式差不多但还是有问题，我需要你完全用现成的maniptrans那边的代码，如果在这个文件夹不方便调取，那你就新创建几个文件把maniptrans那边所有你这边需要的代码复制过来懂吗，再自作聪明自己写，就给我去死。

下面你可能会用到
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/base.py
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/humoto_dataset.py
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/human_model/bone_names.py
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/human_model/human_model.py
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/utils/load_humoto.py
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/utils/np_torch_conversion.py
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/utils/pytorch3d_render_helper.py
/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/main/dataset/utils/rotation_helper.py


# 引入平台代码 (假设你在正确的路径下)
from humoto_dataset import HumotoDatasetBase, BONE_MAP_RH

注意：下面的你可以参考，不一定完全照搬！！！
# 你需要模拟一个 Dataset 类或者直接实例化它来利用它的逻辑

def improved_main():
    # 1. 直接复用平台的数据加载，它会自动处理 Z-up -> Y-up
    from load_humoto import load_one_humoto_sequence
    
    # 2. 加载数据 (这会自动处理物体的 Y-up 转换)
    raw_data = load_one_humoto_sequence(
        data_path=os.path.dirname(RAW_DATA_PATH), # 注意 load_one_humoto 需要目录路径
        include_text=False,
        y_up=True, # 关键：开启 Y-up 转换
        object_model=False,
        pose_params=True
    )

    # 3. 初始化 HumanModel (复用平台代码)
    from human_model import HumanModelDifferentiable
    human_model = HumanModelDifferentiable(
        character_data_path=MODEL_JSON_PATH, 
        device="cpu"
    )

    # 4. 准备 FK 数据
    # 获取 pose params (注意：这里需要处理插值，如果不需要插值直接用 raw_data['armature_pose_params'])
    pose_params_dict = raw_data['armature_pose_params']
    
    # 将 numpy 转为 tensor 并转为矩阵
    pose_params_tensor_dict = {}
    from rotation_helper import quaternion_to_matrix_torch
    
    for k, v in pose_params_dict.items():
        t = torch.tensor(v, dtype=torch.float32)
        quat = t[:, :4] # [T, 4]
        pos = t[:, 4:]  # [T, 3]
        
        rot_mat = quaternion_to_matrix_torch(quat) # 使用平台的 helper
        mat = torch.eye(4).repeat(t.shape[0], 1, 1)
        mat[:, :3, :3] = rot_mat
        mat[:, :3, 3] = pos
        
        # === 关键：应用 Global Fix (模拟 HumotoDatasetBase 的逻辑) ===
        # HumotoDatasetBase 中通常 fix_coordinate_system=True (X轴旋转270度)
        # 你需要确认你的目标环境是否需要这个旋转。如果需要，必须加在这里。
        if k == "mixamorig:Hips":
            # 这里需要手动构建那个旋转矩阵，或者如果不确定，最好还是实例化 HumotoDatasetBase
            pass 
            
        pose_params_tensor_dict[k] = mat

    # 5. 计算 FK (使用平台的 Model)
    # 这保证了数学和父子层级处理与平台完全一致
    joint_positions = human_model.compute_joint_positions(pose_params_tensor_dict)


