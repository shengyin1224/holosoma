cd /home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/

python run_pipeline.py \
    --task_name baking_with_spatula_mixing_bowl_and_scooping_to_tray-244 \
    --obj_name mixing_bowl

python data_conversion/convert_data_format_mj.py \
    --input_file demo_results/g1/object_interaction/omomo/baking_with_spatula_mixing_bowl_and_scooping_to_tray-244_original.npz \
    --output_name demo_results/g1/object_interaction/omomo/output_for_isaac_60hz.npz \
    --robot g1 \
    --object_name table \
    --headless True \
    --input_fps 60 \
    --output_fps 60


python prepare_isaac_data.py \
    --input demo_results/g1/object_interaction/omomo/output_for_isaac_60hz.npz \
    --output demo_results/g1/object_interaction/omomo/isaac_final_ready.npz

# 必须先回到 holosoma 核心目录
cd /home/ubuntu/DATA1/shengyin/humanoid/holosoma

# 运行官方的 replay 脚本
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    simulator:isaacgym \
    --command.setup_terms.motion_command.params.motion_config.motion_file="/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_results/g1/object_interaction/omomo/output_for_isaac_60hz.npz" \
    --training.headless=False \
    --training.num_envs=1