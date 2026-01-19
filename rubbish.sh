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

