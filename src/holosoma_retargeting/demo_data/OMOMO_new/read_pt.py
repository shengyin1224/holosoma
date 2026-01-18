# import torch

# pt_path = "/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/OMOMO_new/sub3_largebox_003.pt"

# data = torch.load(pt_path, map_location="cpu")

# print("数据类型:", type(data))

# if isinstance(data, dict):
#     print("这是一个 dict，包含的 key 有：")
#     for k, v in data.items():
#         print(f"  key: {k}, type: {type(v)}")
# elif torch.is_tensor(data):
#     print("这是一个 Tensor，shape:", data.shape)
# else:
#     print("其他类型，内容为：")
#     print(data)
import pickle
import numpy as np

pkl_path = "/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/height_dict.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("数据类型:", type(data))
print("包含的 key:", data.keys())

body_quat_w = data["body_quat_w"]

print("\nbody_names 类型:", type(body_quat_w))
print("body_names shape:", body_quat_w.shape)

# print("\nbody_names 内容:")
# for i, name in enumerate(body_names):
#     print(i, name)
