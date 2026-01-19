# read_npz.py
import numpy as np

npz_path = "/home/magic-4090/yueruchen/holosoma/src/holosoma_retargeting/converted_res/object_interaction/low_chair_walk.npz"

def main():
    data = np.load(npz_path, allow_pickle=True)

    print(f"NPZ 文件路径: {npz_path}")
    print(f"包含的 key 数量: {len(data.files)}\n")

    for key in data.files:
        value = data[key]
        print(f"Key: {key}")
        print(f"  Type : {type(value)}")

        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")

            # 如果是 object / pickle，给出额外提示
            if value.dtype == object:
                print(f"  Note : object array (可能包含 list / dict / 自定义对象)")
        else:
            print(f"  Value: {value}")

        print("-" * 50)

if __name__ == "__main__":
    main()
