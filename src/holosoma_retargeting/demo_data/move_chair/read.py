import pickle

# Path to the pickle file
file_path = "/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_data/move_chair/object_movement_723.pkl"

# Load the data from the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Check if 'body_pos_w' is present in the data and print it
if 'body_pos_w' in data:
    body_pos_w = data['body_pos_w']
    print("body_pos_w:", body_pos_w)
else:
    print("Key 'body_pos_w' not found in the data.")



