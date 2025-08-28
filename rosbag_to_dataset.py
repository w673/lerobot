from pathlib import Path

import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torchvision import transforms

import os
import json
import cv2
import pandas as pd
from tqdm import tqdm
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


import numpy as np
import torch
from sensor_msgs.msg import Image

def ros_img_to_tensor(msg: Image):
    if msg.encoding not in ["rgb8", "bgr8", "mono8"]:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    # 从 msg.data 转 numpy
    channels = 3 if msg.encoding in ["rgb8", "bgr8"] else 1
    img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)

    # 如果是 BGR 转 RGB
    if msg.encoding == "bgr8":
        img_np = img_np[..., ::-1]

    # 转 PyTorch
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor.unsqueeze(0)      # 添加 batch 维度

    return img_tensor




# Create a directory to store the video of the evaluation
output_directory = Path("outputs/train/act_aloha_sim_insertion_human")
output_directory.mkdir(parents=True, exist_ok=True)

datasets = LeRobotDataset.create(
    repo_id = "Ww1313w/true_robot",
    features={
        "observation.images.top": {
            "dtype": "video",
            "shape": [
                3,
                360,
                640
            ],
            "names": [
                "channel",
                "height",
                "width"
            ],
            "video_info": {
                "video.fps": 50.0,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                    "rw",
                    "gripper"
                ]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                    "rw",
                    "gripper"
                ]
            }
        }
    },
    root = "./local_dataset_1",
    fps = 50,
    use_videos = True,
)

# Select your device
device = "cuda"



flag = 0

bag_folder = "/home/w/lerobot/pick_place"
target_topics = ["/right/current_ee_pose", "/xr/right_hand_inputs", "/right_target_ee_pose", "/camera/color/image_raw"]
contain = [0, 0, 0, 0]
state = numpy.zeros((8,), dtype=numpy.float32)
action = numpy.zeros((8,), dtype=numpy.float32)
image = None
# 循环遍历文件夹里的所有文件
for foldername in os.listdir(bag_folder):
    for filename in os.listdir(os.path.join(bag_folder, foldername)):
        if filename.endswith(".db3") or filename.endswith(".bag"):  # ROS 2 默认 SQLite3 后缀是 .db3
            bag_path = os.path.join(bag_folder, foldername)
            bag_path = os.path.join(bag_path, filename)
            print(f"Processing: {bag_path}")

            # 设置 StorageOptions
            storage_options = rosbag2_py.StorageOptions(
                uri=bag_path,
                storage_id="sqlite3"  # ROS2 默认
            )

            # 设置 ConverterOptions
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr"
            )

            # 创建读取器
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)

            # 遍历每条消息
            while reader.has_next():
                topic, rawdata, timestamp = reader.read_next()

                # 如果不是我们想要的 topic，就跳过
                if topic not in target_topics:
                    continue

                # 反序列化消息
                if topic in ["/right/current_ee_pose"]:
                    contain[0] = 1
                    msg = deserialize_message(rawdata, get_message("geometry_msgs/msg/Pose"))
                    state[0] = msg.position.x
                    state[1] = msg.position.y
                    state[2] = msg.position.z
                    state[3] = msg.orientation.x
                    state[4] = msg.orientation.y
                    state[5] = msg.orientation.z
                    state[6] = msg.orientation.w
                elif topic in ["/xr/right_hand_inputs"]:
                    contain[1] = 1
                    msg = deserialize_message(rawdata, get_message("sensor_msgs/msg/Joy"))
                    action[7] = msg.buttons[4]  # gripper
                elif topic in ["/camera/color/image_raw"]:
                    contain[2] = 1
                    msg = deserialize_message(rawdata, get_message("sensor_msgs/msg/Image"))
                    image = ros_img_to_tensor(msg)
                elif topic in ["/right_target_ee_pose"]:
                    contain[3] = 1
                    msg = deserialize_message(rawdata, get_message("geometry_msgs/msg/Pose"))
                    action[0] = msg.position.x
                    action[1] = msg.position.y
                    action[2] = msg.position.z
                    action[3] = msg.orientation.x
                    action[4] = msg.orientation.y
                    action[5] = msg.orientation.z
                    action[6] = msg.orientation.w
                if contain == [1, 1, 1, 1]:
                    add_state = torch.from_numpy(state).to(torch.float32).to(device, non_blocking=True).unsqueeze(0)
                    add_action = torch.from_numpy(action).to(torch.float32).to(device, non_blocking=True).unsqueeze(0)
                    datasets.add_frame(
                        frame={
                            "observation.state": add_state.squeeze(0).cpu().numpy().astype(numpy.float32),
                            "observation.images.top": image.squeeze(),
                            "action": add_action.squeeze(0).cpu().numpy().astype(numpy.float32),
                        },
                        task="pick up the cube and place it.",
                    )
                    contain = [0, 0, 0, 0]
                    state = numpy.zeros((8,), dtype=numpy.float32)
                    state[7] = action[7]
                    action = numpy.zeros((8,), dtype=numpy.float32)
                    image = None    
            datasets.save_episode()
            flag += 1
            print(f"Finished episode {flag}, saved to dataset.")
            datasets.clear_episode_buffer()