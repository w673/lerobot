from pathlib import Path

import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torchvision import transforms

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/train/act_aloha_sim_transfer_cube_human")
output_directory.mkdir(parents=True, exist_ok=True)

datasets = LeRobotDataset.create(
    repo_id = "lerobot/act_aloha_sim_insertion_human_1",
    features={
        "observation.images.top": {
            "dtype": "video",
            "shape": [
                3,
                480,
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
                14
            ],
            "names": {
                "motors": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper"
                ]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": {
                "motors": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper"
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

# pretrained_policy_path = Path("outputs/train/aloha_sim_insertion_human")
pretrained_policy_path = "lerobot/act_aloha_sim_insertion_human"

policy = ACTPolicy.from_pretrained(pretrained_policy_path)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_aloha/AlohaInsertion-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=500,
)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)
print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)
print(env.action_space)

flag = 0
for i in range(2000):
    datasets.clear_episode_buffer()
    # Reset the policy and environments to prepare for rollout
    policy.reset()
    numpy_observation, info = env.reset(seed=i)

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render())

    step = 0
    done = False
    while not done:

        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(numpy_observation["agent_pos"])
        image = torch.from_numpy(numpy_observation["pixels"]["top"])

        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.top": image,
            "task": "Transfer",
            "task_description": "pick up the peg with the right arm, pick up the socket with the left arm, and then insert the peg into the socket.",
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done

        if not terminated:
            datasets.add_frame(
                frame={
                    "observation.state": state.squeeze(0).cpu().numpy().astype(numpy.float32),
                    "observation.images.top": image.squeeze(),
                    "action": action.squeeze(0).cpu().numpy().astype(numpy.float32),
                },
                task="pick up the peg with the right arm, pick up the socket with the left arm, and then insert the peg into the socket.",
            )
        else:
            # 终止后的帧不加入数据集
            break

        step += 1

    if terminated:
        print("Success!")
        flag += 1
        video_path = output_directory.joinpath("rollout" + str(i) + "s" + ".mp4")
        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.metadata["render_fps"]
        
        
    else:
        print("Failure!")
        fps = env.metadata["render_fps"]
        video_path = output_directory.joinpath("rollout" + str(i) + "f" + ".mp4")
    datasets.save_episode()
    # Encode all frames into a mp4 video.
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
    datasets.clear_episode_buffer()

print(f"Video of the evaluation is available in '{video_path}'.")
print(f"{flag=}")