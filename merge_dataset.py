import os
import glob
import shutil
import re
import pandas as pd
import json
import numpy as np
import subprocess
import tempfile


# Define helper function to convert NumPy types to Python native types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


# Function to check if ffmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("Warning: ffmpeg not found on system path. Video validation will be limited.")
        return False


# Function to validate and repair video files
def validate_and_repair_video(source_path, destination_path, feature_name):
    """Validate a video file and repair it if needed before copying."""
    try:
        # Create a temp file for the repaired video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Try to repair the video using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', source_path, 
            '-c', 'copy', '-movflags', '+faststart',  # Fast copy with improved structure
            temp_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Warning: Could not repair video {os.path.basename(source_path)} for feature '{feature_name}'. Using original.")
            # Fall back to simple copy if repair fails
            shutil.copy2(source_path, destination_path)
        else:
            # Use the repaired video
            shutil.copy2(temp_path, destination_path)
            
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            
        return True
    except Exception as e:
        print(f"Error processing video {os.path.basename(source_path)} for feature '{feature_name}': {str(e)}")
        return False


# Define the new repository structure
repo_id = "/home/w/.cache/huggingface/lerobot/Ww1313w/TransferCube_Insertion"
base_dir = repo_id  # Main repository directory
data_dir = os.path.join(base_dir, "data")  # Data folder within repo
data_chunk_dir = os.path.join(data_dir, "chunk-000")  # Chunk folder for data
videos_dir = os.path.join(base_dir, "videos")  # Videos folder within repo
videos_chunk_dir = os.path.join(videos_dir, "chunk-000")  # Chunk folder for videos
meta_dir = os.path.join(base_dir, "meta")  # Meta folder within repo


# Define paths for source datasets
dataset1_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_1/data/chunk-000"
dataset2_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_2/data/chunk-000"
dataset1_videos_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_1/videos/chunk-000"
dataset2_videos_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_2/videos/chunk-000"
dataset1_meta_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_1/meta/episodes.jsonl"
dataset2_meta_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_2/meta/episodes.jsonl"
dataset1_stats_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_1/meta/episodes_stats.jsonl"
dataset2_stats_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_2/meta/episodes_stats.jsonl"
dataset1_tasks_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_1/meta/tasks.jsonl"
dataset2_tasks_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_2/meta/tasks.jsonl"
dataset1_info_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_1/meta/info.json"
dataset2_info_path = "/home/w/.cache/huggingface/lerobot/Ww1313w/local_dataset_2/meta/info.json"


# Check if ffmpeg is available for video processing
has_ffmpeg = check_ffmpeg()


# Create the directory structure if it doesn't exist
os.makedirs(data_chunk_dir, exist_ok=True)
os.makedirs(videos_chunk_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)
print(f"Created directory structure: {data_chunk_dir}")
print(f"Created directory structure: {videos_chunk_dir}")
print(f"Created directory structure: {meta_dir}")


# Find all parquet files in both datasets
dataset1_files = sorted(glob.glob(os.path.join(dataset1_path, "episode_*.parquet")))
dataset2_files = sorted(glob.glob(os.path.join(dataset2_path, "episode_*.parquet")))


# First, determine the maximum index value in the first dataset
max_index_value = -1
for file_path in dataset1_files:
    df = pd.read_parquet(file_path, engine='pyarrow')
    if 'index' in df.columns and not df.empty:
        current_max = df['index'].max()
        max_index_value = max(max_index_value, current_max)


print(f"Maximum index value in first dataset: {max_index_value}")
index_offset = max_index_value + 1
print(f"Index offset for second dataset: {index_offset}")


# Process data files from the first dataset, keeping original names
next_index = 0
for file_path in dataset1_files:
    filename = os.path.basename(file_path)
    destination = os.path.join(data_chunk_dir, filename)
    
    print(f"Copying from dataset1 data: {filename}")
    # Just copy the file directly (task_index remains 0)
    shutil.copy2(file_path, destination)
    
    # Update the next index based on the current file
    match = re.search(r'episode_(\d+)\.parquet', filename)
    if match:
        index = int(match.group(1))
        next_index = max(next_index, index + 1)


print(f"Completed copying {len(dataset1_files)} files from first dataset data")
print(f"Next episode index will be: {next_index}")


# Process data files from the second dataset with new indices and update task_index
for file_path in dataset2_files:
    filename = os.path.basename(file_path)
    new_filename = f"episode_{next_index:06d}.parquet"
    destination = os.path.join(data_chunk_dir, new_filename)
    
    print(f"Processing from dataset2 data: {filename} as {new_filename}")
    
    # Read the parquet file
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    # Update task_index to 1 for all rows
    df['task_index'] = 1
    
    # Update the index column by adding the offset
    if 'index' in df.columns:
        df['index'] = df['index'] + index_offset
    
    # Save to the new location
    df.to_parquet(destination, engine='pyarrow')
    
    next_index += 1


print(f"Completed processing {len(dataset2_files)} files from second dataset data")
print(f"Combined dataset now has {len(dataset1_files) + len(dataset2_files)} episodes in data")


# Get the dataset-level episode mapping to use for all features
episode_mapping = {}
# First, map all episodes from dataset1 to themselves
for file_path in dataset1_files:
    filename = os.path.basename(file_path)
    match = re.search(r'episode_(\d+)\.parquet', filename)
    if match:
        old_index = int(match.group(1))
        episode_mapping[f"dataset1_{old_index:06d}"] = old_index


# Then, map all episodes from dataset2 to new indices
next_index = max(episode_mapping.values()) + 1 if episode_mapping else 0
for file_path in dataset2_files:
    filename = os.path.basename(file_path)
    match = re.search(r'episode_(\d+)\.parquet', filename)
    if match:
        old_index = int(match.group(1))
        episode_mapping[f"dataset2_{old_index:06d}"] = next_index
        next_index += 1


# Handle video files
# Get feature folders from both datasets
feature_folders = set()
if os.path.exists(dataset1_videos_path):
    feature_folders.update([f for f in os.listdir(dataset1_videos_path) 
                           if os.path.isdir(os.path.join(dataset1_videos_path, f))])
if os.path.exists(dataset2_videos_path):
    feature_folders.update([f for f in os.listdir(dataset2_videos_path) 
                           if os.path.isdir(os.path.join(dataset2_videos_path, f))])
    
print(f"Found features: {feature_folders}")


# Process each feature folder
for feature in feature_folders:
    # Create feature directory in the output
    feature_output_dir = os.path.join(videos_chunk_dir, feature)
    os.makedirs(feature_output_dir, exist_ok=True)
    
    # Process videos from dataset 1
    feature_dir1 = os.path.join(dataset1_videos_path, feature)
    if os.path.exists(feature_dir1):
        videos1 = sorted(glob.glob(os.path.join(feature_dir1, "episode_*.mp4")))
        successful_copies = 0
        
        for video_path in videos1:
            video_name = os.path.basename(video_path)
            match = re.search(r'episode_(\d+)\.mp4', video_name)
            if match:
                old_index = int(match.group(1))
                # Use the same episode number from the original dataset
                new_index = episode_mapping.get(f"dataset1_{old_index:06d}", old_index)
                new_video_name = f"episode_{new_index:06d}.mp4"
                destination = os.path.join(feature_output_dir, new_video_name)
                
                print(f"Processing from dataset1 videos/{feature}: {video_name} as {new_video_name}")
                if has_ffmpeg:
                    if validate_and_repair_video(video_path, destination, feature):
                        successful_copies += 1
                else:
                    # Fall back to simple copy if ffmpeg isn't available
                    try:
                        shutil.copy2(video_path, destination)
                        successful_copies += 1
                    except Exception as e:
                        print(f"Error copying video: {str(e)}")
        
        print(f"Completed processing {successful_copies}/{len(videos1)} videos from feature '{feature}' in dataset1")
    
    # Process videos from dataset 2
    feature_dir2 = os.path.join(dataset2_videos_path, feature)
    if os.path.exists(feature_dir2):
        videos2 = sorted(glob.glob(os.path.join(feature_dir2, "episode_*.mp4")))
        successful_copies = 0
        
        for video_path in videos2:
            video_name = os.path.basename(video_path)
            match = re.search(r'episode_(\d+)\.mp4', video_name)
            if match:
                old_index = int(match.group(1))
                # Use the new mapped episode number for dataset2
                new_index = episode_mapping.get(f"dataset2_{old_index:06d}")
                if new_index is None:
                    print(f"Warning: No mapping found for dataset2 episode {old_index:06d}")
                    continue
                
                new_video_name = f"episode_{new_index:06d}.mp4"
                destination = os.path.join(feature_output_dir, new_video_name)
                
                print(f"Processing from dataset2 videos/{feature}: {video_name} as {new_video_name}")
                if has_ffmpeg:
                    if validate_and_repair_video(video_path, destination, feature):
                        successful_copies += 1
                else:
                    # Fall back to simple copy if ffmpeg isn't available
                    try:
                        shutil.copy2(video_path, destination)
                        successful_copies += 1
                    except Exception as e:
                        print(f"Error copying video: {str(e)}")
        
        print(f"Completed processing {successful_copies}/{len(videos2)} videos from feature '{feature}' in dataset2")


# Process episodes metadata file
output_meta_path = os.path.join(meta_dir, "episodes.jsonl")
with open(output_meta_path, 'w') as outfile:
    # Process dataset1 metadata
    if os.path.exists(dataset1_meta_path):
        with open(dataset1_meta_path, 'r') as infile:
            for line in infile:
                metadata = json.loads(line)
                # Keep the original episode index for dataset1
                episode_index = metadata["episode_index"]
                metadata["episode_index"] = episode_mapping.get(f"dataset1_{episode_index:06d}", episode_index)
                outfile.write(json.dumps(metadata) + '\n')
        print(f"Processed metadata from {dataset1_meta_path}")


    # Process dataset2 metadata
    if os.path.exists(dataset2_meta_path):
        with open(dataset2_meta_path, 'r') as infile:
            for line in infile:
                metadata = json.loads(line)
                # Update the episode index for dataset2 based on our mapping
                episode_index = metadata["episode_index"]
                new_index = episode_mapping.get(f"dataset2_{episode_index:06d}")
                
                if new_index is not None:
                    metadata["episode_index"] = new_index
                    outfile.write(json.dumps(metadata) + '\n')
                else:
                    print(f"Warning: No mapping found for dataset2 episode {episode_index}")
        print(f"Processed metadata from {dataset2_meta_path}")


# Process episodes_stats metadata file
output_stats_path = os.path.join(meta_dir, "episodes_stats.jsonl")
with open(output_stats_path, 'w') as outfile:
    # Process dataset1 stats
    if os.path.exists(dataset1_stats_path):
        with open(dataset1_stats_path, 'r') as infile:
            for line in infile:
                stats_data = json.loads(line)
                # Keep the original episode index for dataset1
                episode_index = stats_data["episode_index"]
                new_index = episode_mapping.get(f"dataset1_{episode_index:06d}", episode_index)
                
                # Update the episode index in the main object
                stats_data["episode_index"] = new_index
                
                # Also update episode_index in stats if it exists
                if "stats" in stats_data and "episode_index" in stats_data["stats"]:
                    stats_data["stats"]["episode_index"]["min"] = [new_index]
                    stats_data["stats"]["episode_index"]["max"] = [new_index]
                    stats_data["stats"]["episode_index"]["mean"] = [float(new_index)]
                
                # Convert all values to JSON serializable types
                stats_data = convert_to_serializable(stats_data)
                
                outfile.write(json.dumps(stats_data) + '\n')
        print(f"Processed stats from {dataset1_stats_path}")


    # Process dataset2 stats
    if os.path.exists(dataset2_stats_path):
        with open(dataset2_stats_path, 'r') as infile:
            for line in infile:
                stats_data = json.loads(line)
                # Update the episode index for dataset2 based on our mapping
                episode_index = stats_data["episode_index"]
                new_index = episode_mapping.get(f"dataset2_{episode_index:06d}")
                
                if new_index is not None:
                    # Update the episode index in the main object
                    stats_data["episode_index"] = new_index
                    
                    # Also update episode_index in stats if it exists
                    if "stats" in stats_data and "episode_index" in stats_data["stats"]:
                        stats_data["stats"]["episode_index"]["min"] = [new_index]
                        stats_data["stats"]["episode_index"]["max"] = [new_index]
                        stats_data["stats"]["episode_index"]["mean"] = [float(new_index)]
                    
                    # Update task_index to 1 if it exists in stats
                    if "stats" in stats_data and "task_index" in stats_data["stats"]:
                        stats_data["stats"]["task_index"]["min"] = [1]
                        stats_data["stats"]["task_index"]["max"] = [1]
                        stats_data["stats"]["task_index"]["mean"] = [1.0]
                        stats_data["stats"]["task_index"]["std"] = [0.0]
                    
                    # Update index values if they exist in the stats
                    if "stats" in stats_data and "index" in stats_data["stats"]:
                        min_val = stats_data["stats"]["index"]["min"][0] + index_offset
                        max_val = stats_data["stats"]["index"]["max"][0] + index_offset
                        mean_val = stats_data["stats"]["index"]["mean"][0] + index_offset
                        
                        stats_data["stats"]["index"]["min"] = [min_val]
                        stats_data["stats"]["index"]["max"] = [max_val]
                        stats_data["stats"]["index"]["mean"] = [mean_val]
                    
                    # Convert all values to JSON serializable types
                    stats_data = convert_to_serializable(stats_data)
                    
                    outfile.write(json.dumps(stats_data) + '\n')
                else:
                    print(f"Warning: No mapping found for dataset2 episode {episode_index} in stats")
        print(f"Processed stats from {dataset2_stats_path}")


# Process tasks.jsonl metadata file
output_tasks_path = os.path.join(meta_dir, "tasks.jsonl")
with open(output_tasks_path, 'w') as outfile:
    # Process dataset1 tasks (task_index 0)
    if os.path.exists(dataset1_tasks_path):
        with open(dataset1_tasks_path, 'r') as infile:
            for line in infile:
                task_data = json.loads(line)
                # Ensure task_index is 0 for dataset1
                task_data["task_index"] = 0
                outfile.write(json.dumps(task_data) + '\n')
        print(f"Processed tasks from {dataset1_tasks_path}")


    # Process dataset2 tasks (task_index 1)
    if os.path.exists(dataset2_tasks_path):
        with open(dataset2_tasks_path, 'r') as infile:
            for line in infile:
                task_data = json.loads(line)
                # Update task_index to 1 for dataset2
                task_data["task_index"] = 1
                outfile.write(json.dumps(task_data) + '\n')
        print(f"Processed tasks from {dataset2_tasks_path}")


# Process info.json
# Load info.json files from both datasets
with open(dataset1_info_path, 'r') as f:
    info1 = json.load(f)
    
with open(dataset2_info_path, 'r') as f:
    info2 = json.load(f)


# Create a new info.json for the merged dataset
merged_info = info1.copy()  # Start with dataset1's info


# Ensure we have codebase_version v2.1
merged_info["codebase_version"] = "v2.1"


# Calculate total frames from both datasets
total_frames1 = info1.get("total_frames", 0)
total_frames2 = info2.get("total_frames", 0)


# Update the counts
merged_info["total_episodes"] = len(dataset1_files) + len(dataset2_files)
merged_info["total_frames"] = total_frames1 + total_frames2
merged_info["total_tasks"] = 2  # We now have 2 tasks (task_index 0 and 1)


# Count total videos in the merged dataset
total_videos = 0
for feature in feature_folders:
    feature_output_dir = os.path.join(videos_chunk_dir, feature)
    if os.path.exists(feature_output_dir):
        feature_videos = len(glob.glob(os.path.join(feature_output_dir, "episode_*.mp4")))
        total_videos += feature_videos
        print(f"Feature '{feature}' contains {feature_videos} videos")


merged_info["total_videos"] = total_videos


# Update splits
# Get the highest episode index in the merged dataset
highest_episode = max(episode_mapping.values()) if episode_mapping else 0
merged_info["splits"]["train"] = f"0:{highest_episode + 1}"


# Save the merged info.json
output_info_path = os.path.join(meta_dir, "info.json")
with open(output_info_path, 'w') as f:
    json.dump(merged_info, f, indent=4)


print(f"Created merged info.json at {output_info_path}")