from functools import partial
from pathlib import Path

import numpy as np
from pose_format import Pose, PoseHeader
from tqdm.contrib.concurrent import process_map

from pose_anonymization.data.normalization import pre_process_pose

CURRENT_DIR = Path(__file__).parent


def process_file(file, pose_header: PoseHeader):
    with open(file, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        pose = pre_process_pose(pose, pose_header=pose_header)
    tensor = pose.body.data.filled(0)

    frames_sum = np.sum(tensor, axis=(0, 1))
    frames_squared_sum = np.sum(np.square(tensor), axis=(0, 1))
    unmasked_frames = pose.body.data[:, :, :, 0:1].mask == False
    num_unmasked_frames = np.sum(unmasked_frames, axis=(0, 1))

    return frames_sum, frames_squared_sum, num_unmasked_frames


def calc_mean_and_std(files, pose_header: PoseHeader):
    cumulative_sum, squared_sum, frames_count = None, None, None

    process_func = partial(process_file, pose_header=pose_header)
    results = process_map(process_func, files, max_workers=None, chunksize=1)

    for frames_sum, frames_squared_sum, num_unmasked_frames in results:
        cumulative_sum = frames_sum if cumulative_sum is None else cumulative_sum + frames_sum
        squared_sum = frames_squared_sum if squared_sum is None else squared_sum + frames_squared_sum
        frames_count = num_unmasked_frames if frames_count is None else frames_count + num_unmasked_frames

    mean = cumulative_sum / frames_count
    std = np.sqrt((squared_sum / frames_count) - np.square(mean))

    return mean, std


def main(poses_location: str):
    print("Listing files...")
    files = list(Path(poses_location).glob("*.pose"))
    print(f"Processing {len(files)} files")

    # get a single random pose
    with open(files[0], 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        pose = pre_process_pose(pose)

    mean, std = calc_mean_and_std(files, pose.header)

    # store header
    with open(CURRENT_DIR / "header.poseheader", "wb") as f:
        pose.header.write(f)

    i = 0
    mean_std_info = {}
    for component in pose.header.components:
        component_info = {}
        for point in component.points:
            component_info[point] = {
                "mean": mean[i].tolist(),
                "std": std[i].tolist()
            }
            i += 1
        mean_std_info[component.name] = component_info

    import json

    with open(CURRENT_DIR / "pose_normalization.json", "w", encoding="utf-8") as f:
        json.dump(mean_std_info, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collect normalization info')
    parser.add_argument('--dir', type=str, help='Directory containing the pose files',
                        default="/Volumes/Echo/GCS/sign-mt-poses")

    args = parser.parse_args()

    if not Path(args.dir).exists():
        raise FileNotFoundError(f"Directory {args.dir} does not exist")

    main(args.dir)
