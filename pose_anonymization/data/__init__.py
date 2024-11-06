from functools import lru_cache
from pathlib import Path

import numpy as np
from pose_format import PoseHeader, Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.reader import BufferReader

CURRENT_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def load_pose_header():
    with open(CURRENT_DIR / "header.poseheader", "rb") as f:
        return PoseHeader.read(BufferReader(f.read()))


@lru_cache(maxsize=1)
def load_mean_and_std():
    import json
    with open(CURRENT_DIR / "pose_normalization.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    mean, std = [], []
    for component in data.values():
        for point in component.values():
            mean.append(point["mean"])
            std.append(point["std"])

    # when std is 0, set std to 1
    std = np.array(std)
    std[std == 0] = 1

    return np.array(mean), std

@lru_cache(maxsize=1)
def load_mean_and_std_pose():
    pose_header = load_pose_header()
    mean, std = load_mean_and_std()
    mean_body = NumPyPoseBody(fps=1, data=mean.reshape((1, 1, -1, 3)), confidence=np.ones((1, 1, len(mean))))
    std_body = NumPyPoseBody(fps=1, data=std.reshape((1, 1, -1, 3)), confidence=np.ones((1, 1, len(std))))
    return Pose(header=pose_header, body=mean_body), Pose(header=pose_header, body=std_body)
