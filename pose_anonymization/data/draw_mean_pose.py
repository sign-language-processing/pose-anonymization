from pathlib import Path

import cv2
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic

from pose_anonymization.appearance import normalize_pose_size
from pose_anonymization.data import load_mean_and_std_pose
from pose_anonymization.data.normalization import unshift_hands

if __name__ == "__main__":
    mean_pose, _ = load_mean_and_std_pose()
    unshift_hands(mean_pose)
    normalize_pose_size(mean_pose)

    poses = {
        "full": mean_pose,
        "reduced": reduce_holistic(mean_pose)
    }
    for name, pose in poses.items():
        pose.focus()

        v = PoseVisualizer(pose)
        image_path = Path(__file__).parent / f"mean_pose_{name}.png"
        cv2.imwrite(str(image_path), next(v.draw()))
