from functools import lru_cache

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.generic import pose_normalization_info

from pose_anonymization.data import load_mean_and_std, load_pose_header
from pose_format.utils.generic import normalize_pose_size


def get_pose_appearance(pose: Pose, include_end_frame=False):
    pose = pose.normalize(pose_normalization_info(pose.header))

    if include_end_frame:
        # Assuming the first and last frames are indicative of the signer's appearance
        appearance = pose.body.data[[0, -1]].mean(axis=0)
    else:
        # Assuming the first frame is indicative of the signer's appearance
        appearance = pose.body.data[0]

    return pose, appearance


def change_appearace(pose: Pose, appearance: np.ndarray):
    # Removing the appearance from the pose
    new_pose_data = pose.body.data - appearance

    # Bring back the hands
    hand_components = ['LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS']
    for component in pose.header.components:
        if component.name in hand_components:
            # pylint: disable=protected-access
            start = pose.header._get_point_index(component.name, component.points[0])
            end = start + len(component.points)
            new_pose_data[:, :, start:end] = pose.body.data[:, :, start:end]

    # Bring back the wrists
    body_component = next(c for c in pose.header.components if c.name == 'POSE_LANDMARKS')
    points = [f'{hand}_{point}' for hand in ['LEFT', 'RIGHT'] for point in ['WRIST', 'PINKY', 'INDEX', 'THUMB']]
    existing_points = [p for p in points if p in body_component.points]
    for point in existing_points:
        # pylint: disable=protected-access
        wrist_index = pose.header._get_point_index('POSE_LANDMARKS', point)
        new_pose_data[:, :, wrist_index] = pose.body.data[:, :, wrist_index]

    pose.body.data = new_pose_data

    normalize_pose_size(pose)

    return pose


@lru_cache(maxsize=1)
def get_mean_appearance():
    mean, _ = load_mean_and_std()

    data = mean.reshape((1, 1, -1, 3)) * 1000
    confidence = np.ones((1, 1, len(mean)))
    body = NumPyPoseBody(fps=1, data=data, confidence=confidence)
    pose = Pose(header=load_pose_header(), body=body)

    return pose


def transfer_appearance(pose: Pose, appearance_pose: Pose, include_end_frame=False):
    # Making sure the appearance pose has the same components as the pose, in the same order of points
    pose_components = [c.name for c in pose.header.components]
    pose_components_points = {c.name: c.points for c in pose.header.components}
    appearance_pose = appearance_pose.get_components(pose_components, pose_components_points)

    assert pose.header.total_points() == appearance_pose.header.total_points(), \
        "Appearance pose missing points"

    pose, appearance = get_pose_appearance(pose, include_end_frame)
    _, new_appearance = get_pose_appearance(appearance_pose, include_end_frame)

    # Switching the pose appearance
    return change_appearace(pose, appearance - new_appearance)


def remove_appearance(pose: Pose, include_end_frame=False):
    mean_pose = get_mean_appearance()
    return transfer_appearance(pose, mean_pose, include_end_frame=include_end_frame)
