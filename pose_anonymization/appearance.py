import numpy as np
from pose_format import Pose
from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std, unnormalize_mean_std, unshift_hand


def normalize_pose_size(pose: Pose):
    new_width = 200
    shift = 1.25
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)


def get_pose_apperance(pose: Pose, include_end_frame=False):
    pose = pre_process_mediapipe(pose)
    pose = normalize_mean_std(pose)

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
    for hand in ['LEFT', 'RIGHT']:
        # pylint: disable=protected-access
        wrist_index = pose.header._get_point_index('POSE_LANDMARKS', f'{hand}_WRIST')
        new_pose_data[:, :, wrist_index] = pose.body.data[:, :, wrist_index]

    pose.body.data = new_pose_data

    pose = unnormalize_mean_std(pose)
    for component in hand_components:
        unshift_hand(pose, component)

    normalize_pose_size(pose)

    return pose


def remove_appearance(pose: Pose, include_end_frame=False):
    pose, appearance = get_pose_apperance(pose, include_end_frame)
    return change_appearace(pose, appearance)


def transfer_appearance(pose: Pose, appearance_pose: Pose, include_end_frame=False):
    pose, appearance = get_pose_apperance(pose, include_end_frame)
    _, new_appearance = get_pose_apperance(appearance_pose, include_end_frame)

    # Switching the pose appearance
    return change_appearace(pose, appearance - new_appearance)
