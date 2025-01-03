from pose_format import Pose, PoseHeader
from pose_format.utils.generic import pose_normalization_info, correct_wrists, hands_components

from pose_anonymization.data import load_mean_and_std, load_mean_and_std_pose


def shift_hand(pose: Pose, hand_component: str, wrist_name: str):
    # pylint: disable=protected-access
    wrist_index = pose.header._get_point_index(hand_component, wrist_name)
    hand = pose.body.data[:, :, wrist_index: wrist_index + 21]
    wrist = hand[:, :, 0:1]
    pose.body.data[:, :, wrist_index: wrist_index + 21] = hand - wrist


def shift_hands(pose: Pose):
    (left_hand_component, right_hand_component), _, (wrist, _) = hands_components(pose.header)
    shift_hand(pose, left_hand_component, wrist)
    shift_hand(pose, right_hand_component, wrist)


def unshift_hand(pose: Pose, hand_component: str):
    # pylint: disable=protected-access
    wrist_index = pose.header._get_point_index(hand_component, "WRIST")
    hand = pose.body.data[:, :, wrist_index: wrist_index + 21]
    body_wrist_name = "LEFT_WRIST" if hand_component == "LEFT_HAND_LANDMARKS" else "RIGHT_WRIST"
    # pylint: disable=protected-access
    body_wrist_index = pose.header._get_point_index("POSE_LANDMARKS", body_wrist_name)
    body_wrist = pose.body.data[:, :, body_wrist_index: body_wrist_index + 1]
    pose.body.data[:, :, wrist_index: wrist_index + 21] = hand + body_wrist


def unshift_hands(pose: Pose):
    (left_hand_component, right_hand_component), _, _ = hands_components(pose.header)
    unshift_hand(pose, left_hand_component)
    unshift_hand(pose, right_hand_component)


def pose_like(pose: Pose, pose_header: PoseHeader):
    component_names = [component.name for component in pose_header.components]
    component_points = {component.name: component.points for component in pose_header.components}
    return pose.get_components(component_names, component_points)


def pre_process_pose(pose: Pose, pose_header: PoseHeader = None):
    if pose_header is not None:
        pose = pose_like(pose, pose_header)

    # Align hand wrists with body wrists
    correct_wrists(pose)
    # Adjust pose based on shoulder positions
    pose = pose.normalize(pose_normalization_info(pose.header))
    # Shift hands to origin
    shift_hands(pose)
    return pose


def load_mean_and_std_for_pose(pose: Pose):
    mean_pose, std_pose = load_mean_and_std_pose()
    mean_pose = pose_like(mean_pose, pose.header)
    std_pose = pose_like(std_pose, pose.header)
    return (mean_pose.body.data.reshape((-1, 3)),
            std_pose.body.data.reshape((-1, 3)))


def normalize_mean_std(pose: Pose):
    pose = pre_process_pose(pose)
    mean, std = load_mean_and_std_for_pose(pose)
    pose.body.data = (pose.body.data - mean) / std
    return pose


def unnormalize_mean_std(pose: Pose):
    mean, std = load_mean_and_std()
    pose.body.data = (pose.body.data * std) + mean
    unshift_hands(pose)
    return pose
