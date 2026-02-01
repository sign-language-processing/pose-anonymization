from pose_format import Pose

from pose_anonymization.appearance import get_mean_appearance


def test_get_mean_appearance_returns_valid_pose():
    pose = get_mean_appearance()

    assert isinstance(pose, Pose)
    assert pose.header is not None
    assert pose.body is not None
    assert pose.body.data.shape[0] > 0  # has frames
    assert pose.body.data.shape[2] > 0  # has points
    assert pose.body.data.shape[3] == 3  # xyz coordinates
