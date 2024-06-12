from pathlib import Path

from matplotlib import pyplot as plt
from pose_format import Pose
from pose_format.numpy.representation.distance import DistanceRepresentation
from pose_format.utils.optical_flow import OpticalFlowCalculator
# pylint: disable=import-error
from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13

example_dir = Path(__file__).parent.parent / "assets" / "example"

fig = plt.figure(figsize=(8, 2))

for directory in [example_dir / "original", example_dir / "anonymized"]:
    poses = []
    for word in ["kleine", "kinder", "essen", "pizza"]:
        with open(directory / f"{word}.pose", 'rb') as pose_file:
            poses.append(Pose.read(pose_file.read()))

    concatenated = concatenate_poses(poses).get_components(["FACE_LANDMARKS"])
    calculator = OpticalFlowCalculator(fps=30, distance=DistanceRepresentation())
    flow = calculator(concatenated.body.data).sum(-1).squeeze()

    # Plot the flow on the main plot, (173, 1)
    plt.plot(flow, label=directory.name.capitalize())

plt.yticks([])
plt.legend(loc='upper left')
plt.ylabel("Optical Flow")
plt.tight_layout()
fig.show()
fig.savefig(example_dir / "optical_flow.pdf")
