import argparse

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


def main(pose_path: str, output_path: str):
    with open(pose_path, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        middle_frame = pose.body.data.shape[0] // 2
        pose.body = pose.body[middle_frame:middle_frame + 1]

    vis = PoseVisualizer(pose)
    frame = next(vis.draw(background_color=(255, 255, 255)))
    vis.save_frame(output_path, frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract middle frame from a pose file')
    parser.add_argument('--pose', type=str, help='Path to the pose file')
    parser.add_argument('--output', type=str, help='Path to the output file')
    args = parser.parse_args()

    main(args.pose, args.output)
