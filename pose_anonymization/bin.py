#!/usr/bin/env python

import argparse

from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, correct_wrists

from pose_anonymization.appearance import remove_appearance, transfer_appearance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='path to input pose file')
    parser.add_argument('--appearance', type=str, help='path to appearance pose file')
    parser.add_argument('--output', required=True, type=str, help='path to output pose file')
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.input, "rb") as f:
        pose = Pose.read(f.read())

    pose = reduce_holistic(pose)
    correct_wrists(pose)

    if args.appearance:
        with open(args.appearance, "rb") as f:
            appearance_pose = Pose.read(f.read())

        pose = transfer_appearance(pose, appearance_pose)
    else:
        pose = remove_appearance(pose)

    with open(args.output, "wb") as f:
        pose.write(f)


if __name__ == '__main__':
    main()
