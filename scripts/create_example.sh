#!/bin/bash

set -e
set -x

# Original Poses
visualize_pose -i "assets/example/original/kleine.pose" -o "assets/example/original/kleine.mp4"
visualize_pose -i "assets/example/original/kinder.pose" -o "assets/example/original/kinder.mp4"
visualize_pose -i "assets/example/original/essen.pose" -o "assets/example/original/essen.mp4"
visualize_pose -i "assets/example/original/pizza.pose" -o "assets/example/original/pizza.mp4"

ffmpeg -i assets/example/original/kleine.mp4 assets/example/original/kleine.gif -y
ffmpeg -i assets/example/original/kinder.mp4 assets/example/original/kinder.gif -y
ffmpeg -i assets/example/original/essen.mp4 assets/example/original/essen.gif -y
ffmpeg -i assets/example/original/pizza.mp4 assets/example/original/pizza.gif -y

python ./scripts/extract_middle_frame.py --pose=assets/example/original/kleine.pose --output=assets/example/original/kleine.png
python ./scripts/extract_middle_frame.py --pose=assets/example/original/kinder.pose --output=assets/example/original/kinder.png
python ./scripts/extract_middle_frame.py --pose=assets/example/original/essen.pose --output=assets/example/original/essen.png
python ./scripts/extract_middle_frame.py --pose=assets/example/original/pizza.pose --output=assets/example/original/pizza.png

rm assets/example/original/kleine.mp4
rm assets/example/original/kinder.mp4
rm assets/example/original/essen.mp4
rm assets/example/original/pizza.mp4


# Anonymized Poses
python -m pose_anonymization.bin --input="assets/example/original/kleine.pose" --output="assets/example/anonymized/kleine.pose"
python -m pose_anonymization.bin --input="assets/example/original/kinder.pose" --output="assets/example/anonymized/kinder.pose"
python -m pose_anonymization.bin --input="assets/example/original/essen.pose" --output="assets/example/anonymized/essen.pose"
python -m pose_anonymization.bin --input="assets/example/original/pizza.pose" --output="assets/example/anonymized/pizza.pose"

visualize_pose -i "assets/example/anonymized/kleine.pose" -o "assets/example/anonymized/kleine.mp4"
visualize_pose -i "assets/example/anonymized/kinder.pose" -o "assets/example/anonymized/kinder.mp4"
visualize_pose -i "assets/example/anonymized/essen.pose" -o "assets/example/anonymized/essen.mp4"
visualize_pose -i "assets/example/anonymized/pizza.pose" -o "assets/example/anonymized/pizza.mp4"

ffmpeg -i assets/example/anonymized/kleine.mp4 assets/example/anonymized/kleine.gif -y
ffmpeg -i assets/example/anonymized/kinder.mp4 assets/example/anonymized/kinder.gif -y
ffmpeg -i assets/example/anonymized/essen.mp4 assets/example/anonymized/essen.gif -y
ffmpeg -i assets/example/anonymized/pizza.mp4 assets/example/anonymized/pizza.gif -y

python ./scripts/extract_middle_frame.py --pose=assets/example/anonymized/kleine.pose --output=assets/example/anonymized/kleine.png
python ./scripts/extract_middle_frame.py --pose=assets/example/anonymized/kinder.pose --output=assets/example/anonymized/kinder.png
python ./scripts/extract_middle_frame.py --pose=assets/example/anonymized/essen.pose --output=assets/example/anonymized/essen.png
python ./scripts/extract_middle_frame.py --pose=assets/example/anonymized/pizza.pose --output=assets/example/anonymized/pizza.png

rm assets/example/anonymized/kleine.mp4
rm assets/example/anonymized/kinder.mp4
rm assets/example/anonymized/essen.mp4
rm assets/example/anonymized/pizza.mp4

# Transferred Appearance
python -m pose_anonymization.bin --input="assets/example/original/kleine.pose" --output="assets/example/interpreter/kleine.pose" --appearance="assets/example/interpreter.pose"
python -m pose_anonymization.bin --input="assets/example/original/kinder.pose" --output="assets/example/interpreter/kinder.pose" --appearance="assets/example/interpreter.pose"
python -m pose_anonymization.bin --input="assets/example/original/essen.pose" --output="assets/example/interpreter/essen.pose" --appearance="assets/example/interpreter.pose"
python -m pose_anonymization.bin --input="assets/example/original/pizza.pose" --output="assets/example/interpreter/pizza.pose" --appearance="assets/example/interpreter.pose"

visualize_pose -i "assets/example/interpreter/kleine.pose" -o "assets/example/interpreter/kleine.mp4"
visualize_pose -i "assets/example/interpreter/kinder.pose" -o "assets/example/interpreter/kinder.mp4"
visualize_pose -i "assets/example/interpreter/essen.pose" -o "assets/example/interpreter/essen.mp4"
visualize_pose -i "assets/example/interpreter/pizza.pose" -o "assets/example/interpreter/pizza.mp4"

ffmpeg -i assets/example/interpreter/kleine.mp4 assets/example/interpreter/kleine.gif -y
ffmpeg -i assets/example/interpreter/kinder.mp4 assets/example/interpreter/kinder.gif -y
ffmpeg -i assets/example/interpreter/essen.mp4 assets/example/interpreter/essen.gif -y
ffmpeg -i assets/example/interpreter/pizza.mp4 assets/example/interpreter/pizza.gif -y

python ./scripts/extract_middle_frame.py --pose=assets/example/interpreter/kleine.pose --output=assets/example/interpreter/kleine.png
python ./scripts/extract_middle_frame.py --pose=assets/example/interpreter/kinder.pose --output=assets/example/interpreter/kinder.png
python ./scripts/extract_middle_frame.py --pose=assets/example/interpreter/essen.pose --output=assets/example/interpreter/essen.png
python ./scripts/extract_middle_frame.py --pose=assets/example/interpreter/pizza.pose --output=assets/example/interpreter/pizza.png

rm assets/example/interpreter/kleine.mp4
rm assets/example/interpreter/kinder.mp4
rm assets/example/interpreter/essen.mp4
rm assets/example/interpreter/pizza.mp4
