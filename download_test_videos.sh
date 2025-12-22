#!/bin/bash
# Download test videos for YOLO object detection
# These are free stock videos suitable for testing object detection

set -e

echo "Downloading test videos for YOLO Object Detection..."
echo ""

# Create test_videos directory if it doesn't exist
mkdir -p test_videos

# Video 1: People walking (from Pixabay - free, no attribution required)
echo "1. Downloading people walking video..."
curl -L -o test_videos/people_walking.mp4 \
  "https://cdn.pixabay.com/video/2019/07/18/25307-349368863_tiny.mp4"
echo "   ✓ Downloaded: test_videos/people_walking.mp4"

# Video 2: Traffic and cars (from Pixabay)
echo "2. Downloading traffic video..."
curl -L -o test_videos/traffic.mp4 \
  "https://cdn.pixabay.com/video/2016/08/23/4803-180010611_tiny.mp4"
echo "   ✓ Downloaded: test_videos/traffic.mp4"

# Video 3: Person with bicycle (from Pixabay)
echo "3. Downloading bicycle video..."
curl -L -o test_videos/bicycle.mp4 \
  "https://cdn.pixabay.com/video/2021/03/11/67308-523895103_tiny.mp4"
echo "   ✓ Downloaded: test_videos/bicycle.mp4"

echo ""
echo "All test videos downloaded successfully!"
echo ""
echo "Video details:"
ls -lh test_videos/*.mp4

echo ""
echo "Usage examples:"
echo "  python main.py --source test_videos/people_walking.mp4"
echo "  python main.py --source test_videos/traffic.mp4 --output output.mp4 --data-output data.json --headless"
