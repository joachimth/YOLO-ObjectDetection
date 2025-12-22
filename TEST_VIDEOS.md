# Test Videos for YOLO Object Detection

This document provides information about test videos for YOLO object detection and the automated GitHub workflow.

## Automated Testing

The repository includes a GitHub Actions workflow (`.github/workflows/video-test.yml`) that automatically:
- Runs on every push to main or `claude/**` branches
- Downloads a test video
- Processes it through YOLO object detection
- Saves annotated video and detection data as artifacts
- Creates a summary of detected objects

### Viewing Test Results

After a workflow run completes:
1. Go to the **Actions** tab in your GitHub repository
2. Click on the latest workflow run
3. Scroll down to **Artifacts** section
4. Download:
   - `annotated-video` - The processed video with bounding boxes
   - `detection-data` - JSON file with all detection information

The workflow summary also displays:
- Total frames processed
- Total objects detected
- Breakdown of detected object classes

## Downloading Test Videos Locally

Use the provided script to download free test videos:

```bash
./download_test_videos.sh
```

This downloads three test videos to the `test_videos/` directory:
1. **people_walking.mp4** - People walking (good for person detection)
2. **traffic.mp4** - Traffic and cars (good for vehicle detection)
3. **bicycle.mp4** - Person with bicycle (good for multi-object detection)

## Running Tests Locally

### Basic video processing:
```bash
python main.py --source test_videos/people_walking.mp4
```

### With output video and data export (for CI/CD testing):
```bash
python main.py \
  --source test_videos/traffic.mp4 \
  --output output.mp4 \
  --data-output detections.json \
  --headless \
  --verbose
```

### Test different models:
```bash
# Faster, less accurate (nano model - default)
python main.py --source test_videos/traffic.mp4 --model yolo11n.pt

# More accurate, slower (medium model)
python main.py --source test_videos/traffic.mp4 --model yolo11m.pt
```

## Free Video Sources

Here are excellent sources for free test videos:

### 1. Pixabay (Free, No Attribution Required)
- **URL:** https://pixabay.com/videos/
- **License:** Free for commercial use, no attribution required
- **Good for:** Various scenarios - people, traffic, animals, sports

**Search suggestions:**
- "people walking"
- "traffic"
- "street"
- "pedestrians"
- "cars driving"
- "bicycle"
- "skateboard"
- "dog"
- "cat"

### 2. Pexels (Free, No Attribution Required)
- **URL:** https://www.pexels.com/videos/
- **License:** Free for commercial use, no attribution required
- **Good for:** High-quality professional footage

**Search suggestions:**
- "city street"
- "busy intersection"
- "parking lot"
- "shopping mall"
- "park"

### 3. Coverr (Free, No Attribution Required)
- **URL:** https://coverr.co/
- **License:** Free for commercial use
- **Good for:** Short clips, professional quality

### 4. Sample Videos (Technical Test Videos)
- **URL:** https://sample-videos.com/
- **License:** Free sample videos
- **Good for:** Various resolutions and formats

## Best Practices for Test Videos

### Video Characteristics for Good Testing:

1. **Duration:** 5-30 seconds (short enough for quick tests, long enough for meaningful data)
2. **Resolution:** 720p or 1080p (higher resolution = slower processing)
3. **Content:**
   - Multiple object types for comprehensive testing
   - Different lighting conditions
   - Various object sizes (near and far)
   - Movement (not static scenes)

### Recommended Test Scenarios:

1. **Urban Street Scene:**
   - Tests: person, car, bus, truck, bicycle, motorcycle, traffic light
   - Good for multi-object detection

2. **Park or Outdoor:**
   - Tests: person, dog, bicycle, backpack, sports ball
   - Good for tracking and varying distances

3. **Indoor Shopping/Mall:**
   - Tests: person, handbag, backpack, cell phone
   - Good for crowded scenes

4. **Traffic Intersection:**
   - Tests: car, truck, bus, motorcycle, bicycle, traffic light, stop sign
   - Good for high-speed object detection

5. **Sports/Activities:**
   - Tests: person, sports ball, tennis racket, skateboard
   - Good for fast-moving objects

## Creating Your Own Test Video

### From Webcam (Short Clip):
```bash
# Record 10 seconds from webcam
python -c "
import cv2
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('my_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
for _ in range(300):  # 300 frames at 30fps = 10 seconds
    ret, frame = cap.read()
    if ret:
        out.write(frame)
cap.release()
out.release()
"
```

### Download from YouTube (Legal Content Only):
```bash
# Install yt-dlp
pip install yt-dlp

# Download a short clip (with permission/legal content)
yt-dlp -f "best[height<=720]" --download-sections "*0:00-0:30" "VIDEO_URL" -o "test_clip.mp4"
```

## COCO Object Classes

YOLO11 detects 80 object classes from the COCO dataset:

**Common objects detected:**
- **People:** person
- **Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Traffic:** traffic light, fire hydrant, stop sign, parking meter
- **Animals:** bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Sports:** frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, tennis racket
- **Furniture:** chair, couch, bed, dining table, toilet
- **Electronics:** tv, laptop, mouse, remote, keyboard, cell phone
- **Kitchen:** bottle, wine glass, cup, fork, knife, spoon, bowl
- **Food:** banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

For a complete list, see: https://docs.ultralytics.com/datasets/detect/coco/

## Analyzing Detection Results

The JSON output file contains:
```json
{
  "metadata": {
    "video_source": "test_videos/traffic.mp4",
    "model": "yolo11n.pt",
    "confidence_threshold": 0.5,
    "timestamp": "2025-12-22T10:30:00",
    "resolution": "1280x720",
    "fps": 30
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "objects": ["car", "car", "person"],
      "unique_objects": ["car", "person"],
      "object_count": 3
    }
  ],
  "summary": {
    "total_frames": 150,
    "total_detections": 450,
    "class_counts": {
      "car": 300,
      "person": 100,
      "truck": 50
    }
  }
}
```

### Useful Analysis Commands:

```bash
# Count total detections
cat detections.json | jq '.summary.total_detections'

# Get most detected class
cat detections.json | jq '.summary.class_counts | to_entries | max_by(.value) | .key'

# Count frames with detections
cat detections.json | jq '[.frames[] | select(.object_count > 0)] | length'

# Average objects per frame
cat detections.json | jq '[.frames[].object_count] | add / length'
```

## Troubleshooting

### Video Won't Process
- Check video codec (use MP4 with H.264)
- Verify file isn't corrupted: `ffmpeg -v error -i video.mp4 -f null -`
- Try re-encoding: `ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4`

### No Objects Detected
- Lower confidence threshold: `--confidence 0.3`
- Use better model: `--model yolo11m.pt`
- Check video content (YOLO only detects COCO classes)

### Slow Processing
- Use smaller model: `--model yolo11n.pt`
- Reduce video resolution
- Use GPU if available

## License Notes

**Important:** When using test videos:
- Ensure you have the right to use the video
- For GitHub workflows, use only freely licensed content
- Pixabay and Pexels videos are safe for automated testing
- Never include copyrighted content in the repository

---

**Last Updated:** 2025-12-22
**Related Files:**
- `.github/workflows/video-test.yml` - GitHub Actions workflow
- `download_test_videos.sh` - Script to download test videos
- `main.py` - Main detection application
