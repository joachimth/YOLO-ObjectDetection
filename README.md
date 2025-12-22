# YOLO Object Detection 🎯

Real-time object detection using YOLO11 and webcam feed. Detect and track 80 different object classes with state-of-the-art accuracy and speed.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)

## Features ✨

- **Real-time Detection**: Process webcam feed at high FPS with live FPS counter
- **80 Object Classes**: Detect people, vehicles, animals, and common objects
- **Object Tracking**: Track objects between frames with unique IDs
- **Visual Feedback**: Bounding boxes and labels drawn on detected objects
- **Statistics Display**: Live object counting and detection statistics on screen
- **Alert System**: Customizable alerts for specific object classes
- **Confidence Filtering**: Configurable threshold to reduce false positives
- **Multiple YOLO Models**: Support for nano, small, medium, large, and extra-large variants
- **Configuration Files**: YAML-based configuration with CLI override support
- **Professional CLI**: Full command-line interface with argparse
- **Video Output**: Save annotated videos with detections
- **Data Export**: Export detection data to JSON format
- **Headless Mode**: Run without display for CI/CD environments
- **Automated Testing**: GitHub Actions workflow for video testing

## Quick Start 🚀

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- 6MB disk space for YOLO model weights

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/joachimth/YOLO-ObjectDetection.git
   cd YOLO-ObjectDetection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **For CI/CD environments** (no display):
   ```bash
   pip install opencv-python-headless>=4.10.0 ultralytics>=8.3.0 pyyaml>=6.0
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

The YOLO11 model will be downloaded automatically on first run (~6MB).

## Usage 📖

### Basic Usage

Simply run the script to start real-time object detection:

```bash
python main.py
```

- The webcam feed will open in a window titled "AI Vision - YOLO Object Detection"
- Detected objects will be highlighted with green bounding boxes
- Object labels will appear above each detection
- FPS counter displayed in top-left (yellow)
- Statistics shown on left side (white)
- Console will print detected objects periodically
- Press 'q' or ESC to exit

### Command-Line Options

Full CLI support with professional argument parsing:

```bash
# View all available options
python main.py --help

# Use different camera
python main.py --source 1

# Use video file
python main.py --source video.mp4

# Use different model
python main.py --model yolo11s.pt

# Adjust confidence threshold
python main.py --confidence 0.7

# Custom bounding box color (red in BGR format)
python main.py --color 0,0,255

# Enable object tracking
python main.py --tracking

# Set up alerts for specific objects
python main.py --alert person car dog

# Disable FPS counter
python main.py --no-fps

# Disable statistics display
python main.py --no-stats

# Combine multiple options
python main.py --source video.mp4 --model yolo11m.pt --confidence 0.6 --tracking --alert person
```

### Video Processing & Export

Process video files and export results:

```bash
# Process video and save annotated output
python main.py --source input.mp4 --output output.mp4

# Export detection data to JSON
python main.py --source input.mp4 --data-output detections.json

# Headless mode (no display window) for CI/CD
python main.py --source input.mp4 --output output.mp4 --data-output detections.json --headless

# Complete example: process video, save results, headless
python main.py \
  --source test_video.mp4 \
  --model yolo11n.pt \
  --confidence 0.5 \
  --output annotated_video.mp4 \
  --data-output detection_data.json \
  --headless \
  --verbose
```

**Detection Data Format** (JSON):
```json
{
  "metadata": {
    "video_source": "test.mp4",
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
      "objects": ["car", "person", "person"],
      "unique_objects": ["car", "person"],
      "object_count": 3
    }
  ],
  "summary": {
    "total_frames": 150,
    "total_detections": 450,
    "class_counts": {
      "car": 300,
      "person": 150
    }
  }
}
```

### Configuration File Usage

The application automatically loads `config.yaml` if it exists:

```bash
# Copy example config and customize
cp config.example.yaml config.yaml

# Edit config.yaml with your settings
# Then just run (config.yaml is auto-loaded)
python main.py

# Or use a custom config file
python main.py --config custom.yaml

# CLI arguments override config file settings
python main.py --confidence 0.8
```

Default `config.yaml`:
```yaml
model_name: "yolo11n.pt"
confidence_threshold: 0.5
video_source: 0
enable_tracking: false
show_fps: true
show_stats: true
alert_objects: []
box_color: [0, 255, 0]
```

See `config.example.yaml` for detailed documentation of all YOLO11 models and settings.

### Available Models (YOLO11)

| Model | Size | Parameters | CPU Speed | Accuracy | Use Case |
|-------|------|------------|-----------|----------|----------|
| yolo11n.pt | 6MB | 2.6M | 80+ FPS | Good | Real-time on CPU, edge devices |
| yolo11s.pt | 18MB | 9.4M | 50+ FPS | Better | Balanced performance |
| yolo11m.pt | 40MB | 20.1M | 30+ FPS | Great | Higher accuracy, GPU recommended |
| yolo11l.pt | 50MB | 25.3M | 15+ FPS | Excellent | Maximum accuracy, GPU required |
| yolo11x.pt | 56MB | 56.9M | 8+ FPS | Best | Research/offline, powerful GPU |

**Why YOLO11?**
- 22% fewer parameters than YOLOv8 with higher accuracy
- Better stability under domain shift
- Improved small-object detection
- Production-ready (released Sept 2024)

## Detected Object Classes 🏷️

The model can detect 80 COCO dataset classes:

**People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Indoor Objects**: chair, couch, potted plant, bed, dining table, toilet, TV, laptop, mouse, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors

**Outdoor Objects**: traffic light, fire hydrant, stop sign, parking meter, bench

**Sports**: sports ball, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Food**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

And more! Full list available in the code via `model.names`.

## Technical Details 🔧

### Architecture

- **Detection Model**: YOLO11 (You Only Look Once version 11)
- **Framework**: Ultralytics
- **Video Processing**: OpenCV
- **Default Model**: YOLO11 Nano (optimized for speed)
- **Inference**: Real-time single-shot detection

### Performance

- **FPS**: 30+ fps on modern CPUs with nano model
- **Latency**: <50ms per frame
- **GPU Support**: Automatic CUDA acceleration if available
- **Memory**: ~200MB RAM usage

### System Requirements

**Minimum**:
- CPU: Dual-core 2.0 GHz
- RAM: 4GB
- Python: 3.8+

**Recommended**:
- CPU: Quad-core 3.0 GHz or NVIDIA GPU (CUDA support)
- RAM: 8GB
- Python: 3.10+

## Automated Testing 🤖

The repository includes a GitHub Actions workflow that automatically tests YOLO detection on every push:

### How It Works

1. **Triggers**: Runs on push to `main` or `claude/**` branches, and on pull requests
2. **Process**: Downloads a test video, runs detection, exports results
3. **Artifacts**: Saves annotated video and detection data for download
4. **Summary**: Displays detection statistics in workflow summary

### Viewing Test Results

After a workflow run:
1. Go to **Actions** tab in GitHub
2. Click the latest workflow run
3. View the **Summary** with detection statistics
4. Download artifacts:
   - `annotated-video` - Processed video with bounding boxes
   - `detection-data` - JSON file with all detections

### Running Tests Locally

Download test videos:
```bash
./download_test_videos.sh
```

Run detection on test video:
```bash
python main.py \
  --source test_videos/people_walking.mp4 \
  --output output.mp4 \
  --data-output detections.json \
  --headless
```

See **[TEST_VIDEOS.md](TEST_VIDEOS.md)** for:
- Free test video sources (Pixabay, Pexels)
- Best practices for test videos
- How to analyze detection results
- Complete testing guide

## Advanced Usage 🎓

### GPU Acceleration

For CUDA-enabled GPUs, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The code will automatically use GPU if available.

### Customizing Visualization

**Change bounding box color** (line 23):
```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                                         ^^^^^^^^^^^
#                                         BGR color: (Blue, Green, Red)
```

Common colors:
- Green: `(0, 255, 0)`
- Red: `(0, 0, 255)`
- Blue: `(255, 0, 0)`
- Yellow: `(0, 255, 255)`
- Purple: `(255, 0, 255)`

**Change text style** (line 24):
```python
cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                                      ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^               ^
#                                      Font family               Size              Thickness
```

## Troubleshooting 🔍

### Camera Not Opening

**Issue**: `cv2.VideoCapture(0)` fails to open camera

**Solutions**:
- Check if another application is using the camera
- Try different device index: `cv2.VideoCapture(1)`
- On Linux, check permissions: `sudo usermod -a -G video $USER`
- Verify camera is connected: `ls /dev/video*`

### Model Download Fails

**Issue**: YOLO11 model doesn't download

**Solutions**:
- Check internet connection
- Manually download from [Ultralytics GitHub](https://github.com/ultralytics/assets/releases)
- Place `yolo11n.pt` in project directory
- Check firewall settings

### Low FPS / Laggy Performance

**Issue**: Detection is slow

**Solutions**:
- Use nano model: `YOLO("yolo11n.pt")` (default)
- Reduce frame resolution before processing
- Skip frames: process every 2nd or 3rd frame
- Use GPU acceleration (see Advanced Usage)
- Close other applications

### No Objects Detected

**Issue**: Objects in frame but not detected

**Solutions**:
- Lower confidence threshold (e.g., 0.3)
- Ensure adequate lighting
- Try larger model for better accuracy
- Check if object is in COCO dataset classes
- Move camera closer to objects

## Project Structure 📁

```
YOLO-ObjectDetection/
├── main.py                      # Main application with CLI and tracking
├── requirements.txt             # Python dependencies
├── config.yaml                  # Default configuration (auto-loaded)
├── config.example.yaml          # Example config with detailed docs
├── download_test_videos.sh      # Script to download test videos
├── .github/
│   └── workflows/
│       └── video-test.yml       # GitHub Actions workflow
├── README.md                    # This file
├── TEST_VIDEOS.md               # Test video guide
├── CLAUDE.md                    # AI assistant documentation
└── yolo11n.pt                   # Model weights (auto-downloaded)
```

## Contributing 🤝

Contributions are welcome! Recently completed features:

- [x] Add command-line arguments for configuration
- [x] Add FPS counter display
- [x] Object tracking between frames
- [x] Object counting and statistics
- [x] Alert system for specific objects
- [x] Configuration file support
- [x] Video output (save annotated videos)
- [x] Export detection data to JSON
- [x] Headless mode for CI/CD
- [x] GitHub Actions workflow for automated testing

Areas for future improvement:

- [ ] Multi-camera support
- [ ] Web interface for remote viewing
- [ ] Export detection data to CSV format
- [ ] Custom alert actions (email, webhook, sound)
- [ ] Region of interest (ROI) detection
- [ ] Heat map generation for object movement
- [ ] Performance benchmarking tools
- [ ] Docker containerization

## License 📄

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments 🙏

- **Ultralytics** for the excellent YOLO11 implementation
- **OpenCV** community for computer vision tools
- **COCO Dataset** for training data and class definitions

## Resources 📚

- [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [COCO Dataset](https://cocodataset.org/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## Support 💬

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to CLAUDE.md for development guidance

---

**Made with ❤️ using YOLO11 and OpenCV**
