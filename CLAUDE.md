# CLAUDE.md - AI Assistant Documentation

## Project Overview

**YOLO-ObjectDetection** is a real-time object detection application using YOLOv8 (You Only Look Once) from Ultralytics. The project captures video from a webcam and performs real-time object detection, drawing bounding boxes and labels on detected objects.

### Key Technologies
- **YOLOv8**: State-of-the-art object detection model (yolov8n.pt - nano variant)
- **OpenCV (cv2)**: Computer vision library for video capture and display
- **Ultralytics**: YOLOv8 implementation and model management

## Repository Structure

```
YOLO-ObjectDetection/
├── main.py           # Main application entry point
└── CLAUDE.md         # This file - AI assistant documentation
```

### File Descriptions

#### `main.py`
The single-file application containing all functionality:
- **Model Loading** (lines 5): Loads YOLOv8 nano model
- **Object Detection** (lines 8-26): `detect_objects()` function processes frames
- **Main Loop** (lines 29-44): `main()` function handles video capture and display
- **Entry Point** (lines 47-48): Standard Python execution guard

## Code Architecture

### Core Components

1. **Model Initialization**
   ```python
   model = YOLO("yolov8n.pt")
   ```
   - Uses YOLOv8 nano model (smallest, fastest variant)
   - Model is loaded once at module level for efficiency
   - On first run, downloads the model weights automatically

2. **Detection Pipeline** (`detect_objects()`)
   - **Input**: Video frame (numpy array)
   - **Output**: Annotated frame + list of detected object labels
   - **Process**:
     - Runs YOLO inference on frame
     - Filters detections by confidence threshold (>0.5)
     - Draws bounding boxes (green, 2px thickness)
     - Adds text labels above boxes
   - Located at: `main.py:8-26`

3. **Video Capture Loop** (`main()`)
   - Opens webcam (device index 0)
   - Continuously captures frames
   - Displays results in "AI Vision" window
   - Prints detected objects to console
   - Located at: `main.py:29-44`

## Dependencies

### Required Packages
```
opencv-python (cv2)  # Video capture and image processing
ultralytics          # YOLOv8 model implementation
```

### Installation
```bash
pip install opencv-python ultralytics
```

### Model Files
- **yolov8n.pt**: Downloaded automatically on first run
- Model supports 80 COCO dataset classes (person, car, dog, etc.)

## Development Workflows

### Running the Application
```bash
python main.py
```
- Requires webcam/camera device
- Press any key to exit (handled by cv2.waitKey)
- Displays real-time detection in window

### Testing Changes
1. Modify code in `main.py`
2. Run application to test
3. Verify webcam feed and detection results
4. Check console output for detected objects

### Common Development Tasks

#### Changing Detection Threshold
Current threshold: 0.5 at `main.py:17`
```python
if confidence > 0.5:  # Adjust this value
```
- Lower (e.g., 0.3): More detections, more false positives
- Higher (e.g., 0.7): Fewer detections, higher precision

#### Switching YOLO Models
At `main.py:5`:
- `yolov8n.pt`: Nano (fastest, least accurate)
- `yolov8s.pt`: Small
- `yolov8m.pt`: Medium
- `yolov8l.pt`: Large
- `yolov8x.pt`: Extra large (slowest, most accurate)

#### Changing Video Source
At `main.py:30`:
```python
cap = cv2.VideoCapture(0)  # 0 = default webcam
```
- `1`, `2`, etc.: Other camera devices
- `"video.mp4"`: Video file path
- `"http://..."`: IP camera stream

#### Customizing Visualization
Bounding box styling at `main.py:23`:
```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                                         ^^^^^^^^^^^  ^
#                                         Color (BGR)  Thickness
```

Text styling at `main.py:24`:
```python
cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^  ^^^^^^^^^^^  ^
#                                        Font                           Size Color        Thickness
```

## Code Conventions

### Style Guidelines
- **Indentation**: 4 spaces
- **Naming**: snake_case for functions and variables
- **Line Length**: Generally under 100 characters
- **Imports**: Grouped at top (stdlib, third-party)

### Current Patterns
- Global model initialization for efficiency
- Single-responsibility functions
- Clear variable names (frame, results, detected_objects)
- Confidence filtering before processing
- Clean resource management (cap.release(), cv2.destroyAllWindows())

### Error Handling
Currently minimal - consider adding:
- Camera access failure handling
- Model loading error handling
- Frame read failure handling (partially present with `if not ret`)

## Extension Points

### Adding New Features

1. **Save Detections to File**
   - Add file I/O in main loop
   - Log detections with timestamps

2. **Object Counting**
   - Track object counts per class
   - Display statistics on frame

3. **Custom Alert System**
   - Trigger alerts for specific objects
   - Send notifications (email, webhook)

4. **Performance Metrics**
   - Add FPS counter
   - Display inference time

5. **Video Recording**
   - Use cv2.VideoWriter to save annotated video
   - Add start/stop recording controls

### Adding Configuration
Consider creating a config structure for:
- Model selection
- Confidence threshold
- Video source
- Display settings
- Color schemes

Example:
```python
CONFIG = {
    'model': 'yolov8n.pt',
    'confidence_threshold': 0.5,
    'video_source': 0,
    'box_color': (0, 255, 0),
    'box_thickness': 2
}
```

## Best Practices for AI Assistants

### When Modifying Code

1. **Always Read First**: Use Read tool on `main.py` before making changes
2. **Understand Context**: The file is small - read the entire file to understand flow
3. **Test Considerations**: Changes affect real-time video processing
4. **Dependencies**: Be aware that cv2 and ultralytics must be installed

### Common Modification Scenarios

**Scenario: Add new detection filter**
- Location: Inside `detect_objects()` function
- Add filtering logic after confidence check at `main.py:17`

**Scenario: Change UI/display**
- Location: Inside `detect_objects()` for per-object annotations
- Location: Inside `main()` for overall window/display settings

**Scenario: Performance optimization**
- Consider: Frame skipping, resolution reduction, model variant
- Location: `main()` loop for frame handling, line 5 for model selection

**Scenario: Add command-line arguments**
- Use argparse module
- Parse arguments before `main()` call
- Pass configuration to functions

### Safety Considerations

1. **Camera Privacy**: Code accesses webcam - ensure user is aware
2. **Resource Cleanup**: Always maintain proper release of camera and windows
3. **Model Downloads**: First run downloads ~6MB model file
4. **Performance**: Nano model is optimized for speed; larger models need GPU

### Git Workflow

- **Branch**: Always work on `claude/add-claude-documentation-*` branches
- **Commits**: Clear, descriptive messages
- **Push**: Use `git push -u origin <branch-name>` with retry logic
- **Testing**: Run the application before committing

### Documentation Updates

When making significant changes, update:
- This CLAUDE.md file with new patterns/conventions
- Add comments in code for complex logic
- Update function docstrings if adding new functions

## Troubleshooting Guide

### Common Issues

**Camera not opening**
- Check if another application is using the camera
- Try different device index (0, 1, 2)
- Verify camera permissions

**Model download fails**
- Check internet connection
- Manually download from Ultralytics GitHub
- Verify disk space (~6MB needed)

**Poor detection performance**
- Increase confidence threshold to reduce false positives
- Upgrade to larger model (yolov8s.pt or yolov8m.pt)
- Ensure adequate lighting

**Low FPS**
- Use nano model (yolov8n.pt)
- Reduce frame resolution
- Skip frames (process every N frames)
- Use GPU if available (CUDA-enabled PyTorch)

## Resources

### Official Documentation
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### YOLO Model Variants
- YOLOv8n: 3.2M parameters, fastest
- YOLOv8s: 11.2M parameters
- YOLOv8m: 25.9M parameters
- YOLOv8l: 43.7M parameters
- YOLOv8x: 68.2M parameters, most accurate

### Supported Object Classes
80 COCO classes including:
- Persons, vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (dog, cat, bird, horse, cow, etc.)
- Common objects (chair, bottle, laptop, phone, etc.)

Full list available via: `model.names` dictionary

---

**Last Updated**: 2025-12-22
**Repository**: YOLO-ObjectDetection
**Primary File**: main.py (49 lines)
