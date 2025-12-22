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
├── main.py              # Main application entry point (300 lines)
├── requirements.txt     # Python dependencies with version specs
├── README.md           # User-facing documentation
└── CLAUDE.md           # This file - AI assistant documentation
```

### File Descriptions

#### `main.py`
Modern object-oriented application with full CLI support:
- **Module Docstring** (lines 1-9): File-level documentation
- **Imports** (lines 11-19): Modern imports with type hints support
- **Logging Configuration** (lines 22-28): Structured logging setup
- **ObjectDetector Class** (lines 31-190): Main detection logic
  - `__init__()` (lines 34-67): Initialize with configurable parameters
  - `detect_objects()` (lines 69-120): Process frames and annotate
  - `run_detection()` (lines 122-190): Main detection loop with error handling
- **Argument Parser** (lines 193-250): Full CLI argument handling
- **Main Entry Point** (lines 253-299): Application initialization and execution

## Code Architecture

### Modern Design Patterns (2025)

The codebase follows modern Python best practices:
- **Type Hints**: Full type annotations throughout (PEP 484)
- **Class-Based Design**: OOP with ObjectDetector class
- **Dependency Injection**: Configurable parameters via constructor
- **Proper Logging**: Structured logging instead of print statements
- **Error Handling**: Try-except blocks with proper cleanup
- **Resource Management**: Finally blocks for cleanup
- **CLI Interface**: argparse for professional command-line handling
- **Docstrings**: Complete function and class documentation

### Core Components

1. **ObjectDetector Class** (`main.py:31-190`)
   - Encapsulates all detection logic
   - Configurable via constructor parameters
   - Thread-safe and reusable

   **Key Methods**:
   - `__init__()`: Initialize model with custom settings
   - `detect_objects()`: Process single frame
   - `run_detection()`: Main detection loop

2. **Model Initialization** (`main.py:61-67`)
   ```python
   self.model = YOLO(model_name)
   ```
   - Loads YOLOv8 model variant (default: nano)
   - Error handling for model loading failures
   - Logs device information (CPU/GPU)
   - Model downloaded automatically on first run

3. **Detection Pipeline** (`detect_objects()` at `main.py:69-120`)
   - **Input**: `np.ndarray` (BGR frame)
   - **Output**: `Tuple[np.ndarray, List[str]]` (annotated frame, labels)
   - **Process**:
     - Runs YOLO inference with `verbose=False`
     - Filters by confidence threshold (configurable)
     - Draws bounding boxes with custom colors/thickness
     - Adds labels with confidence scores (e.g., "person 0.87")
   - Type-safe with full annotations

4. **Video Capture Loop** (`run_detection()` at `main.py:122-190`)
   - Opens video source (camera index or file path)
   - Validates video source accessibility
   - Logs video properties (resolution, FPS)
   - Processes frames continuously
   - Handles keyboard interrupts gracefully
   - Proper resource cleanup in finally block
   - Multiple exit methods: 'q' key, ESC key, or Ctrl+C

5. **CLI Argument Parsing** (`main.py:193-250`)
   - Professional argparse implementation
   - Supports all common use cases
   - Validation of inputs (confidence range, color format)
   - Help text with examples

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

**Basic Usage** (default webcam):
```bash
python main.py
```

**With Command-Line Arguments**:
```bash
# Use different camera
python main.py --source 1

# Use video file
python main.py --source video.mp4

# Use different model
python main.py --model yolov8s.pt

# Adjust confidence threshold
python main.py --confidence 0.7

# Custom bounding box color (red)
python main.py --color 0,0,255

# Verbose logging
python main.py --verbose

# Combined options
python main.py --source video.mp4 --model yolov8m.pt --confidence 0.6 --color 255,0,0
```

**View Help**:
```bash
python main.py --help
```

### Exit Methods
- Press **'q'** key
- Press **ESC** key
- Press **Ctrl+C** (keyboard interrupt)

### Testing Changes

1. **Modify code** in `main.py`
2. **Run with test parameters**:
   ```bash
   python main.py --verbose --confidence 0.3
   ```
3. **Check logs** for structured output
4. **Verify** webcam feed and detection results
5. **Monitor** console for detection logs (every 30 frames)

### Common Development Tasks

#### Changing Detection Threshold
**Via CLI** (recommended):
```bash
python main.py --confidence 0.7
```

**Via Code** (line 37):
```python
confidence_threshold: float = 0.5  # Change default
```
- Lower (e.g., 0.3): More detections, more false positives
- Higher (e.g., 0.7): Fewer detections, higher precision

#### Switching YOLO Models
**Via CLI** (recommended):
```bash
python main.py --model yolov8m.pt
```

**Via Code** (line 36):
```python
model_name: str = "yolov8n.pt"  # Change default
```

Available models:
- `yolov8n.pt`: Nano (6MB, fastest, good accuracy)
- `yolov8s.pt`: Small (22MB, fast, better accuracy)
- `yolov8m.pt`: Medium (52MB, balanced)
- `yolov8l.pt`: Large (87MB, slow, excellent)
- `yolov8x.pt`: Extra large (136MB, slowest, best)

#### Changing Video Source
**Via CLI** (recommended):
```bash
python main.py --source 1              # Second camera
python main.py --source video.mp4      # Video file
python main.py --source rtsp://...     # IP camera
```

**Via Code**: Modify `video_source` parameter in `run_detection()` call

#### Customizing Visualization
**Via CLI**:
```bash
python main.py --color 0,0,255 --thickness 3  # Red, thicker boxes
```

**Via Code**: Modify ObjectDetector constructor parameters (lines 38-41):
```python
box_color: Tuple[int, int, int] = (0, 255, 0)  # BGR color
box_thickness: int = 2                          # Line thickness
font_scale: float = 0.6                        # Text size
font_thickness: int = 2                        # Text thickness
```

#### Adding Custom Features
The class-based design makes it easy to extend:

```python
class CustomDetector(ObjectDetector):
    def detect_objects(self, frame):
        # Call parent method
        annotated_frame, objects = super().detect_objects(frame)

        # Add custom logic
        # ... your code here ...

        return annotated_frame, objects
```

## Code Conventions

### Style Guidelines (PEP 8 + Modern Python)
- **Indentation**: 4 spaces (strictly enforced)
- **Naming**:
  - snake_case for functions, variables, methods
  - PascalCase for classes (e.g., ObjectDetector)
  - UPPER_CASE for constants
- **Line Length**: 88-100 characters (Black formatter style)
- **Imports**: Grouped and ordered (stdlib → third-party → local)
- **Quotes**: Double quotes for docstrings, single quotes preferred for strings
- **Type Hints**: Required for all function signatures and class attributes
- **Docstrings**: Google-style docstrings for all public methods

### Current Patterns (2025)

**Object-Oriented Design**:
- Encapsulation via ObjectDetector class
- Dependency injection through constructor
- Single Responsibility Principle per method

**Type Safety**:
```python
def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
```
- Full type annotations
- Modern union syntax: `int | str` instead of `Union[int, str]`

**Logging Instead of Print**:
```python
logger.info(f"Video opened: {width}x{height} @ {fps}fps")
```
- Structured logging with levels (INFO, DEBUG, WARNING, ERROR)
- Timestamps and formatted output

**Error Handling**:
- Try-except blocks with specific exception types
- Finally blocks for resource cleanup
- Proper error messages via logging
- RuntimeError for unrecoverable errors

**Resource Management**:
```python
try:
    # Main loop
finally:
    cap.release()
    cv2.destroyAllWindows()
```
- Guaranteed cleanup even on errors
- No resource leaks

**Command-Line Interface**:
- Professional argparse with help text
- Input validation before processing
- Exit codes (0 for success, 1 for errors)

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

### Adding Configuration File Support

The current CLI-based configuration is flexible, but you could add config file support:

```python
from pathlib import Path
import yaml  # or json

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# In main():
if args.config:
    config = load_config(Path(args.config))
    detector = ObjectDetector(**config)
```

Example `config.yaml`:
```yaml
model_name: yolov8m.pt
confidence_threshold: 0.6
box_color: [0, 255, 0]
box_thickness: 2
font_scale: 0.7
```

## Best Practices for AI Assistants

### When Modifying Code

1. **Always Read First**: Use Read tool on `main.py` before making changes
2. **Understand the Class Structure**: Code is object-oriented - understand ObjectDetector class
3. **Maintain Type Hints**: Add type annotations to all new functions/methods
4. **Use Logging**: Never use `print()` - always use `logger.info/debug/warning/error()`
5. **Follow Error Handling Patterns**: Use try-except-finally blocks
6. **Test CLI Arguments**: Verify argparse integration after changes
7. **Update Docstrings**: Maintain Google-style docstrings

### Common Modification Scenarios

**Scenario: Add new detection filter**
- **Location**: Inside `ObjectDetector.detect_objects()` method
- **Line**: After confidence check at `main.py:94`
- **Example**: Filter by specific object classes
```python
if confidence > self.confidence_threshold and label in ['person', 'car']:
```

**Scenario: Add new CLI argument**
- **Location**: `parse_arguments()` function at `main.py:193-250`
- **Steps**:
  1. Add argument to parser
  2. Update `main()` to handle new argument
  3. Pass to ObjectDetector constructor or method
  4. Update type hints

**Scenario: Change UI/display**
- **Location**: `ObjectDetector.detect_objects()` for per-object annotations (lines 98-118)
- **Location**: `ObjectDetector.run_detection()` for window settings (line 125)
- **Example**: Add FPS counter to display

**Scenario: Performance optimization**
- **Approaches**:
  - Frame skipping in `run_detection()` loop
  - Resolution reduction before inference
  - Model variant change (default in constructor)
  - Batch processing (modify `detect_objects()`)

**Scenario: Add new method to ObjectDetector**
- **Requirements**:
  - Type hints for parameters and return value
  - Google-style docstring
  - Proper error handling
  - Use self.logger for logging

**Scenario: Extend ObjectDetector class**
- **Recommended**: Use inheritance
```python
class AdvancedDetector(ObjectDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization

    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        # Override with custom logic
        pass
```

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
**Primary File**: main.py (300 lines)
**Python Version**: 3.8+ (modern type hints require 3.10+ for `int | str` syntax)
**Architecture**: Object-Oriented with CLI support
**Code Quality**: Production-ready with full type hints, logging, and error handling
