"""
YOLO Object Detection Application

Real-time object detection using YOLOv8 and webcam feed.
Detects and tracks objects from 80 COCO dataset classes.

Author: YOLO-ObjectDetection
Updated: 2025-12-22
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Real-time object detection using YOLOv8."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2,
        enable_tracking: bool = False,
        show_fps: bool = True,
        show_stats: bool = True,
        alert_objects: Optional[Set[str]] = None
    ):
        """
        Initialize the Object Detector.

        Args:
            model_name: YOLOv8 model variant (yolov8n/s/m/l/x.pt)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            box_color: BGR color tuple for bounding boxes
            box_thickness: Thickness of bounding box lines
            font_scale: Scale of text labels
            font_thickness: Thickness of text
            enable_tracking: Enable object tracking between frames
            show_fps: Display FPS counter on frame
            show_stats: Display object statistics on frame
            alert_objects: Set of object class names to trigger alerts
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.enable_tracking = enable_tracking
        self.show_fps = show_fps
        self.show_stats = show_stats
        self.alert_objects = alert_objects or set()

        # Statistics tracking
        self.total_detections = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.frame_times: List[float] = []
        self.max_frame_times = 30  # Keep last 30 frame times for FPS calculation

        logger.info(f"Loading YOLO model: {model_name}")
        try:
            self.model = YOLO(model_name)
            logger.info(f"Model loaded successfully. Device: {self.model.device}")
            if self.enable_tracking:
                logger.info("Object tracking enabled")
            if self.alert_objects:
                logger.info(f"Alert objects: {', '.join(self.alert_objects)}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect_objects(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Detect objects in a frame and draw annotations.

        Args:
            frame: Input image frame (BGR format)

        Returns:
            Tuple of (annotated_frame, list_of_detected_labels)
        """
        # Run YOLO inference (with or without tracking)
        if self.enable_tracking:
            results = self.model.track(frame, verbose=False, persist=True, conf=self.confidence_threshold)
        else:
            results = self.model(frame, verbose=False)

        detected_objects = []
        alerted_objects = []

        # Process each detection
        for r in results:
            for box in r.boxes:
                # Extract detection information
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Filter by confidence threshold
                if confidence > self.confidence_threshold:
                    label = self.model.names[class_id]
                    detected_objects.append(label)

                    # Update statistics
                    self.total_detections += 1
                    self.class_counts[label] += 1

                    # Check for alerts
                    if label in self.alert_objects and label not in alerted_objects:
                        alerted_objects.append(label)
                        logger.warning(f"🚨 ALERT: {label} detected!")

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Use different color for alerted objects
                    box_color = (0, 0, 255) if label in self.alert_objects else self.box_color

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        self.box_thickness
                    )

                    # Draw label with confidence and tracking ID
                    if self.enable_tracking and hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                        label_text = f"ID:{track_id} {label} {confidence:.2f}"
                    else:
                        label_text = f"{label} {confidence:.2f}"

                    cv2.putText(
                        frame,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        box_color,
                        self.font_thickness
                    )

        return frame, detected_objects

    def calculate_fps(self) -> float:
        """Calculate FPS based on recent frame times."""
        if len(self.frame_times) < 2:
            return 0.0

        # Calculate average time per frame
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter on frame."""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),  # Yellow color
            2
        )

    def draw_stats(self, frame: np.ndarray, current_objects: List[str]) -> None:
        """Draw object statistics on frame."""
        # Count current frame objects
        current_counts = defaultdict(int)
        for obj in current_objects:
            current_counts[obj] += 1

        # Prepare stats text
        y_offset = 60
        cv2.putText(
            frame,
            f"Total Detections: {self.total_detections}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        y_offset += 25
        cv2.putText(
            frame,
            f"Current Frame: {len(current_objects)} objects",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Show top 5 most detected classes
        y_offset += 25
        top_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_classes:
            cv2.putText(
                frame,
                "Top Detections:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            for i, (cls, count) in enumerate(top_classes):
                y_offset += 20
                current = current_counts.get(cls, 0)
                cv2.putText(
                    frame,
                    f"  {cls}: {count} (now: {current})",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1
                )

    def run_detection(
        self,
        video_source: int | str = 0,
        window_name: str = "AI Vision - YOLO Object Detection"
    ) -> None:
        """
        Run real-time object detection on video source.

        Args:
            video_source: Camera device index or video file path
            window_name: Name of the display window
        """
        logger.info(f"Opening video source: {video_source}")

        # Open video capture
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            raise RuntimeError(f"Cannot access video source: {video_source}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        logger.info(f"Video opened: {width}x{height} @ {fps}fps")
        logger.info("Press 'q' or ESC to exit")

        frame_count = 0
        last_time = time.time()

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to read frame. Exiting...")
                    break

                # Track frame time for FPS calculation
                current_time = time.time()
                frame_time = current_time - last_time
                last_time = current_time

                # Update frame times for FPS calculation
                self.frame_times.append(frame_time)
                if len(self.frame_times) > self.max_frame_times:
                    self.frame_times.pop(0)

                # Detect objects in frame
                annotated_frame, detected_objects = self.detect_objects(frame)

                # Draw FPS counter
                if self.show_fps:
                    fps_value = self.calculate_fps()
                    self.draw_fps(annotated_frame, fps_value)

                # Draw statistics
                if self.show_stats:
                    self.draw_stats(annotated_frame, detected_objects)

                # Display frame
                cv2.imshow(window_name, annotated_frame)

                # Log detections periodically (every 30 frames)
                if detected_objects and frame_count % 30 == 0:
                    unique_objects = set(detected_objects)
                    logger.info(f"Detected: {', '.join(unique_objects)}")

                frame_count += 1

                # Check for exit key (q or ESC)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27 is ESC key
                    logger.info("Exit key pressed")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise
        finally:
            # Clean up resources
            logger.info("Releasing resources...")
            logger.info(f"Session Statistics:")
            logger.info(f"  Total frames processed: {frame_count}")
            logger.info(f"  Total detections: {self.total_detections}")
            if self.class_counts:
                logger.info(f"  Objects detected by class:")
                for cls, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"    {cls}: {count}")
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Detection stopped")


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary of configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config or {}
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time object detection using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use default webcam
  %(prog)s --source 1               # Use second camera
  %(prog)s --source video.mp4       # Use video file
  %(prog)s --model yolov8s.pt       # Use small model
  %(prog)s --confidence 0.7         # Higher confidence threshold
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: camera index (0, 1, ...) or video file path (default: 0)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO model variant (default: yolov8n.pt)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections, 0.0-1.0 (default: 0.5)'
    )

    parser.add_argument(
        '--color',
        type=str,
        default='0,255,0',
        help='Bounding box color in BGR format (default: 0,255,0 for green)'
    )

    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Bounding box line thickness (default: 2)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--tracking',
        action='store_true',
        help='Enable object tracking between frames'
    )

    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Disable FPS counter display'
    )

    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable statistics display'
    )

    parser.add_argument(
        '--alert',
        type=str,
        nargs='+',
        help='Object classes to trigger alerts (e.g., --alert person car dog)'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load config file if provided
    config = {}
    if args.config:
        try:
            config = load_config(Path(args.config))
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1

    # Merge CLI args with config (CLI args take precedence)
    model_name = args.model if hasattr(args, 'model') else config.get('model_name', 'yolov8n.pt')
    confidence = args.confidence if hasattr(args, 'confidence') else config.get('confidence_threshold', 0.5)
    thickness = args.thickness if hasattr(args, 'thickness') else config.get('box_thickness', 2)
    source = args.source if hasattr(args, 'source') else config.get('video_source', '0')

    # Handle tracking
    enable_tracking = args.tracking if args.tracking else config.get('enable_tracking', False)

    # Handle FPS and stats display
    show_fps = not args.no_fps if hasattr(args, 'no_fps') else config.get('show_fps', True)
    show_stats = not args.no_stats if hasattr(args, 'no_stats') else config.get('show_stats', True)

    # Handle alert objects
    alert_objects = set(args.alert) if args.alert else set(config.get('alert_objects', []))

    # Parse color argument
    try:
        color_str = args.color if hasattr(args, 'color') else config.get('box_color', '0,255,0')
        if isinstance(color_str, list):
            color = tuple(color_str)
        else:
            color = tuple(map(int, color_str.split(',')))
        if len(color) != 3 or not all(0 <= c <= 255 for c in color):
            raise ValueError
    except ValueError:
        logger.error("Invalid color format. Use BGR format: B,G,R (e.g., 0,255,0)")
        return 1

    # Convert source to int if it's a digit
    video_source = int(source) if str(source).isdigit() else source

    # Validate confidence threshold
    if not 0.0 <= confidence <= 1.0:
        logger.error("Confidence threshold must be between 0.0 and 1.0")
        return 1

    try:
        # Initialize detector
        detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence,
            box_color=color,
            box_thickness=thickness,
            enable_tracking=enable_tracking,
            show_fps=show_fps,
            show_stats=show_stats,
            alert_objects=alert_objects
        )

        # Run detection
        detector.run_detection(video_source=video_source)

        return 0

    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
