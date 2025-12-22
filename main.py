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
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
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
        font_thickness: int = 2
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
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness

        logger.info(f"Loading YOLO model: {model_name}")
        try:
            self.model = YOLO(model_name)
            logger.info(f"Model loaded successfully. Device: {self.model.device}")
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
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        detected_objects = []

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

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        self.box_color,
                        self.box_thickness
                    )

                    # Draw label with confidence
                    label_text = f"{label} {confidence:.2f}"
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        self.box_color,
                        self.font_thickness
                    )

        return frame, detected_objects

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

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to read frame. Exiting...")
                    break

                # Detect objects in frame
                annotated_frame, detected_objects = self.detect_objects(frame)

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
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Detection stopped")


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

    return parser.parse_args()


def main() -> int:
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse color argument
    try:
        color = tuple(map(int, args.color.split(',')))
        if len(color) != 3 or not all(0 <= c <= 255 for c in color):
            raise ValueError
    except ValueError:
        logger.error("Invalid color format. Use BGR format: B,G,R (e.g., 0,255,0)")
        return 1

    # Convert source to int if it's a digit
    video_source = int(args.source) if args.source.isdigit() else args.source

    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        logger.error("Confidence threshold must be between 0.0 and 1.0")
        return 1

    try:
        # Initialize detector
        detector = ObjectDetector(
            model_name=args.model,
            confidence_threshold=args.confidence,
            box_color=color,
            box_thickness=args.thickness
        )

        # Run detection
        detector.run_detection(video_source=video_source)

        return 0

    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
