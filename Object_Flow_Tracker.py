import cv2
import numpy as np
from typing import Union, Tuple, List, Dict
import time
import os
from ultralytics import YOLO
import torch
import logging
import sys
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class YOLOFlowTracker:
    def __init__(self,
                 yolo_model_path: str,
                 confidence_threshold: float = 0.5,
                 window_size: int = 21,
                 max_level: int = 3,
                 track_len: int = 10,
                 grid_size: int = 20):  # Grid cell size in pixels
        """Initialize the tracker with detailed error checking."""
        try:
            logger.info("Initializing YOLOFlowTracker...")
            
            # Initialize YOLO model
            self.yolo_model = YOLO(yolo_model_path)
            self.conf_threshold = confidence_threshold
            
            # Store configuration
            self.window_size = window_size
            self.max_level = max_level
            self.track_len = track_len
            self.grid_size = grid_size
            
            # Initialize tracking parameters
            self.lk_params = dict(
                winSize=(window_size, window_size),
                maxLevel=max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            # Initialize state variables
            self.prev_gray = None
            self.tracks = {}
            self.track_ids = 0
            self.prev_time = time.time()
            self.fps = 0
            self.frame_count = 0
            self.prev_points = None
            self.prev_grid_flow = None
            
            # Device information
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                self.device_name = torch.cuda.get_device_name(0)
            else:
                self.device_name = "CPU"
            
            logger.info("YOLOFlowTracker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLOFlowTracker: {str(e)}")
            raise

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO-v8."""
        try:
            results = self.yolo_model(frame, verbose=False)[0]
            detections = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf)
                
                if conf > self.conf_threshold:
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return []

    def create_grid_points(self, bbox: Tuple[int, int, int, int], margin: int = 50) -> np.ndarray:
        """Create a grid of points around the bounding box."""
        x1, y1, x2, y2 = bbox
        
        # Add margin around bbox
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = x2 + margin
        y2 = y2 + margin
        
        # Create grid points
        x_points = np.arange(x1, x2, self.grid_size)
        y_points = np.arange(y1, y2, self.grid_size)
        
        # Create meshgrid
        xv, yv = np.meshgrid(x_points, y_points)
        
        # Reshape to Nx2 array of points
        return np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)

    def draw_grid_flow(self, img: np.ndarray, points: np.ndarray, 
                      next_points: np.ndarray, status: np.ndarray) -> None:
        """Draw grid flow arrows with professional styling."""
        mask = status.flatten() == 1
        points = points[mask]
        next_points = next_points[mask]
        
        # Calculate flow vectors
        flow_vectors = next_points - points
        
        # Calculate vector magnitudes
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
        max_magnitude = np.max(magnitudes) if magnitudes.size > 0 else 1
        
        for pt1, pt2, magnitude in zip(points, next_points, magnitudes):
            # Skip very small movements
            if magnitude < 0.5:
                continue
                
            # Normalize magnitude for color intensity
            intensity = min(magnitude / max_magnitude, 1.0)
            
            # Create color gradient from blue to yellow based on magnitude
            color = (0,
                    int(255 * intensity),  # Green component
                    int(255 * (1 - intensity)))  # Red component
            
            # Calculate arrow properties
            angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            
            # Draw arrow with adaptive size
            arrow_length = min(length, self.grid_size * 0.8)
            arrow_head_size = min(5 + intensity * 5, arrow_length * 0.3)
            
            # Calculate end point
            end_pt = (int(pt1[0] + arrow_length * np.cos(angle)),
                     int(pt1[1] + arrow_length * np.sin(angle)))
            
            # Draw arrow shaft with anti-aliasing
            cv2.line(img, tuple(map(int, pt1)), end_pt, color, 1, cv2.LINE_AA)
            
            # Draw arrow head
            arrow_head_angle = np.pi / 6  # 30 degrees
            pt_left = (
                int(end_pt[0] - arrow_head_size * np.cos(angle + arrow_head_angle)),
                int(end_pt[1] - arrow_head_size * np.sin(angle + arrow_head_angle))
            )
            pt_right = (
                int(end_pt[0] - arrow_head_size * np.cos(angle - arrow_head_angle)),
                int(end_pt[1] - arrow_head_size * np.sin(angle - arrow_head_angle))
            )
            
            # Draw arrow head with anti-aliasing
            cv2.polylines(img, [np.array([end_pt, pt_left, pt_right])],
                         True, color, 1, cv2.LINE_AA)

    def draw_performance_metrics(self, img: np.ndarray) -> None:
        """Draw FPS and device information on the frame."""
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time

        # Create semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (250, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Draw metrics
        metrics = [
            f"FPS: {self.fps}",
            f"Device: {self.device_name}",
            f"Grid Size: {self.grid_size}px"
        ]
        
        for i, text in enumerate(metrics):
            cv2.putText(img, text, (20, 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with detection and tracking."""
        try:
            if frame is None:
                logger.error("Received empty frame")
                return None
            
            # Convert frame to grayscale for optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            
            # Initialize previous frame if needed
            if self.prev_gray is None:
                self.prev_gray = gray
                return vis
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            for det in detections:
                # Draw bounding box
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create grid points around the detected object
                grid_points = self.create_grid_points(det['bbox'])
                
                if grid_points.size > 0:
                    # Calculate optical flow for grid points
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, grid_points,
                        None, **self.lk_params
                    )
                    
                    if next_points is not None:
                        # Draw flow grid
                        self.draw_grid_flow(vis, grid_points, next_points, status)
                        
                        # Draw main object motion arrow
                        center = det['center']
                        mean_flow = np.mean(next_points[status.flatten() == 1] - 
                                          grid_points[status.flatten() == 1], axis=0)
                        if not np.isnan(mean_flow).any():
                            end_point = (int(center[0] + mean_flow[0] * 2),
                                       int(center[1] + mean_flow[1] * 2))
                            cv2.arrowedLine(vis, center, end_point, 
                                          (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
            
            # Draw performance metrics
            self.draw_performance_metrics(vis)
            
            # Update previous frame
            self.prev_gray = gray.copy()
            
            return vis
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame if frame is not None else None

    def process_video(self, source: Union[str, int], display: bool = True) -> None:
        """Process video stream with detection and tracking."""
        cap = None
        window_name = 'Object Flow Tracking'
        
        try:
            logger.info(f"Opening video source: {source}")
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video source: {source}")
            
            if display:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None and display:
                    try:
                        cv2.imshow(window_name, processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User requested quit")
                            break
                    except Exception as e:
                        logger.error(f"Error displaying frame: {str(e)}")
                
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            
        finally:
            if cap is not None:
                cap.release()
            if display:
                cv2.destroyAllWindows()
                for _ in range(5):
                    cv2.waitKey(1)

def main():
    try:
        logger.info("Starting object flow tracking application...")
        
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        logger.info(f"Using device: {device}")
        
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            logger.error(f"YOLO model not found at: {model_path}")
            return
        
        tracker = YOLOFlowTracker(
            yolo_model_path=model_path,
            confidence_threshold=0.5,
            grid_size=20  # Adjust grid density
        )
        
        tracker.process_video('Data/2.mp4')  # Use camera index 0
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main()