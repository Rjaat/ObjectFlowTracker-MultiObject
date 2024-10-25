import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
from typing import Union, Optional
import torch
import logging
import sys
from Object_Flow_Tracker import YOLOFlowTracker  # Import the original tracker class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class StreamlitFlowTracker:
    def __init__(self):
        """Initialize the Streamlit Flow Tracker application."""
        st.set_page_config(page_title="Object Flow Tracker", layout="wide")
        self.setup_sidebar()
        self.initialize_session_state()
        self.tracker = None
        self.initialize_tracker()

    def setup_sidebar(self):
        """Setup the sidebar with configuration options."""
        st.sidebar.title("Settings")
        
        # Model settings
        st.sidebar.header("Model Configuration")
        self.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Tracking settings
        st.sidebar.header("Tracking Configuration")
        self.grid_size = st.sidebar.slider(
            "Grid Size (pixels)",
            min_value=10,
            max_value=50,
            value=20,
            step=5
        )
        
        self.window_size = st.sidebar.slider(
            "Window Size",
            min_value=7,
            max_value=31,
            value=21,
            step=2
        )
        
        # Display settings
        st.sidebar.header("Display Settings")
        self.display_fps = st.sidebar.checkbox("Show FPS", value=True)
        self.display_grid = st.sidebar.checkbox("Show Flow Grid", value=True)

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0

    def initialize_tracker(self):
        """Initialize the YOLO Flow Tracker with current settings."""
        try:
            model_path = "yolov8n.pt"  # Ensure this path is correct
            self.tracker = YOLOFlowTracker(
                yolo_model_path=model_path,
                confidence_threshold=self.confidence_threshold,
                window_size=self.window_size,
                grid_size=self.grid_size
            )
            logger.info("Tracker initialized successfully")
        except Exception as e:
            st.error(f"Error initializing tracker: {str(e)}")
            logger.error(f"Tracker initialization failed: {str(e)}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame using the tracker."""
        if self.tracker is None:
            return frame
        
        try:
            processed_frame = self.tracker.process_frame(frame)
            return processed_frame if processed_frame is not None else frame
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame

    def handle_video_upload(self) -> Optional[str]:
        """Handle video file upload and return the path."""
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            try:
                # Create a temporary file to store the uploaded video
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                return tfile.name
            except Exception as e:
                st.error(f"Error handling uploaded file: {str(e)}")
                return None
        return None

    def run_video_processing(self, video_path: str):
        """Process a video file and display results."""
        try:
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            stop_button = st.button("Stop Processing")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame, channels="RGB", use_column_width=True)
                
                if st.session_state.get('stop_processing', False):
                    break
                
            cap.release()
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Video processing error: {str(e)}")

    def run_camera_stream(self):
        """Handle live camera stream processing."""
        try:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            stop_button = st.button("Stop Camera")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame, channels="RGB", use_column_width=True)
                
                if st.session_state.get('stop_processing', False):
                    break
                
            cap.release()
            
        except Exception as e:
            st.error(f"Error accessing camera: {str(e)}")
            logger.error(f"Camera stream error: {str(e)}")

    def run(self):
        """Main application loop."""
        st.title("Object Flow Tracker")
        
        # Input source selection
        source_type = st.radio("Select Input Source", ["Camera", "Video File"])
        
        if source_type == "Video File":
            video_path = self.handle_video_upload()
            if video_path and st.button("Process Video"):
                st.session_state.processing = True
                self.run_video_processing(video_path)
                
        else:  # Camera option
            if st.button("Start Camera"):
                st.session_state.camera_active = True
                self.run_camera_stream()
                
        # Display device information
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.sidebar.info(f"Running on: {device}")

def main():
    app = StreamlitFlowTracker()
    app.run()

if __name__ == "__main__":
    main()