# Object Flow Tracking Application with YOLO and Streamlit

This Docker image contains an object detection application built using **Python 3.10** with **YOLO**, **Streamlit**, and **OpenCV**. The app supports real-time object detection from a system camera and provides a Streamlit web interface for visualizing the detection results.

## Key Features
- **Streamlit**: A user-friendly web interface for displaying object detection results.
- **YOLO (Ultralytics)**: Leverages the YOLO model for efficient object detection.
- **OpenCV**: Captures live video from the system’s camera.
- **Streamlit-WebRTC**: Enables real-time video stream processing in the web interface.
- **Torch**: Provides the backbone for machine learning computations.
- **Lightweight**: Uses `opencv-python-headless` for a minimal image size.

## Included Dependencies
- streamlit
- opencv-python-headless
- ultralytics
- torch
- numpy==1.23.5

## Running the Application

## Option 1: Clone the Repository and Run Locally
clone the repository, install the dependencies, and run the application locally.
1. Clone the Repository:
   ```
   git clone https://github.com/411049/object-flow-tracker-app.git
   cd object-flow-tracker-app

   ```
2. Install the Required Python Packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Application:
   ```
   streamlit run APP.py
   ```
4. Access the Application: Open your browser and navigate to:
   ```
   http://localhost:8501
   ```


### Option 2: Pull and Run from Docker Hub

Alternatively, you can pull and run the Docker image, use the following commands.

1. Pull the Docker image:
   ```
   docker pull 411049/object-flow-tracker-app:latest
   ```

2. Run the application:
  ```
  docker run -it --rm \
    --device=/dev/video0 \
    -p 8501:8501 \
    411049/object-flow-tracker-app:latest

   ```

Command Breakdown:

 -   `--device=/dev/video0`: Grants access to the system's camera.
 -  `-p 8501:8501`: Exposes the Streamlit web interface on port 8501.
 -  `-it --rm`: Runs the container interactively and removes it after you exit.

Once the container is running, access the web application by opening your browser and navigating to:
```
http://localhost:8501
```

## Dockerfile Overview
```
# Use a Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt to install dependencies
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port for Streamlit (default is 8501)
EXPOSE 8501

# Command to run your application (use Streamlit as example)
CMD ["streamlit", "run", "streamlit_app.py"]

```

## Notes

-   This application uses YOLO for object detection.
-   Ensure that the system’s camera is connected and accessible.
-   The Streamlit interface provides an easy-to-use way to visualize the object detection results.
