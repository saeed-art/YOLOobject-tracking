# YOLOv8 Object Detection & Tracking

This project contains Python scripts for real-time object detection and tracking using YOLOv8, OpenCV, and the SORT tracking algorithm.

## Folder Structure
- `scripts/`: Contains all the Python scripts for webcam tracking and video processing.
- `Images/`: Directory for input images.
- `Videos/`: Directory for input and output videos.
- `Weights/`: Directory for YOLOv8 model weights (e.g., `yolov8n.pt`).

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the scripts from the `scripts` folder.

## Acknowledgements
- The `sort.py` script is from **[Alex Bewley's SORT repository](https://github.com/abewley/sort)** for real-time tracking of multiple objects. It is licensed under the GPL-3.0 License.
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics).
