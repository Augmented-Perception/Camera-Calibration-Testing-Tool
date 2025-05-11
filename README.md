# Live Camera Capture Project

This project is designed to capture live video streams from 4 different cameras, align them, and blend them together into a single seamless view.

## Requirements
- Python 3.x
- OpenCV

## Setup
1. Install Python 3.x if not already installed.
2. Install the required Python packages:
   ```
   pip install opencv-python numpy
   ```

## Project Structure
- `main.py`: Main application that captures, warps, and blends camera feeds
- `camera_utils.py`: Utility functions for camera calibration and image blending
- `calibration_tool.py`: Tool for calibrating cameras using a chessboard pattern

## Calibration
For best results, you should calibrate your cameras before creating a blended view:

1. Print a chessboard pattern (9x6 corners recommended)
2. Run the calibration tool:
   ```
   python calibration_tool.py
   ```
3. Follow the on-screen instructions to capture calibration images
4. The tool will generate calibration files for each camera

## Running the Project
Run the main script to start capturing video streams:
```
python main.py
```

## Controls
- Press 'q' to quit the application
