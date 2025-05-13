import cv2
import numpy as np
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_checkerboard(frame):
    """
    Simple checkerboard detection optimized for phone screens.
    Returns True if a checkerboard pattern is detected.
    """
    if frame is None:
        return False, None
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and analyze contours
        squares = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Filter out small contours
                continue
                
            # Approximate the contour to a polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Check if it's a square (4 corners)
            if len(approx) == 4:
                # Check if it's roughly square-shaped
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                if 0.8 <= aspect_ratio <= 1.2:  # Allow some tolerance
                    squares.append(approx)
        
        # Check if we found enough squares in a grid pattern
        if len(squares) >= 4:  # At least 2x2 squares
            # Sort squares by position (top-left to bottom-right)
            squares = sorted(squares, key=lambda x: (x[:, 0, 1].mean(), x[:, 0, 0].mean()))
            
            # Convert squares to points for visualization
            points = []
            for square in squares:
                points.append(square.mean(axis=0))
            points = np.array(points, dtype=np.float32)
            
            return True, points
            
        return False, None
        
    except Exception as e:
        logger.error(f"Error in checkerboard detection: {str(e)}")
        return False, None

def run_demo():
    """
    Camera calibration demo using chessboard patterns.
    """
    # Create folder for calibration images
    calib_dir = "calibration_images"
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
    
    print("""
    Camera Calibration Tool
    ======================
    
    This tool will help you calibrate multiple cameras for blending.
    
    Instructions:
    1. Point each camera at a chessboard pattern (printed or displayed)
    2. Press 'c' to capture an image and check for checkerboard pattern
    3. Press 'q' to quit
    
    Note: At least a 2x2 square checkerboard pattern must be visible.
    """)
    
    # Open all 4 cameras
    cameras = []
    for i in range(4):
        try:
            cam = cv2.VideoCapture(i)
            if not cam.isOpened():
                logger.warning(f"Could not open camera {i}")
                continue
            
            # Set camera properties for better capture
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            cameras.append(cam)
            logger.info(f"Successfully initialized camera {i}")
        except Exception as e:
            logger.error(f"Error initializing camera {i}: {str(e)}")
    
    if not cameras:
        logger.error("No cameras detected")
        return
    
    frame_count = 0
    logger.info(f"Active camera IDs: {list(range(len(cameras)))}")
    
    while True:
        frames = []
        detection_status = []
        
        for i, cam in enumerate(cameras):
            try:
                ret, frame = cam.read()
                if not ret:
                    continue
                
                # Resize for display
                frame = cv2.resize(frame, (480, 270))
                
                # Try to find checkerboard pattern
                has_pattern, points = detect_checkerboard(frame)
                detection_status.append(has_pattern)
                
                # Add visual indicators
                if has_pattern and points is not None:
                    # Draw detected points
                    for point in points:
                        pos = tuple(map(int, point.ravel()))
                        cv2.circle(frame, pos, 3, (0, 255, 0), -1)
                    
                    # Draw connections between points
                    for i in range(len(points)-1):
                        pt1 = tuple(map(int, points[i].ravel()))
                        pt2 = tuple(map(int, points[i+1].ravel()))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                    
                    cv2.putText(frame, "Checkerboard Detected!", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Checkerboard", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Camera {i+1}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                frames.append(frame)
            except Exception as e:
                logger.error(f"Error processing camera {i} frame: {str(e)}")
        
        try:
            # Display all cameras in a grid
            if len(frames) >= 4:
                top_row = np.hstack((frames[0], frames[1]))
                bottom_row = np.hstack((frames[2], frames[3]))
                grid_view = np.vstack((top_row, bottom_row))
            else:
                grid_view = np.vstack(frames) if len(frames) > 1 else frames[0]
            
            cv2.imshow('Camera Calibration', grid_view)
        except Exception as e:
            logger.error(f"Error creating grid view: {str(e)}")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            frame_count += 1
            detection_results = []
            
            for i, cam in enumerate(cameras):
                try:
                    ret, cal_frame = cam.read()
                    if not ret:
                        detection_results.append(False)
                        continue

                    has_pattern, _ = detect_checkerboard(cal_frame)
                    detection_results.append(has_pattern)

                    if has_pattern:
                        filename = f"{calib_dir}/cam{i+1}_frame{frame_count}.jpg"
                        cv2.imwrite(filename, cal_frame)
                        logger.info(f"Camera {i+1}: Checkerboard detected and image saved")
                    else:
                        logger.info(f"Camera {i+1}: No checkerboard pattern detected")
                except Exception as e:
                    logger.error(f"Error processing capture from camera {i}: {str(e)}")
                    detection_results.append(False)
            
            print("Detection results:", detection_results)
            
    # Cleanup
    for cam in cameras:
        try:
            cam.release()
        except Exception as e:
            logger.error(f"Error releasing camera: {str(e)}")
            
    cv2.destroyAllWindows()
    logger.info("Calibration tool shutdown complete")

if __name__ == "__main__":
    run_demo()
