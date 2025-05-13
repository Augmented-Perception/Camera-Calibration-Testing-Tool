import cv2
import numpy as np
import sys
import os

def run_demo():
    """
    This is a more realistic calibration demo that you would use to calibrate
    real cameras in your system. It uses chessboard patterns for calibration.
    
    Steps:
    1. Use a chessboard pattern and take multiple pictures with each camera
    2. Find chessboard corners in each image
    3. Calculate camera intrinsic parameters
    4. (For multi-camera systems) Find relative positions between cameras
    """
    # Define chessboard pattern dimensions
    pattern_size = (9, 6)  # number of corners in width and height
    square_size = 3      # size of each square in cm
    
    # World coordinates of corners (assuming Z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    # Create folder for calibration images if it doesn't exist
    calib_dir = "calibration_images"
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
    
    print("""
    Camera Calibration Tool
    ======================
    
    This tool will help you calibrate multiple cameras for blending.
    
    Instructions:
    1. Point each camera at a chessboard pattern (printed or displayed)
    2. Press 'c' to capture an image for calibration
    3. Take at least 10 images from different angles
    4. Press 'q' to quit and calculate calibration parameters
    
    Note: For best results, make sure the chessboard is visible in each capture.
    """)
    
    # Open all 4 cameras
    cameras = []
    for i in range(4):
        cam = cv2.VideoCapture(i)
        if not cam.isOpened():
            print(f"Could not open camera {i}")
            continue
        cameras.append(cam)
    
    if not cameras:
        print("Error: No cameras detected")
        return
    
    # Storage for calibration data
    cam_calibration_data = [
        {
            "obj_points": [],  # 3D points in real world space
            "img_points": [],  # 2D points in image plane
            "images": []       # Filenames of saved images
        } 
        for _ in cameras
    ]
    
    frame_count = 0
    print(f"Camera IDs: {list(range(len(cameras)))}")
    
    # Display stream and capture calibration images
    while True:
        frames = []
        for i, cam in enumerate(cameras):
            ret, frame = cam.read()
            if not ret:
                print(f"Failed to grab frame from camera {i}")
                continue
            
            # Resize for display
            frame = cv2.resize(frame, (960, 540))
            
            # Try to find chessboard corners
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_chess, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            # If found, add visual indicator
            if ret_chess:
                cv2.drawChessboardCorners(frame, pattern_size, corners, ret_chess)
                cv2.putText(frame, "Chessboard Detected!", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera label
            cv2.putText(frame, f"Camera {i+1}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            frames.append(frame)
        
        # Display all cameras in a grid
        if len(frames) >= 4:
            top_row = np.hstack((frames[0], frames[1]))
            bottom_row = np.hstack((frames[2], frames[3]))
            grid_view = np.vstack((top_row, bottom_row))
        else:
            grid_view = np.vstack(frames) if len(frames) > 1 else frames[0]
        
        cv2.imshow('Camera Calibration', grid_view)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture images for calibration
            frame_count += 1
            for i, cam in enumerate(cameras):
                ret, cal_frame = cam.read()
                if not ret:
                    continue

                filename = f"{calib_dir}/cam{i+1}_frame{frame_count}.jpg"
                cv2.imwrite(filename, cal_frame)

                gray = cv2.cvtColor(cal_frame, cv2.COLOR_BGR2GRAY)
                ret_chess, corners = cv2.findChessboardCorners(gray, pattern_size, None)
                if ret_chess:
                    # Refine corner positions
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cam_calibration_data[i]["obj_points"].append(objp)
                    cam_calibration_data[i]["img_points"].append(corners2)
                    cam_calibration_data[i]["images"].append(filename)
                    print(f"Saved calibration image for camera {i+1} (total: {len(cam_calibration_data[i]['images'])})")
                else:
                    print(f"Warning: Could not find chessboard in camera {i+1}")
            
    cv2.destroyAllWindows()
    print("\nCalibration complete. You can now use the calibration data with the main blending program.")

if __name__ == "__main__":
    run_demo()
    # Release cameras
    for cam in cameras:
        cam.release()

    # Perform calibration for each camera
    print("\nPerforming camera calibration...")

    for i, data in enumerate(cam_calibration_data):
        if not data["img_points"]:
            print(f"Camera {i+1}: No calibration data collected")
            continue

        print(f"Camera {i+1}: Calibrating with {len(data['img_points'])} images...")

        # Get image dimensions from first saved image
        img = cv2.imread(data["images"][0], cv2.IMREAD_GRAYSCALE)
        img_size = img.shape[::-1]

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            data["obj_points"], data["img_points"], img_size, None, None
        )

        if ret:
            # Save calibration data
            np.savez(f"camera{i+1}_calibration.npz",
                        camera_matrix=mtx,
                        dist_coeffs=dist,
                        rvecs=rvecs,
                        tvecs=tvecs)

            print(f"Camera {i+1}: Calibration successful")
            print(f"  - Camera Matrix:\n{mtx}")
            print(f"  - Distortion Coefficients: {dist.ravel()}")

            # Calculate reprojection error
            mean_error = 0
            for j in range(len(data["obj_points"])):
                imgpoints2, _ = cv2.projectPoints(
                    data["obj_points"][j], rvecs[j], tvecs[j], mtx, dist
                )
                error = cv2.norm(data["img_points"][j], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error

            print(f"  - Reprojection Error: {mean_error/len(data['obj_points'])}")
        else:
            print(f"Camera {i+1}: Calibration failed")

    cv2.destroyAllWindows()
    print("\nCalibration complete. You can now use the calibration data with the main blending program.")
