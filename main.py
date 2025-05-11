import cv2
import numpy as np
from camera_utils import calibrate_cameras, create_weight_maps

def capture_and_display():
    # Open 4 camera streams
    cameras = [cv2.VideoCapture(i) for i in range(4)]

    if not all([cam.isOpened() for cam in cameras]):
        print("Error: One or more cameras could not be opened.")
        return

    # Get frame dimensions from first camera for reference
    _, test_frame = cameras[0].read()
    if test_frame is None:
        print("Error: Could not read frame from camera 0.")
        return
    
    # We'll use 960x540 as our standard frame size
    frame_size = (960, 540)
    output_size = (1920, 1080)
    
    # Calibrate cameras (in a real app, this would be done separately and saved)
    print("Calibrating cameras...")
    calibration = calibrate_cameras()
    homographies = calibration['homographies']
    
    # Compute weight maps for blending
    weight_maps = [
        create_weight_maps(frame_size) for _ in range(4)
    ]
    
    # Create the output panorama canvas - all cameras will be warped into this space
    panorama = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    
    print("Starting video capture and blending...")
    while True:
        frames = []
        for idx, cam in enumerate(cameras):
            ret, frame = cam.read()
            if not ret:
                print(f"Error: Could not read frame from camera {idx+1}.")
                break
            
            # Resize each frame to standard size
            frame = cv2.resize(frame, frame_size)
            
            # Add label to the frame
            cv2.putText(frame, f"Camera {idx + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frames.append(frame)

        if len(frames) != 4:
            break

        # Display original 4-up view
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid_view = np.vstack((top_row, bottom_row))
        cv2.imshow('Original Camera Grid', grid_view)
        
        # Reset panorama canvas
        panorama.fill(0)
        
        # Warp and blend each frame onto the panorama
        for idx, frame in enumerate(frames):
            # Apply homography transformation to warp frame to reference coordinate system
            warped = cv2.warpPerspective(
                frame, 
                homographies[idx], 
                (output_size[0], output_size[1])
            )
            
            # Create mask for non-zero pixels in the warped image
            mask = (warped > 0).astype(np.float32)
            
            # Apply weight map to create gradual blending weights
            # For simplicity, we're using a fake weight map that just fades from edges
            # Scale the weight map to match the warped size
            weight_map = cv2.resize(weight_maps[idx], frame_size)
            weight_map = cv2.warpPerspective(weight_map, homographies[idx], (output_size[0], output_size[1]))
            weight_map = weight_map[:, :, np.newaxis]  # Add channel dimension
            
            # Apply the weight map to the warped image
            warped_weighted = warped.astype(np.float32) * weight_map * mask
            
            # Add to the panorama
            panorama = cv2.add(
                panorama, 
                warped_weighted.astype(np.uint8)
            )
        
        # Use a simpler approach for normalization to avoid issues with masks
        # In a real application, you'd use more sophisticated blending
        panorama_norm = panorama.copy()
        non_zero_mask = (panorama.sum(axis=2) > 0)
        
        if np.any(non_zero_mask):
            # Manual normalization approach
            # Find min/max values in non-zero areas
            min_val = float('inf')
            max_val = 0
            
            for c in range(3):  # Process each channel
                channel = panorama[:,:,c]
                channel_masked = channel[non_zero_mask]
                if len(channel_masked) > 0:
                    channel_min = channel_masked.min()
                    channel_max = channel_masked.max()
                    min_val = min(min_val, channel_min)
                    max_val = max(max_val, channel_max)
            
            # Only normalize if we have a valid range
            if min_val < max_val:
                # Apply normalization manually
                scale = 255.0 / (max_val - min_val)
                panorama_norm = np.zeros_like(panorama)
                for c in range(3):
                    panorama_norm[:,:,c] = np.where(
                        non_zero_mask,
                        np.clip((panorama[:,:,c] - min_val) * scale, 0, 255),
                        0
                    )
        
        # Display the blended panorama
        cv2.imshow('Blended Panorama', panorama_norm)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_display()
