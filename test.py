import sys
import cv2
import numpy as np
import ArducamDepthCamera as ac

MAX_DISTANCE = 1000
VIDEO_FILE = "sample_rgbd.mp4"

def process_frame(depth_buf: np.ndarray, amplitude_buf: np.ndarray) -> np.ndarray:
    depth_buf = np.nan_to_num(depth_buf)
    amplitude_buf[amplitude_buf <= 7] = 0
    amplitude_buf[amplitude_buf > 7] = 255
    depth_buf = (1 - (depth_buf / MAX_DISTANCE)) * 255
    depth_buf = np.clip(depth_buf, 0, 255)
    return depth_buf.astype(np.uint8) & amplitude_buf.astype(np.uint8)

def check_obstacle(depth_buf: np.ndarray) -> bool:
    return np.min(depth_buf) < MAX_DISTANCE

def change_orientation(result_frame: np.ndarray) -> None:
    gray_result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2GRAY)
    M = cv2.moments(gray_result_frame)

    if M["m00"] != 0:
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    height, width = result_frame.shape[:2]
    offsetX, offsetY = cX - width / 2, cY - height / 2
    THRESHOLD = 50
    target_location = None

    if offsetX > THRESHOLD:
        target_location = "Move right"
    elif offsetX < -THRESHOLD:
        target_location = "Move left"
    if offsetY > THRESHOLD:
        target_location = "Move up"
    elif offsetY < -THRESHOLD:
        target_location = "Move down"

    if target_location:
        print(f"Adjusting orientation: {target_location}")

# Create a video capture object
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    sys.exit("Failed to open the video file")

cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Split the frame into left and right halves
    height, width = frame.shape[:2]
    left_half = frame[:, :width//2, :]
    right_half = frame[:, width//2:, :]

    # Display the left half (RGB view)
    cv2.imshow("preview", left_half)

    # Process the right half for obstacle detection
    depth_buf, amplitude_buf = right_half[:, :, 0], right_half[:, :, 1]
    result_image = process_frame(depth_buf, amplitude_buf)
    result_image_colored = cv2.applyColorMap(result_image, cv2.COLORMAP_JET)

    obstacle = check_obstacle(depth_buf)

    if obstacle:
        # Display the right half with obstacle detection
        cv2.imshow("obstacle_preview", result_image_colored)

        # Also, print the status and adjust the orientation
        print("Obstacle detected")
        change_orientation(result_image_colored)
    else:
        print("No obstacle detected")

    key = cv2.waitKey(30)# Framerate

    if key == ord("q"):
        break

cap.release()
