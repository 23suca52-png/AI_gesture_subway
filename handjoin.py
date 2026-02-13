from email.mime import image

import cv2
from matplotlib import image
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt
import time as time_module

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose


# Setup the Pose function for videos.
pose_video = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

from posedection import detectPose


# =====================================================
# UI ENHANCEMENT FUNCTIONS
# =====================================================
def draw_text_with_bg(
    frame,
    text,
    pos,
    font_scale=0.8,
    thickness=2,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
):
    """Draw text with background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = pos

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - 5, y - text_size[1] - 5),
        (x + text_size[0] + 5, y + 5),
        bg_color,
        -1,
    )

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def draw_status_panel(frame, hand_status, distance, fps):
    """Draw a status panel on the left side"""
    h, w = frame.shape[:2]
    panel_width = 300
    panel_x = 10

    # Semi-transparent overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width + 20, 220), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 30
    line_height = 40

    # Title
    draw_text_with_bg(
        frame,
        "HAND JOIN DETECTOR",
        (panel_x, y),
        font_scale=0.9,
        text_color=(0, 255, 255),
        bg_color=(0, 0, 0),
    )
    y += line_height + 10

    # Hand Status
    status_color = (0, 255, 0) if "Joined" in hand_status else (0, 0, 255)
    cv2.putText(
        frame,
        f"Status: {hand_status}",
        (panel_x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )
    y += line_height

    # Distance
    cv2.putText(
        frame,
        f"Distance: {distance}px",
        (panel_x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 0),
        2,
    )
    y += line_height

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",
        (panel_x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def draw_wrist_indicator(frame, hand_status):
    """Draw wrist join indicator at the top"""
    h, w = frame.shape[:2]

    is_joined = "Joined" in hand_status
    color = (0, 255, 0) if is_joined else (0, 0, 255)

    # Draw large indicator circle
    radius = 20
    center = (w - 50, 50)
    cv2.circle(frame, center, radius, color, -1)
    cv2.circle(frame, center, radius, (255, 255, 255), 2)

    # Draw label
    label = "✓ JOINED" if is_joined else "✗ SEPARATED"
    label_color = (255, 255, 255)
    cv2.putText(
        frame, label, (w - 150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2
    )


def draw_instructions(frame):
    """Draw instructions at the bottom"""
    h, w = frame.shape[:2]

    instructions = ["Join both hands together to start game| ESC to quit"]

    bg_height = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bg_height), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, text in enumerate(instructions):
        y = h - 25
        cv2.putText(
            frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )


def checkHandsJoined(image, results, draw=False, display=False):

    # Get the height and width of the input imoge.
    height, width, _ = image.shape

    # Create a copy of the input image to write the hands status Label on.
    output_image = image.copy()

    # Get the left wrist Landmark x and y coordinates.
    left_wrist_landmark = (
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height,
    )

    # Get the right wrist Landmark x and y coordinates.
    right_wrist_landmark = (
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height,
    )

    # Calculate the euclidean distance between the left and right wrist.
    euclidean_distance = int(
        hypot(
            left_wrist_landmark[0] - right_wrist_landmark[0],
            left_wrist_landmark[1] - right_wrist_landmark[1],
        )
    )

    # Compare the distance between the wrists with a appropriate threshold to check if both hands are joined.
    if euclidean_distance < 130:
        # Set the hands status to joined.
        hand_status = "Hands Joined"
        # Set the color value to green.
        color = (0, 255, 0)

    # otherwise.
    else:
        # Set the hands status to not joined.
        hand_status = "Hands Not Joined"
        # Set the color value to red.
        color = (0, 0, 255)

    # Check if the Hands Joined status and hands distance are specified to be written on the output image.
    if draw:
        # Write the classified hands status on the image.
        cv2.putText(
            output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3
        )
        # Write the the distance between the wrists on the image.
        cv2.putText(
            output_image,
            f"Distance: {euclidean_distance}",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            color,
            3,
        )

    # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")

    # Otherwise
    else:
        # Return the output image and the classified hands status indicating whether the hands are joined or not.
        return output_image, hand_status


# Initiolize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# FPS calculation
prev_frame_time = 0
curr_frame_time = 0

# Create named window for resizing purposes.
cv2.namedWindow("🎮 Hand Join Detector", cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Calculate FPS
    curr_frame_time = time_module.time()
    fps = 1 / (curr_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = curr_frame_time

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the height and width of the frame of the webcame video.
    frame_height, frame_width, _ = frame.shape

    # Perform the pose detection on the frame.
    frame, results = detectPose(frame, pose_video, draw=True)

    # Check if the pose Landmarks in the frame are detected.
    hand_status = "Unknown"
    distance = 0
    if results.pose_landmarks:

        # Check if the left and right hands are joined.
        frame, hand_status = checkHandsJoined(frame, results, draw=True)

        # Calculate distance for display
        left_wrist_landmark = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
            * frame_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
            * frame_height,
        )
        right_wrist_landmark = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
            * frame_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            * frame_height,
        )
        distance = int(
            hypot(
                left_wrist_landmark[0] - right_wrist_landmark[0],
                left_wrist_landmark[1] - right_wrist_landmark[1],
            )
        )

    # Draw UI elements
    draw_wrist_indicator(frame, hand_status)
    draw_status_panel(frame, hand_status, distance, fps)
    draw_instructions(frame)

    # Display the frame.
    cv2.imshow("🎮 Hand Join Detector", frame)

    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if k == 27:
        break


# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
