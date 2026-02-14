import cv2
from matplotlib import image
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt
import time as time_module

from posedection import detectPose

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images.
pose_image = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=1
)

# Setup the Pose function for videos.
pose_video = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


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


def draw_status_panel(frame, posture, fps, threshold_y):
    """Draw a status panel on the left side"""
    h, w = frame.shape[:2]
    panel_width = 300
    panel_x = 10

    # Semi-transparent overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width + 20, 250), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 30
    line_height = 40

    # Title
    draw_text_with_bg(
        frame,
        "UP/DOWN CONTROL",
        (panel_x, y),
        font_scale=0.9,
        text_color=(0, 255, 255),
        bg_color=(0, 0, 0),
    )
    y += line_height + 10

    # Posture Status
    posture_color = (
        (0, 255, 0)
        if posture == "Jumping"
        else (255, 100, 0) if posture == "Crouching" else (100, 200, 200)
    )
    cv2.putText(
        frame,
        f"Posture: {posture}",
        (panel_x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        posture_color,
        2,
    )
    y += line_height

    # Threshold Y
    cv2.putText(
        frame,
        f"Threshold: {threshold_y}px",
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


def draw_vertical_zones(frame, mid_y):
    """Draw vertical zones for jump/crouch"""
    h, w = frame.shape[:2]

    # Draw threshold line
    cv2.line(frame, (0, mid_y), (w, mid_y), (0, 255, 0), 3)

    # Draw zone labels
    cv2.putText(
        frame,
        "JUMP ZONE",
        (w // 2 - 80, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "NORMAL",
        (w // 2 - 60, mid_y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "CROUCH ZONE",
        (w // 2 - 100, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 100, 0),
        2,
    )


def draw_instructions(frame):
    """Draw instructions at the bottom"""
    h, w = frame.shape[:2]

    instructions = [
        "Raise shoulders above threshold to JUMP | Lower shoulders below threshold to CROUCH | ESC to quit"
    ]

    bg_height = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bg_height), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, text in enumerate(instructions):
        y = h - 25
        cv2.putText(
            frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )


def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):

    height, width, _ = image.shape
    # Create a copy of the input imoge to write the posture Label on.
    output_image = image.copy()

    # Retreive the y-coordinate of the left shoulder Landmark.
    left_y = int(
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height
    )

    # Retreive the y-coordinate of the right shoulder Landmark.
    right_y = int(
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height
    )

    # Calculate the y-coordinate of the mid-point of both shoulders.
    actual_mid_y = abs(right_y + left_y) // 2

    # Calculate the upper and Lower bounds of the threshold.
    lower_bound = MID_Y - 35
    upper_bound = MID_Y + 35

    # Check if the person has jumped that is when the y-coordinate of the mid-point
    # of both shoulders is less than the lower bound.
    if actual_mid_y < lower_bound:

        # Set the posture to jumping.
        posture = "Jumping"

    # Check if the person has crouched that is when the y-coordinate of the mid-point
    # of both shoulders is greater than the upper bound.
    elif actual_mid_y > upper_bound:

        # Set the posture to crouching.
        posture = "Crouching"

    # Otherwise the person is standing and the y-coordinate of the mid-point
    # of both shoulders is between the upper and Lower bounds.
    else:

        # Set the posture to Standing straight.
        posture = "Standing"

    # Check if the posture and a horizontal Line at the threshold is specified to be drawn.
    if draw:

        # Write the posture of the person on the image.
        cv2.putText(
            output_image,
            posture,
            (5, height - 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            3,
        )

        # Draw a Line at the intial center y-coordinate of the person (threshold).
        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")

    # Otherwise
    else:

        # Return the output image and posture indicating
        return output_image, posture


if __name__ == "__main__":
    # Initiolize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    # FPS calculation
    prev_frame_time = 0
    curr_frame_time = 0

    # Create named window for resizing purposes.
    cv2.namedWindow(" Vertical Movement Control", cv2.WINDOW_NORMAL)

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

        # Calculate threshold Y (center of frame)
        threshold_y = frame_height // 2

        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)

        # Check if the pose Landmarks in the frame are detected.
        posture = "Unknown"
        if results.pose_landmarks:

            # Check horizontal position of the person and draw a line at the center of the frame.
            frame, posture = checkJumpCrouch(
                frame, results, MID_Y=threshold_y, draw=True
            )

        # Draw UI elements
        draw_vertical_zones(frame, threshold_y)
        draw_status_panel(frame, posture, fps, threshold_y)
        draw_instructions(frame)

        # Display the frame.
        cv2.imshow("Vertical Movement Control", frame)

        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if k == 27:
            break

    # Release the VideoCapture Object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()
