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


def draw_status_panel(frame, horizontal_position, fps):
    """Draw a status panel on the left side"""
    h, w = frame.shape[:2]
    panel_width = 300
    panel_x = 10

    # Semi-transparent overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width + 20, 200), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 30
    line_height = 40

    # Title
    draw_text_with_bg(
        frame,
        "LEFT/RIGHT CONTROL",
        (panel_x, y),
        font_scale=0.9,
        text_color=(0, 255, 255),
        bg_color=(0, 0, 0),
    )
    y += line_height + 10

    # Position Status
    pos_color = (
        (255, 0, 0)
        if horizontal_position == "Left"
        else (0, 255, 0) if horizontal_position == "Center" else (255, 0, 255)
    )
    cv2.putText(
        frame,
        f"Position: {horizontal_position}",
        (panel_x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        pos_color,
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


def draw_horizontal_zones(frame):
    """Draw horizontal zones for left/center/right"""
    h, w = frame.shape[:2]

    # Draw vertical divider lines
    cv2.line(frame, (w // 3, 0), (w // 3, h), (100, 100, 100), 2)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (100, 100, 100), 2)

    # Highlight center line
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 3)

    # Zone labels
    cv2.putText(
        frame, "LEFT", (w // 6 - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
    )
    cv2.putText(
        frame,
        "CENTER",
        (w // 2 - 70, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "RIGHT",
        (5 * w // 6 - 40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2,
    )


def draw_instructions(frame):
    """Draw instructions at the bottom"""
    h, w = frame.shape[:2]

    instructions = [
        "Move LEFT or RIGHT to control | Stay in CENTER for no movement | ESC to quit"
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


def checkLeftRight(image, results, draw=False, display=False):

    # Declare a variable to store the horizontal position (Left, center, right) of the person.
    horizontal_position = None

    # Get the height and width of the image.
    height, width, _ = image.shape

    # Create a copy of the input image to write the horizontal position on.
    output_image = image.copy()

    # Retreive the x-coordinate of the Left shoulder Landmark.
    left_x = int(
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width
    )

    # Retreive the x-corrdinate of the right shoulder Landmark.
    right_x = int(
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width
    )

    # Check if the person is at Left that is when both shoulder Landmarks x-corrdinates
    # are less than or equal to the x-corrdinate of the center of the image.
    if right_x <= width // 2 and left_x <= width // 2:

        # Set the person's position to left.
        horizontal_position = "Left"

    # Check if the person is at right that is when both shoulder Landmarks x-corrdinates
    # are greater than or equal to the x-corrdinate of the center of the image.
    elif right_x >= width // 2 and left_x >= width // 2:

        # Set the person's position to right.
        horizontal_position = "Right"

    # Check if the person is at center that is when right shoulder Landmark x-corrdinate is greater than or equal to
    # and Left shoulder Landmark x-corrdinate is less than or equal to the x-corrdinate of the center of the image.
    elif right_x >= width // 2 and left_x <= width // 2:

        # Set the person's position to center.
        horizontal_position = "Center"

    # Check if the person's horizontal position and a line at the center of the image is specified to be drawn.
    if draw:

        # Write the horizontal position of the person on the image.
        cv2.putText(
            output_image,
            horizontal_position,
            (5, height - 10),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            3,
        )

        # Draw a Line at the center of the image.
        cv2.line(
            output_image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2
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
        # Return the output image and the person's horizontal position.
        return output_image, horizontal_position


if __name__ == "__main__":
    # Initiolize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    # FPS calculation
    prev_frame_time = 0
    curr_frame_time = 0

    # Create named window for resizing purposes.
    cv2.namedWindow(" Horizontal Movement Control", cv2.WINDOW_NORMAL)

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
        horizontal_position = "Unknown"
        if results.pose_landmarks:

            # Check horizontal position of the person and draw a line at the center of the frame.
            frame, horizontal_position = checkLeftRight(frame, results, draw=True)

        # Draw UI elements
        draw_horizontal_zones(frame)
        draw_status_panel(frame, horizontal_position, fps)
        draw_instructions(frame)

        # Display the frame.
        cv2.imshow(" Horizontal Movement Control", frame)

        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if k == 27:
            break

    # Release the VideoCapture Object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()
