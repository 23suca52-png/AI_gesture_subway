import cv2
from matplotlib import image
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt

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


def detectPose(image, pose, draw=False, display=False):

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Check if any Landmarks are detected and are specified to be drawn.
    if results.pose_landmarks and draw:

        # Draw Pose Landmarks on the output image.
        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=3, circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(49, 125, 237), thickness=2, circle_radius=2
            ),
        )

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")
        plt.show()

    # Otherwise
    else:

        # Return the output image and the results of pose Landmarks detection.
        return output_image, results


# Read a sample image and perform pose Landmarks detection on it.
# IMG_PATH = "media/sample.jpg"
# image = cv2.imread(IMG_PATH)
# detectPose(image, pose_image, draw=True, display=True)
