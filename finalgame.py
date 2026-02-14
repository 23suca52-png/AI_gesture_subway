import cv2
from matplotlib import image
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt
import time as time_module


from handjoin import checkHandsJoined
from leftright import checkLeftRight
from posedection import detectPose
from updown import checkJumpCrouch

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


# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Create named window for resizing purposes.
cv2.namedWindow("Subway Surfers with Pose Detection", cv2.WINDOW_NORMAL)

# Initialize a variable to store the time of the previous frame.
time1 = 0

# Initialize a variable to store the state of the game (started or not).
game_started = False

# Initialize a variable to store the index of the current horizontal position of the person.
# At Start the character is at center so the index is 1 and it can move Left (value e) and right (value 2).
x_pos_index = 1

# Initialize a variable to store the index of the current vertical posture of the person.
# At Start the person is standing so the index is 1 and he can crouch (value 0) and jump (value 2).
y_pos_index = 1

# Declate a variable to store the intial y-coordinate of the mid-point of both shoulders of the person.
MID_Y = None

# Initialize a counter to store count of the number of consecutive frames with person's hands joined.
counter = 0

# Initialize the number of consecutive frames on which we want to check if person hands joined before starting the go
num_of_frames = 20

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the height and width of the frame of the webcame video.
    frame_height, frame_width, _ = frame.shape

    # Perform the pose detection on the frome.
    frame, results = detectPose(frame, pose_video, draw=game_started)

    # Check if the pose Landmarks in the frame are detected.
    if results.pose_landmarks:

        # Check if the game has started
        if game_started:

            # Commands to control the horizontal movements of the character.
            # ---------------------Horizontal Movements-------------------

            # Get horizontal position of the person in the frame.
            frame, horizontal_position = checkLeftRight(frame, results, draw=True)

            # Check if the person has moved to left from center or to center from right.
            if (horizontal_position == "Left" and x_pos_index != 0) or (
                horizontal_position == "Center" and x_pos_index == 2
            ):
                print("LEFT SENT")
                # Press the Left arrow key.
                pyautogui.press("left")

                # Update the horizontal position index of the character.
                x_pos_index -= 1

            # Check if the person has moved to Right from center or to center from
            elif (horizontal_position == "Right" and x_pos_index != 2) or (
                horizontal_position == "Center" and x_pos_index == 0
            ):
                print("RIGHT SENT")
                # Press the right arrow key.
                pyautogui.press("right")

                # Update the horizontal position index of the character.
                x_pos_index += 1

            # ------------------------------------------------------------
            # Command to start the game again after death of the character.
            # -------------------------------------------------------------

            # Check if the left and right hands are joined.
            if (
                checkHandsJoined(frame, results)[1] == "Hands Joined"
                and not game_started
            ):
                print("SPACE SENT")
                # Press the space key.
                pyautogui.press("space")
                game_started = True
                print("Game Started")

        # ----------------------------------------------------------------------------

        else:

            # Comand to Start the game first time.
            # ----------------------------------------------------------

            # Write the text representing the way to start the game on the frame.
            cv2.putText(
                frame,
                "JOIN BOTH HANDS TO START THE GAME.",
                (5, frame_height - 10),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                3,
            )

            # Check if the left and right hands are joined.
            if checkHandsJoined(frame, results)[1] == "Hands Joined":

                # Increment the count of consecutive frames with +ve condition.
                counter += 1

                # Check if the counter is equal to the required number of consecutive frames.
                if counter == num_of_frames:

                    # Update the value of the variable that stores the game state.
                    game_started = True

                    # Retreive the y-coordinate of the left shoulder Landmark.
                    left_y = int(
                        results.pose_landmarks.landmark[
                            mp_pose.PoseLandmark.RIGHT_SHOULDER
                        ].y
                        * frame_height
                    )

                    # Retreive the y-coordinate of the right shoulder Landmark.
                    right_y = int(
                        results.pose_landmarks.landmark[
                            mp_pose.PoseLandmark.LEFT_SHOULDER
                        ].y
                        * frame_height
                    )

                    # Calculate the intial y-coordinate of the mid-point of both shoulders of the person.
                    MID_Y = abs(right_y + left_y) // 2

                    # Move to 1300, 800, then click the left mouse button to start the game.
                    # pyautogui.click(x=1300, y=800, button='left')

            # Otherwise if the left and right hands are not joined.
            else:

                # Update the counter value to zero.
                counter = 0

        # ---------------------------vertical movements---------------------------------
        # Commands to control the vertical movements of the character.
        # ------------------------------------------------------------------------------

        # Check if the intial y-coordinate of the mid-point of both shoulders of the person has a value.
        if MID_Y:

            # Get posture (jumping, crouching or standing) of the person in the frame.
            frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)

            # Check if the person has jumped.
            if posture == "Jumping" and y_pos_index == 1:
                print("UP Sent")
                # Press the up arrow key
                pyautogui.press("up")
                # Update the veritcal position index of the character.
                y_pos_index += 1

            # Check if the person has crouched.
            elif posture == "Crouching" and y_pos_index == 1:
                print("DOWN SENT")
                # Press the down arrow key
                pyautogui.press("down")
                # Update the veritcal position index of the character.
                y_pos_index -= 1

            # Check if the person has stood.
            elif posture == "Standing" and y_pos_index != 1:

                # Update the vertical position index of the character.
                y_pos_index = 1

    # Otherwise if the pose Landmarks in the frame are not detected.
    else:

        # Update the counter value to zero.
        counter = 0

    # Calculate the frames updates in one second

    # Set the time for this frame to the current time.
    time2 = time()

    # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
    if (time2 - time1) > 0:

        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)

        # write the calculated number of frames per second on the frame.
        cv2.putText(
            frame,
            "FPS: {}".format(int(frames_per_second)),
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 255, 0),
            3,
        )

    # Update the previous frame time to this frame time.
    # As this frame will become
    time1 = time2

    # Display the frame.
    cv2.imshow("Subway Surfers with Pose Detection", frame)

    # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if k == 27:
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
