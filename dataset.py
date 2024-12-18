import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Helper function to calculate the angle between shoulders
def calculate_shoulder_angle(landmarks, image_width, image_height):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    left_x, left_y = left_shoulder.x * image_width, left_shoulder.y * image_height
    right_x, right_y = right_shoulder.x * image_width, right_shoulder.y * image_height
    nose_x, nose_y = nose.x * image_width, nose.y * image_height

    # Calculate the angle formed by the shoulders and the nose
    vector1 = np.array([left_x - right_x, left_y - right_y])
    vector2 = np.array([nose_x - right_x, nose_y - right_y])
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Helper function for slacking detection
def is_slacking_advanced(landmarks, image_width, image_height):
    shoulder_angle = calculate_shoulder_angle(landmarks, image_width, image_height)
    # If shoulder angle deviates too much from expected posture
    if shoulder_angle < 140 or shoulder_angle > 180:  # Adjust thresholds based on tests
        return True
    return False

cap = cv2.VideoCapture(0)

# Timer variables
slack_timer_start = None
away_timer_start = None
cooldown_timer_start = time.time()
slack_threshold = 30  # 30 seconds of slacking
away_threshold = 30  # 30 seconds away from camera
reset_time = 600  # Reset every 10 minutes (600 seconds)
administrator_notified = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # Get image dimensions
    height, width, _ = frame.shape

    # Initialize box color and text
    box_color = (0, 255, 0)  # Green
    message = "Working"
    slack_timer_display = ""  # Default slack timer should be empty

    # Reset the slack timer and away timer every 10 minutes
    if time.time() - cooldown_timer_start > reset_time:
        slack_timer_start = None
        away_timer_start = None
        cooldown_timer_start = time.time()

    if result.pose_landmarks:
        # Extract landmarks
        landmarks = result.pose_landmarks.landmark

        # Check if the user is slacking (leaning or misaligned)
        if is_slacking_advanced(landmarks, width, height):
            if slack_timer_start is None:
                slack_timer_start = time.time()  # Start slack timer if not already started
            elif time.time() - slack_timer_start > slack_threshold:
                # Flag as slacking if 30 seconds pass
                box_color = (0, 0, 255)  # Red
                message = "Administrator notified: Staff is slacking!"
                slack_timer_display = f"Slack Time: {int(time.time() - slack_timer_start)}s"
                if not administrator_notified:
                   
                    administrator_notified = True
        else:
            # Reset slack timer if user is in proper posture
            slack_timer_start = None
            slack_timer_display = ""  # Don't show slack timer when working

        # Draw shoulder markers for visual feedback
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_x, left_y = int(left_shoulder.x * width), int(left_shoulder.y * height)
        right_x, right_y = int(right_shoulder.x * width), int(right_shoulder.y * height)
        cv2.circle(frame, (left_x, left_y), 5, (255, 0, 0), -1)  # Blue dot on left shoulder
        cv2.circle(frame, (right_x, right_y), 5, (255, 0, 0), -1)  # Blue dot on right shoulder
        cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)  # Line connecting shoulders

        # Get bounding box coordinates around head level
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * height
        head_top = int(nose_y - 120)  # Adjust to place the box higher (40px above the nose)
        head_bottom = int(nose_y + 500)  # Adjust for height (80px total height for the box)

        # Set bounding box around the head and body
        x_coords = [landmark.x * width for landmark in landmarks]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))

        # Draw the bounding box
        cv2.rectangle(frame, (x_min, head_top), (x_max, head_bottom), box_color, 2)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # If the person is in the frame, reset away timer
        away_timer_start = None
    else:
        # No landmarks detected (away from the camera)
        if away_timer_start is None:
            away_timer_start = time.time()  # Start away timer if away from camera
        elif time.time() - away_timer_start > away_threshold:
            # Flag as away if 30 seconds pass
            box_color = (0, 0, 255)  # Red
            message = "STAFF AWAY FROM CAMERA"
            slack_timer_display = ""  # Don't show slack timer when away
            if not administrator_notified:
                print("Administrator notified: Staff is away from camera!")
                administrator_notified = True

    # Display the message on the frame
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    # Display the slack timer only if slacking
    if slack_timer_display:
        cv2.putText(frame, slack_timer_display, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the away timer if away
    if away_timer_start:
        elapsed_time = int(time.time() - away_timer_start)
        cv2.putText(frame, f"Away Time: {elapsed_time}s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Staff activity detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
