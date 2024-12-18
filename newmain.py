import cv2
import mediapipe as mp
import pickle
import time

# Load the trained model
with open("pose_classifier.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Video capture
cap = cv2.VideoCapture(0)  # Use webcam or video file

# Timer variables
slack_timer_start = None
away_timer_start = None
slack_threshold = 30  # 30 seconds of slacking
away_threshold = 30  # 30 seconds away from camera
reset_time = 600  # Reset every 10 minutes (600 seconds)
administrator_notified = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # Get image dimensions
    height, width, _ = frame.shape

    # Initialize box color and message
    box_color = (0, 255, 0)  # Green (working)
    message = "Working"
    slack_timer_display = ""

    # Reset the slack timer and away timer every 10 minutes
    if time.time() - slack_timer_start > reset_time:
        slack_timer_start = None
        away_timer_start = None

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Extract features for prediction
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        # Predict label
        predicted_label = model.predict([features])[0]

        # Check if the predicted label is "Slacking"
        if predicted_label == "Slacking":
            if slack_timer_start is None:
                slack_timer_start = time.time()
            elif time.time() - slack_timer_start > slack_threshold:
                # After 30 seconds, change box color to red and display the message
                box_color = (0, 0, 255)  # Red
                message = "Administrator notified: Staff is slacking!"
                slack_timer_display = f"Slack Time: {int(time.time() - slack_timer_start)}s"
                if not administrator_notified:
                    print("Administrator notified: Staff is slacking!")
                    administrator_notified = True
        else:
            # Reset timer and message if not slacking
            slack_timer_start = None
            slack_timer_display = ""

        # Get bounding box coordinates for visual feedback
        x_coords = [lm.x * width for lm in landmarks]
        y_coords = [lm.y * height for lm in landmarks]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Draw bounding box around the person
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

        # Draw pose landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the message and timer
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    if slack_timer_display:
        cv2.putText(frame, slack_timer_display, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if the person is away from the camera
    if not result.pose_landmarks:
        if away_timer_start is None:
            away_timer_start = time.time()
        elif time.time() - away_timer_start > away_threshold:
            box_color = (0, 0, 255)
            message = "STAFF AWAY FROM CAMERA"
            slack_timer_display = ""
            if not administrator_notified:
                print("Administrator notified: Staff is away from camera!")
                administrator_notified = True

    # Display the video feed
    cv2.imshow("Pose Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
