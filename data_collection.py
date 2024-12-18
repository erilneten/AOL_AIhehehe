import cv2
import mediapipe as mp
import time
import threading
import queue

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1)

# Queue for frames
frame_queue = queue.Queue()

# Video Capture Function in a Separate Thread
def video_capture_thread(queue, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            queue.put(frame)
        else:
            break

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Start the video capture thread
thread = threading.Thread(target=video_capture_thread, args=(frame_queue, cap))
thread.daemon = True
thread.start()

# File for saving pose data
output_file = 'manual_pose.csv'
label = "Working"  # Default label
batch_size = 20
buffer = []
frame_skip = 5  # Process every 5th frame
frame_count = 0

# Open CSV file for writing pose landmarks
with open(output_file, 'w') as file:
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_count += 1

            # Skip frames to reduce load
            if frame_count % frame_skip != 0:
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (320, 240))

            # Convert to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            # Process landmarks if detected
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                row = []
                for landmark in landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])
                row.insert(0, label)  # Add the current label to the row
                buffer.append(row)

                # Draw landmarks on frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Write data to file in batches
            if len(buffer) >= batch_size:
                with open(output_file, 'a') as file:
                    for row in buffer:
                        file.write(','.join(map(str, row)) + '\n')
                buffer.clear()

            # Display frame with landmarks and current label
            cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Manual Pose Labeling", frame)

        # Toggle label with keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            label = "Working"
        elif key == ord('s'):
            label = "Slacking"
        elif key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
