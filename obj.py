import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define classes you want to detect
class_names = ["car", "truck", "motorbike", "bicycle"]

# Load video
input_source = "dubai.mp4"  # Replace with your source
video_clip = VideoFileClip(input_source)

# Iterate over frames
for frame in video_clip.iter_frames(fps=video_clip.fps):
    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in class_names:
                # Get object coordinates
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])

                # Calculate box coordinates
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Draw bounding box and label with color
                color = (0, 255, 0)  # BGR color for the box (green in this case)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with color
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Object Detection", frame)
    
    
    

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release OpenCV windows
cv2.destroyAllWindows()
