import cv2 as cv
import numpy as np

# Distance constants
KNOWN_DISTANCE = 1.143  # meters (approximately 45 inches)
VEHICLE_WIDTH = 70     # pixels

# Object detector constants
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.1

# Colors for object detection
HIGHLIGHT_COLOR = (0, 255, 0)
DISTANCE_BOX_COLOR = (0, 0, 0)
ALERT_COLOR = (0, 0, 255)

# Load pre-trained YOLOv4-tiny model
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Set preferable backend and target for neural network
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Set up object detection model
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load test video
cap = cv.VideoCapture('video1.mp4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        # Break the loop if video has ended
        break

    # Detect objects on the road
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if classes is not None:
        for i in range(len(classes)):
            classid = int(classes[i])
            score = float(scores[i])
            box = boxes[i]
            
            if classid == 2 or classid == 7:  # Vehicle class IDs in the YOLO model
                # Calculate distance based on object width
                distance = (VEHICLE_WIDTH * KNOWN_DISTANCE) / box[2]

                # Set default color for the vehicle rectangle and header
                vehicle_color = HIGHLIGHT_COLOR
                text_color = (0, 0, 0)  # Black

                # Trigger alert if object is too close
                if distance < 1:  # Distance threshold for alert (1 meter)
                    vehicle_color = ALERT_COLOR
                    text_color = (0, 0, 255)  # Red

                # Draw rectangle around the vehicle
                cv.rectangle(frame, box, vehicle_color, 1)

                # Display the distance within a black box
                distance_text = f"{round(distance, 2)} m"
                text_size, _ = cv.getTextSize(distance_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = box[0] + int((box[2] - text_size[0]) / 2)
                text_y = box[1] - 10
                cv.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                             (text_x + text_size[0] + 2, text_y + 2), DISTANCE_BOX_COLOR, -1)
                cv.putText(frame, distance_text, (text_x, text_y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Add "alert" tag when vehicle is too close
                if distance < 1:
                    alert_text = "ALERT!"
                    alert_size, _ = cv.getTextSize(alert_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    alert_x = box[0] + int((box[2] - alert_size[0]) / 2)
                    alert_y = box[1] + box[3] + 20
                    cv.rectangle(frame, (alert_x - 2, alert_y - alert_size[1] - 2),
                                 (alert_x + alert_size[0] + 2, alert_y + 2), DISTANCE_BOX_COLOR, -1)
                    cv.putText(frame, alert_text, (alert_x, alert_y),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv.LINE_AA)

    # Display the frame with detected objects and distances
    cv.imshow('Road Object Detection', frame)
    
    # Exit loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv.destroyAllWindows()
