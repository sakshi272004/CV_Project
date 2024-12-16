import os
import math
from ultralytics import YOLO
import cv2

# Ensure output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

# Correct video file path
video_path = "../Videos/ppe-1-1.mp4"
cap = cv2.VideoCapture(video_path)  # Use 0 for default webcam, change to 1 for secondary camera

if not cap.isOpened():
    print(f"Error: Unable to open video file {os.path.abspath(video_path)}")
    exit()

frame_width = int(cap.get(3))  # Frame width
frame_height = int(cap.get(4))  # Frame height

# Output video writer
out = cv2.VideoWriter('output/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Load YOLO model
model = YOLO("ppe.pt")  # Replace "ppe.pt" with your model path

# Define class names and colors for bounding boxes
classColors = {
    'Hardhat': (0, 255, 0),
    'Mask': (255, 255, 0),
    'NO-Hardhat': (0, 0, 255),
    'NO-Mask': (255, 0, 0),
    'Safety Vest': (0, 255, 255),
}

# Fallback class names if the model doesn't provide them
classNames = model.names if hasattr(model, 'names') else ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'Safety Vest']

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame for efficiency
    img = cv2.resize(img, (640, 480))

    # Perform detection
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence score
            cls = int(box.cls[0])  # Class ID
            class_name = classNames[cls] if cls < len(classNames) else 'Unknown'

            # Set color for bounding box
            mycolor = classColors.get(class_name, (255, 0, 0))

            # Filter detections by confidence threshold
            if conf > 0.5:
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), mycolor, 3)
                
                # Label with class and confidence
                label = f'{class_name} {conf:.2f}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # Save frame with violations
                if "NO-" in class_name:
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.imwrite(f"output/violation_frame_{frame_number}.jpg", img)

    # Write output frame
    out.write(img)

    # Display the frame
    cv2.imshow("Image", img)

    # Exit on pressing '1'
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
