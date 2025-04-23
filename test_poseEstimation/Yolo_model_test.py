# Download an Example Image
import urllib.request

url = "https://ultralytics.com/images/bus.jpg"  # Image with people
urllib.request.urlretrieve(url, "people.jpg")

#  Run YOLOv8 Pose Model
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Load the image
img = cv2.imread("people.jpg")

# Inference
results = model(img)
res = results[0]

boxes = res.boxes.xyxy.cpu().numpy()
keypoints = res.keypoints.xy.cpu().numpy()
confidences = res.keypoints.conf.cpu().numpy()

# COCO-style keypoint connections
skeleton = [
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
    (5, 6),               # shoulders
    (11, 12),             # hips
    (5, 11), (6, 12),     # torso diagonal
    (0, 1), (1, 2), (2, 3), (3, 4),  # head
]

for idx, (box, kp, conf) in enumerate(zip(boxes, keypoints, confidences)):
    x1, y1, x2, y2 = box.astype(int)

    # Label person
    label = f"Person {idx + 1}"
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw keypoints
    for i, (x, y) in enumerate(kp):
        if conf[i] > 0.5:  # draw only confident keypoints
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Draw skeleton lines
    for i, j in skeleton:
        if conf[i] > 0.5 and conf[j] > 0.5:
            pt1 = tuple(map(int, kp[i]))
            pt2 = tuple(map(int, kp[j]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)

# Save result to file
cv2.imwrite("yolo_pose_skeleton.jpg", img)
print("✅ Saved annotated image as 'yolo_pose_skeleton.jpg'")

# (Optional) View the image if not running headless
# cv2.imshow("YOLO Pose Result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Print Keypoints & Confidence
for i, (person_kps, person_conf) in enumerate(zip(keypoints, confidences)):
    print(f"\n🧍 Person {i+1}:")
    for j, ((x, y), c) in enumerate(zip(person_kps, person_conf)):
        print(f"  Keypoint {j}: x={x:.1f}, y={y:.1f}, confidence={c:.2f}")
