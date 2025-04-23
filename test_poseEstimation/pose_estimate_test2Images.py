'''

OBJ_DETECTION AND POSE_ESTIMATE INTEGRATION
-Input:  screen record and send video to our model.  Let model process video using OpenCV to get
individual image frames
-Object detection output:  Calculate overlap of gun and human bounding boxes.  Then output Threat
leve1 0, 1, or 2, WITH Bounding boxes that are labeled with gun type (list data structure?).
    -for each human identified, if theres a gun overlap (threat level 2), make
    threat_list = [(human1_bbox, threat_level, gun1_bbox,  gun1_type)]
-Integration of object detection and pose estimation, output:  crop images and increase or
decrease image size.  Then output Threat leve1 2 or 3, WITH Bounding boxes.

--Also have to adjust picture size depending on how far or close individual is

'''

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = YOLO('yolov8x-pose.pt')  # or yolov8n-pose.pt for lightweight

# Load image
img_path = "man-pointing-with-gun.jpg"
image = cv2.imread(img_path)

img_path2 = "man-poinging-with-his-machine-gun.jpg"
image2 = cv2.imread(img_path2)

# Predict
results = model.predict(image, save=False)

results2 = model.predict(image2, save=False)

# COCO skeleton
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# 🚨 Improved Threat Detection Logic
def is_threatening_pose_directional(keypoints):
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    def direction_vector(a, b):
        v = b - a
        return v / (np.linalg.norm(v) + 1e-6)

    lw_to_nose = direction_vector(left_wrist, nose)
    rw_to_nose = direction_vector(right_wrist, nose)
    le_to_lw = direction_vector(left_elbow, left_wrist)
    re_to_rw = direction_vector(right_elbow, right_wrist)

    left_dot = np.dot(lw_to_nose, le_to_lw)
    right_dot = np.dot(rw_to_nose, re_to_rw)

    def arm_angle(shoulder, elbow, wrist):
        a = shoulder - elbow
        b = wrist - elbow
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    left_angle = arm_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = arm_angle(right_shoulder, right_elbow, right_wrist)

    left_threat = left_angle >= 160 and left_dot >= 0.7
    right_threat = right_angle >= 160 and right_dot >= 0.7

    main_threat = left_threat or right_threat

    # Backup rule
    wrist_dist = np.linalg.norm(left_wrist - right_wrist)
    horizontal_distance = abs(left_wrist[0] - right_wrist[0])
    vertical_offset_from_nose = max(abs(left_wrist[1] - nose[1]), abs(right_wrist[1] - nose[1]))

    backup_threat = (
        wrist_dist < 500 and
        horizontal_distance < 350 and
        vertical_offset_from_nose < 1000
    )

    return main_threat or backup_threat

# Draw keypoints, skeleton, and label
for person in results[0].keypoints.xy:
    keypoints = person.cpu().numpy()

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        color = (0, 255, 255)  # Default yellow
        if i in [5, 6]: color = (0, 255, 0)     # shoulders
        if i in [7, 8]: color = (0, 0, 255)     # elbows
        if i in [9, 10]: color = (255, 0, 0)    # wrists
        cv2.circle(image, (int(x), int(y)), 10, color, -1)
        cv2.putText(image, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw skeleton
    for i, j in skeleton:
        pt1 = tuple(np.round(keypoints[i]).astype(int))
        pt2 = tuple(np.round(keypoints[j]).astype(int))
        cv2.line(image, pt1, pt2, (255, 255, 255), 3)

    # Threat label
    label = "THREAT" if is_threatening_pose_directional(keypoints) else "NO THREAT"
    label_color = (0, 0, 255) if label == "THREAT" else (0, 255, 0)
    x_text, y_text = int(keypoints[0][0]), int(keypoints[0][1])
    label_y = max(50, int(y_text) - 500)  # shift upward but not off-screen
    cv2.putText(image, label, (x_text, label_y),
            cv2.FONT_HERSHEY_DUPLEX, 2.5, label_color, 5)

# Draw keypoints, skeleton, and label
for person in results2[0].keypoints.xy:
    keypoints = person.cpu().numpy()

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        color = (0, 255, 255)  # Default yellow
        if i in [5, 6]: color = (0, 255, 0)     # shoulders
        if i in [7, 8]: color = (0, 0, 255)     # elbows
        if i in [9, 10]: color = (255, 0, 0)    # wrists
        cv2.circle(image2, (int(x), int(y)), 10, color, -1)
        cv2.putText(image2, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw skeleton
    for i, j in skeleton:
        pt1 = tuple(np.round(keypoints[i]).astype(int))
        pt2 = tuple(np.round(keypoints[j]).astype(int))
        cv2.line(image2, pt1, pt2, (255, 255, 255), 3)

    # Threat label
    label = "THREAT" if is_threatening_pose_directional(keypoints) else "NO THREAT"
    label_color = (0, 0, 255) if label == "THREAT" else (0, 255, 0)
    x_text, y_text = int(keypoints[0][0]), int(keypoints[0][1])
    label_y = max(50, int(y_text) - 500)  # shift upward but not off-screen
    cv2.putText(image2, label, (x_text, label_y),
            cv2.FONT_HERSHEY_DUPLEX, 2.5, label_color, 5)


# Convert to RGB and show
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14, 10))
plt.imshow(image_rgb)
plt.title("YOLOv8 Pose + Threat Detection", fontsize=20)
plt.axis('off')
plt.show()

image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14, 10))
plt.imshow(image2_rgb)
plt.title("YOLOv8 Pose + Threat Detection", fontsize=20)
plt.axis('off')
plt.show()