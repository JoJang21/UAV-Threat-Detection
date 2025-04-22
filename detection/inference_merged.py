from ultralytics import YOLO
import glob, os
import cv2
from shapely.geometry import box

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GUN_MODEL_PATH    = 'runs/detect/gun_detection7/weights/last.pt'
PERSON_MODEL_PATH = 'yolov8l.pt'    # COCO‑pretrained
IMAGE_DIR         = './testset/train/images'
OUTPUT_DIR        = './testset/train/merged_output'
IOU_THRESHOLD     = 0.01
CONF_GUN          = 0.5
CONF_PERSON       = 0.5
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

gun_model    = YOLO(GUN_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

def iou(a, b):
    A = box(*a); B = box(*b)
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter/union if union>0 else 0

for img_path in glob.glob(os.path.join(IMAGE_DIR, '*.jpg')):
    # 1) Inference
    gres = gun_model.predict(source=img_path,    conf=CONF_GUN,    classes=[1,2])
    pres = person_model.predict(source=img_path, conf=CONF_PERSON, classes=[0])
    gun_boxes    = gres[0].boxes.xyxy.tolist()    if gres[0].boxes else []
    person_boxes = pres[0].boxes.xyxy.tolist()   if pres[0].boxes else []

    img = cv2.imread(img_path)
    merged_boxes = []

    # 2) For each gun, find its best person and union their boxes
    for g in gun_boxes:
        if not person_boxes:
            continue
        p = max(person_boxes, key=lambda pb: iou(g, pb))
        if iou(g, p) < IOU_THRESHOLD:
            continue

        # compute union box
        x1 = min(g[0], p[0])
        y1 = min(g[1], p[1])
        x2 = max(g[2], p[2])
        y2 = max(g[3], p[3])
        merged_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    # 3) Draw merged boxes (green) and fallback individual boxes if you like
    for (x1,y1,x2,y2) in merged_boxes:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, 'armed_person', (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Optional: if you want to see original separate boxes, uncomment:
    # for g in gun_boxes:
    #     x1,y1,x2,y2 = map(int, g)
    #     cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
    # for p in person_boxes:
    #     x1,y1,x2,y2 = map(int, p)
    #     cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)

    # 4) Save
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"[✓] {os.path.basename(img_path)} → {len(merged_boxes)} armed_person boxes")
