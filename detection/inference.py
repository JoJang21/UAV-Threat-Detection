from ultralytics import YOLO
import glob, os
import cv2
from shapely.geometry import box

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GUN_MODEL_PATH    = 'runs/detect/gun_detection7/weights/best.pt'
PERSON_MODEL_PATH = 'yolov8m.pt'    # COCO‑pretrained
IMAGE_DIR         = './testset/train/images'
OUTPUT_DIR        = './testset/train/merged_output'
IOU_THRESHOLD     = 0.3
CONF_GUN          = 0.5
CONF_PERSON       = 0.5
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load models
gun_model    = YOLO(GUN_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

def iou(a, b):
    """Compute IoU between two xyxy boxes."""
    A = box(*a)
    B = box(*b)
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter / union if union > 0 else 0

for img_path in glob.glob(os.path.join(IMAGE_DIR, '*.jpg')):
    # 1) inference
    gres = gun_model.predict(source=img_path,    conf=CONF_GUN,    classes=[0])
    pres = person_model.predict(source=img_path, conf=CONF_PERSON, classes=[0])

    gun_boxes    = gres[0].boxes.xyxy.tolist()  if len(gres[0].boxes)    else []
    person_boxes = pres[0].boxes.xyxy.tolist() if len(pres[0].boxes) else []

    # load image up front so we can always save it
    img = cv2.imread(img_path)

    # 2a) draw gun boxes (red) wherever they exist
    for gbox in gun_boxes:
        x1, y1, x2, y2 = map(int, gbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(img, 'gun', (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # 2b) draw person boxes (blue) wherever they exist
    for pbox in person_boxes:
        px1, py1, px2, py2 = map(int, pbox)
        cv2.rectangle(img, (px1, py1), (px2, py2), (255,0,0), 2)
        cv2.putText(img, 'person', (px1, py1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # 2c) if you still want to do the “fused” match (gun→best person):
    for gbox in gun_boxes:
        # pick best person by IoU, if any exist
        if person_boxes:
            best_p = max(person_boxes, key=lambda p: iou(gbox,p))
            if iou(gbox, best_p) > IOU_THRESHOLD:
                # draw a thick yellow line between their centers, or annotate however you like
                gx, gy = int((gbox[0]+gbox[2])/2), int((gbox[1]+gbox[3])/2)
                px, py = int((best_p[0]+best_p[2])/2), int((best_p[1]+best_p[3])/2)
                cv2.line(img, (gx,gy), (px,py), (0,255,255), 1)

    # 3) save **every** image
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"[✓] Saved {out_path}  (guns: {len(gun_boxes)}, people: {len(person_boxes)})")

