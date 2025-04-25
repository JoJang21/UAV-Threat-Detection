from ultralytics import YOLO
import glob, os
import cv2
from shapely.geometry import box

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GUN_MODEL_PATH    = 'runs/detect/gun_detection7/weights/last.pt'
PERSON_MODEL_PATH = 'yolov8xl.pt'    # COCO-pretrained  l.pt
IMAGE_DIR         = './testset/train/images'
OUTPUT_DIR        = './testset/train/merged_output'
IOU_THRESHOLD     = 0.01
CONF_GUN          = 0.5
CONF_PERSON       = 0.5
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

curr_dir = os.getcwd()

FULL_GUN_MODEL_PATH = os.path.join(curr_dir, GUN_MODEL_PATH)
FULL_PERSON_MODEL_PATH = os.path.join(curr_dir, PERSON_MODEL_PATH)
gun_model    = YOLO(FULL_GUN_MODEL_PATH)
person_model = YOLO(FULL_PERSON_MODEL_PATH)

IMAGE_DIR = os.path.join(curr_dir, IMAGE_DIR)
OUTPUT_DIR = os.path.join(curr_dir, OUTPUT_DIR)

print(curr_dir)
print(FULL_GUN_MODEL_PATH)
print(FULL_PERSON_MODEL_PATH)
print(IMAGE_DIR)
print(OUTPUT_DIR)

def iou(a, b):
    A = box(*a); B = box(*b)
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter/union if union>0 else 0

for img_path in glob.glob(os.path.join(IMAGE_DIR, '*.jpg')):
    # 1) Inference
    gres = gun_model.predict(source=img_path,    conf=CONF_GUN)#,    classes=[1,2])
    pres = person_model.predict(source=img_path, conf=CONF_PERSON)#, classes=[0])

    # pair up each gun box with its predicted class (1=pistol, 2=rifle)
    gun_boxes = []
    for box_tensor, cls_tensor in zip(gres[0].boxes.xyxy, gres[0].boxes.cls):
        xyxy = box_tensor.tolist()
        cls  = int(cls_tensor.item())
        gun_boxes.append((xyxy, cls))

    person_boxes = pres[0].boxes.xyxy.tolist() if pres[0].boxes else []

    img = cv2.imread(img_path)
    merged = []

    # 2) Match & union
    for g_xyxy, g_cls in gun_boxes:
        print("g_cls: ", g_cls)
        if not person_boxes:
            continue
        # find the person with highest IoU
        best_p = max(person_boxes, key=lambda p: iou(g_xyxy, p))
        if iou(g_xyxy, best_p) < IOU_THRESHOLD:
            continue

        # union the gun+person box
        x1 = min(g_xyxy[0], best_p[0])
        y1 = min(g_xyxy[1], best_p[1])
        x2 = max(g_xyxy[2], best_p[2])
        y2 = max(g_xyxy[3], best_p[3])

        merged.append((int(x1), int(y1), int(x2), int(y2), g_cls))

    # 3) Draw
    for x1, y1, x2, y2, g_cls in merged:
        if g_cls == 1:
            label = 'armed_person_pistol'
            color = (0,200,0)     # dark green
        elif g_cls==2:
            label = 'armed_person_rifle'
            color = (0,255,0)     # bright green
        else:
            label = 'unarmed_person'
            color = (0,0,255)     # red

        # 1) measure text size
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale= 0.5
        thickness = 1
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 2) choose text origin INSIDE the box
        text_x = x1 + 2
        text_y = y1 + h + 2   # a couple pixels below the top edge

        # 3) draw a filled rect as text background
        cv2.rectangle(
            img,
            (x1, y1),                     # top-left
            (x1 + w + 4, y1 + h + 4),     # bottom-right
            color,                        # same color as border
            thickness=-1                  # filled
        )

        # 4) draw the box
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

        # 5) draw the label over that background
        cv2.putText(
            img, label,
            (text_x, text_y),
            font, font_scale,
            (0,0,0),  # black text, white text is (255,255,255)
            thickness,
            lineType=cv2.LINE_AA
        )

    # optionally draw any unmatched guns/people in thinner boxes…

    print(merged)
    # 4) Save
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"[✓] {os.path.basename(img_path)} → {len(merged)} armed boxes")
