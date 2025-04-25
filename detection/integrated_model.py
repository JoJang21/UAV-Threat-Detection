from ultralytics import YOLO
import glob, os
import cv2
from shapely.geometry import box
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageEnhance
import pose_estimator as pest

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GUN_MODEL_PATH    = 'runs/detect/gun_detection7/weights/last.pt'
PERSON_MODEL_PATH = 'yolov8l.pt'    # COCO-pretrained, HOW TO GET xl.pt?????????
IMAGE_DIR         = './testset/train/images'
OUTPUT_DIR        = './testset/train/merged_output'
OBJ_DETECT_DIR    = 'obj_detect_imgs'
IOU_THRESHOLD     = 0.01
CONF_GUN          = 0.5
CONF_PERSON       = 0.5



folder_path = "test_imgs/"
agumented_dir = "augmented_imgs"

input_video = "test_vid_2.MOV" # "test_vid.mp4" #  
frame_dir = "frames"
results_dir = "results"
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

curr_dir = os.getcwd()

FULL_GUN_MODEL_PATH = os.path.join(curr_dir, GUN_MODEL_PATH)
FULL_PERSON_MODEL_PATH = os.path.join(curr_dir, PERSON_MODEL_PATH)
gun_model    = YOLO(FULL_GUN_MODEL_PATH)
person_model = YOLO(FULL_PERSON_MODEL_PATH)

IMAGE_DIR = os.path.join(curr_dir, IMAGE_DIR)
OUTPUT_DIR = os.path.join(curr_dir, OUTPUT_DIR)
OBJ_DETECT_DIR = os.path.join(curr_dir, OBJ_DETECT_DIR)
RESULTS_DIR = os.path.join(curr_dir, results_dir)

CROP_IMGS_DIR = os.path.join(curr_dir, "crop_imgs")
FRAMES_DIR = os.path.join(curr_dir, frame_dir)

print(curr_dir)
print(FULL_GUN_MODEL_PATH)
print(FULL_PERSON_MODEL_PATH)
print(IMAGE_DIR)
print(OUTPUT_DIR)
print(OBJ_DETECT_DIR)

def iou(a, b):
    A = box(*a); B = box(*b)
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter/union if union>0 else 0


def analyze_frames2(folder_path, verbose):
    for img_path in glob.glob(os.path.join(folder_path, '*.png')):#jpg')):
        frame_img = cv2.imread(img_path) 
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        sharpened_image = cv2.filter2D(frame_img, -1, kernel) 
        cv2.imwrite(img_path, sharpened_image)
        
  
        # 1) Inference
        pres = person_model.predict(source=img_path, conf=CONF_PERSON, classes=[0])
        print("PRES: ", pres[0].boxes)
        gres = gun_model.predict(source=img_path,    conf=CONF_GUN,    classes=[1,2])
        print("GRES", gres[0].boxes.xyxy, gres[0].boxes.cls)
        

        # pair up each gun box with its predicted class (1=pistol, 2=rifle)
        gun_boxes = []
        person_boxes = pres[0].boxes.xyxy.tolist() if pres[0].boxes else []
        #for box_tensor, cls_tensor in zip(gres[0].boxes.xyxy, gres[0].boxes.cls):
        merged = []
        for person in person_boxes:
            print("PERSON", person)
            merged.append((int(person[0]), int(person[1]), int(person[2]), int(person[3]), 2))

        img = cv2.imread(img_path)
        
        '''
        # 2) Match & union
        for g_xyxy, g_cls in gun_boxes:
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
        '''
        
        # 3) Draw
        print("IMG_PATH", img_path)
        print("MERGED", merged)
        for x1, y1, x2, y2, g_cls in merged:
            if g_cls == 1:
                label = 'armed_person_pistol'
                color = (255,0,255)     # pink
            elif g_cls==2:
                label = 'armed_person_rifle'
                color = (127,0,255)     # purple
            else:
                label = 'unarmed_person'
                color = (96,96,96)     # gray
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
                (255,255,255), #(0,0,0),  # black text, white text is 
                thickness,
                lineType=cv2.LINE_AA
            )
        # optionally draw any unmatched guns/people in thinner boxes…
        # 4) Save
        obj_det_path = os.path.join(OBJ_DETECT_DIR, os.path.basename(img_path))
        cv2.imwrite(obj_det_path, img)
        print(f"[✓] {os.path.basename(img_path)} → {len(merged)} armed boxes")
      
      
      
      
        threat_level, aim_list, pose_img_path = 0, [], img_path
        # Print Threat Level at top of Frame
        if (len(merged) > 0):
            threat_level, aim_list, pose_img_path = pest.pose_process(img_path, merged, RESULTS_DIR, verbose)
            print(pose_img_path)
            img = cv2.imread(pose_img_path)
        else:
            img_basename = os.path.basename(img_path)
            pose_img_path = os.path.join(RESULTS_DIR, f"pose_{img_basename}")
        img_height, img_width, channels = img.shape
        threat_level_txt = f"Threat Level {threat_level}: "
        add_txt = ""
        threat_level_color = (255,255,255) #white
        threat_text_color = (0,0,0) #black
        match threat_level:
            case 0:
                add_txt = "No Threat"
                threat_level_color = (0,204,0) # green
            case 1:
                add_txt = "Caution"
                threat_level_color = (0,255,255) # yellow
            case 2:
                add_txt = "Warning"
                threat_level_color = (0,170,255) # light orange
            case 3:
                add_txt = "DANGER"
                threat_level_color = (0,85,255) # dark orange
                threat_text_color = (255,255,255)
            case 4:
                add_txt = "CRITICAL"
                threat_level_color = (0,0,255) # red
                threat_text_color = (255,255,255)
            case _:
                add_txt = "No Threat"
                threat_level_color = (0,0,0)
    
        threat_level_txt = threat_level_txt + add_txt
        text_width, text_height = cv2.getTextSize(threat_level_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        textX = img_width//2
        cv2.rectangle(img, (textX - text_width//2, 0), (textX + text_width//2, text_height + 5), threat_level_color, -1)
        cv2.putText(
                img, threat_level_txt,
                (textX - text_width//3, text_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                threat_text_color,
                2,
                lineType=cv2.LINE_AA
        )
        
        # draw box around n-aim vs yes-aim individuals
        aim_num = 0
        if (len(merged) > 0):
            for x1, y1, x2, y2, g_cls in merged:
                if aim_list[aim_num]:
                    color = (0,0,255)     # red
                else:
                    color = (0,255,255)     # yellow
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
                    (0,0,0),
                    thickness,
                    lineType=cv2.LINE_AA
                )
                aim_num += 1
        cv2.imwrite(pose_img_path, img)



def analyze_frames(folder_path, verbose):
    for img_path in glob.glob(os.path.join(folder_path, '*.png')):#jpg')):
        frame_img = cv2.imread(img_path) 
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        sharpened_image = cv2.filter2D(frame_img, -1, kernel) 
        cv2.imwrite(img_path, sharpened_image)
        
  
        # 1) Inference
        pres = person_model.predict(source=img_path, conf=CONF_PERSON, classes=[0])
        print("PRES: ", pres[0].boxes)
        gres = gun_model.predict(source=img_path,    conf=CONF_GUN,    classes=[1,2])
        print("GRES", gres[0].boxes.xyxy, gres[0].boxes.cls)
        

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
        print("IMG_PATH", img_path)
        print("MERGED", merged)
        for x1, y1, x2, y2, g_cls in merged:
            if g_cls == 1:
                label = 'armed_person_pistol'
                color = (255,0,255)     # pink
            elif g_cls==2:
                label = 'armed_person_rifle'
                color = (127,0,255)     # purple
            else:
                label = 'unarmed_person'
                color = (96,96,96)     # gray
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
                (255,255,255), #(0,0,0),  # black text, white text is 
                thickness,
                lineType=cv2.LINE_AA
            )
        # optionally draw any unmatched guns/people in thinner boxes…
        # 4) Save
        obj_det_path = os.path.join(OBJ_DETECT_DIR, os.path.basename(img_path))
        cv2.imwrite(obj_det_path, img)
        print(f"[✓] {os.path.basename(img_path)} → {len(merged)} armed boxes")
      
      
      
      
        threat_level, aim_list, pose_img_path = 0, [], img_path
        # Print Threat Level at top of Frame
        if (len(merged) > 0):
            threat_level, aim_list, pose_img_path = pest.pose_process(img_path, merged, RESULTS_DIR, verbose)
            print(pose_img_path)
            img = cv2.imread(pose_img_path)
        else:
            img_basename = os.path.basename(img_path)
            pose_img_path = os.path.join(RESULTS_DIR, f"pose_{img_basename}")
        img_height, img_width, channels = img.shape
        threat_level_txt = f"Threat Level {threat_level}: "
        add_txt = ""
        threat_level_color = (255,255,255) #white
        threat_text_color = (0,0,0) #black
        match threat_level:
            case 0:
                add_txt = "No Threat"
                threat_level_color = (0,204,0) # green
            case 1:
                add_txt = "Caution"
                threat_level_color = (0,255,255) # yellow
            case 2:
                add_txt = "Warning"
                threat_level_color = (0,170,255) # light orange
            case 3:
                add_txt = "DANGER"
                threat_level_color = (0,85,255) # dark orange
                threat_text_color = (255,255,255)
            case 4:
                add_txt = "CRITICAL"
                threat_level_color = (0,0,255) # red
                threat_text_color = (255,255,255)
            case _:
                add_txt = "No Threat"
                threat_level_color = (0,0,0)
    
        threat_level_txt = threat_level_txt + add_txt
        text_width, text_height = cv2.getTextSize(threat_level_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        textX = img_width//2
        cv2.rectangle(img, (textX - text_width//2, 0), (textX + text_width//2, text_height + 5), threat_level_color, -1)
        cv2.putText(
                img, threat_level_txt,
                (textX - text_width//3, text_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                threat_text_color,
                2,
                lineType=cv2.LINE_AA
        )
        
        # draw box around n-aim vs yes-aim individuals
        aim_num = 0
        if (len(merged) > 0):
            for x1, y1, x2, y2, g_cls in merged:
                if aim_list[aim_num]:
                    color = (0,0,255)     # red
                else:
                    color = (0,255,255)     # yellow
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
                    (0,0,0),
                    thickness,
                    lineType=cv2.LINE_AA
                )
                aim_num += 1
        cv2.imwrite(pose_img_path, img)



#os.makedirs(results_dir, exist_ok=True)
os.makedirs(OBJ_DETECT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(CROP_IMGS_DIR, exist_ok=True)

verbose = False
video = True
if video:
    # Step 1: Extract
    print("Extracting frames from video...")
    #pest.extract_frames(input_video, FRAMES_DIR)

    # Step 2: Analyze pose and threat
    print("Going through the frames")
    #analyze_frames(FRAMES_DIR, verbose)
    analyze_frames2(FRAMES_DIR, verbose)

    # Step 3: Compile into video
    print("Creating video from frames...")
    pest.create_video_from_frames(results_dir, "results_video.mp4", fps=30)
    pest.create_video_from_frames(OBJ_DETECT_DIR, "obj_det_video.mp4", fps=30)
else:
    if agument:
        augmentation_mode = "none" # Change this to the desired augmentation mode
        pest.augment_and_save_all(folder_path, agumented_dir, mode=augmentation_mode)
        analyze_frames(agumented_dir, verbose)
    else:
        analyze_frames(folder_path, verbose)