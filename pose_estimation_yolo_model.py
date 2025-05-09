# -*- coding: utf-8 -*-
"""Pose_Estimation_Yolo_Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kf7yY3Eqxp0EszsvVAy_lIN-OcWxEVf0
"""



"""# Trying with Man with a gun"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageEnhance

#variable to set if augmenting images or not
agument = False
video = True


# Load the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Run pose detection on both uploaded images
'''
image1_path = "r2na_9.png"
image2_path = "r2ya_10.png"

results1 = model(image1_path)
results2 = model(image2_path)
'''

# Function to visualize result
def show_pose_result(result, title, save_path=None):
    plotted_img = result[0].plot()
    
    # Draw custom left shoulder marker
    for person in result[0].keypoints.xy:
        keypoints = person.cpu().numpy()
        left_shoulder = keypoints[5]
        x = left_shoulder[0]
        y = left_shoulder[1]
        cv2.circle(plotted_img, (int(x), int(y)), 5, (0, 0, 255), -1)  # BGR

    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, plotted_img)
        print(f"Saved: {save_path}")

    # Display inline
    # plt.figure(figsize=(6, 8))
    # plt.title(title)
    # plt.imshow(cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

# Show both images with keypoints
#show_pose_result(results1, "Pose Estimation: Man Pointing Gun")
#show_pose_result(results2, "Pose Estimation: Man Pointing Gun (Front View)")



import os
from PIL import Image

def direction_vector(a, b):
    v = b - a
    return v# / (np.linalg.norm(v) + 1e-6)

#lw_to_nose = direction_vector(left_wrist, nose)

def arm_angle(shoulder, elbow, wrist):
    a = shoulder - elbow
    b = wrist - elbow
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def within_circle(circle_center, point, circle_radius):
    distance = math.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)
    return distance < circle_radius

def find_line(px, py, qx, qy):
    if px == qx:
        return None, None  # Vertical line, slope is undefined

    m = (py - qy) / (px - qx)
    b = py - m * px
    return m, b

def find_intersection(m1, b1, m2, b2):
    if m1 == m2:
        return None, None  # Lines are parallel

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def extract_ground_truth_from_name(name):
    name = name.lower()
    if "level1" in name:
        return 1
    elif "level2" in name:
        return 2
    elif "level3" in name:
        return 3
    return None  # Unknown or unlabeled

def augment_image(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    augmentations = [
        img,  # Original
        np.array(img_pil.resize((int(img.shape[1]*1.2), int(img.shape[0]*1.2)))),  # Upscale
        np.array(img_pil.resize((int(img.shape[1]*0.8), int(img.shape[0]*0.8)))),  # Downscale
        np.array(ImageEnhance.Brightness(img_pil).enhance(0.5)),  # Darken
        np.array(ImageEnhance.Brightness(img_pil).enhance(1.5)),  # Lighten
    ]
    
    return [cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) for img_np in augmentations]


def augment_image(img, mode="none"):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    w, h = img_pil.size

    if mode == "none":
        aug = img_pil
    elif mode == "upscale":
        aug = img_pil.resize((int(w * 1.2), int(h * 1.2)))
    elif mode == "downscale":
        aug = img_pil.resize((int(w * 0.8), int(h * 0.8)))
    elif mode == "darken":
        aug = ImageEnhance.Brightness(img_pil).enhance(0.5)
    elif mode == "lighten":
        aug = ImageEnhance.Brightness(img_pil).enhance(1.5)
    elif mode == "contrast_up":
        aug = ImageEnhance.Contrast(img_pil).enhance(1.5)
    elif mode == "contrast_down":
        aug = ImageEnhance.Contrast(img_pil).enhance(0.7)
    elif mode == "rotate15":
        aug = img_pil.rotate(15)
    elif mode == "rotate-15":
        aug = img_pil.rotate(-15)
    elif mode == "sharpen":
        aug = ImageEnhance.Sharpness(img_pil).enhance(2.0)
    elif mode == "blur":
        aug = ImageEnhance.Sharpness(img_pil).enhance(0.5)
    elif mode == "desaturate":
        aug = ImageEnhance.Color(img_pil).enhance(0.5)
    elif mode == "saturate":
        aug = ImageEnhance.Color(img_pil).enhance(1.5)
    elif mode == "multiple":
        aug = img_pil.resize((int(w * 1.1), int(h * 1.1)))  # Slight upscale
        aug = ImageEnhance.Brightness(aug).enhance(1.2)     # Slight brighten
        aug = ImageEnhance.Contrast(aug).enhance(1.3)       # Contrast boost
        aug = ImageEnhance.Sharpness(aug).enhance(1.5)      # Sharpen slightly
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return cv2.cvtColor(np.array(aug), cv2.COLOR_RGB2BGR)

def augment_and_save_all(input_dir, output_dir, mode="none"):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])

    for filename in images:
        path = os.path.join(input_dir, filename)
        img_np = cv2.imread(path)

        if img_np is None:
            print(f"Warning: Failed to read image {filename}")
            continue

        aug_img = augment_image(img_np, mode=mode)
        base_name = os.path.splitext(filename)[0]
        aug_name = f"{base_name}_aug_{mode}.png"
        aug_path = os.path.join(output_dir, aug_name)
        cv2.imwrite(aug_path, aug_img)
        print(f"Saved: {aug_path}")

def extract_frames(input_video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:05d}.png"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to: {output_folder}")


def create_video_from_frames(input_folder, output_video_path, fps=30):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.png')])

    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {input_folder}")

    first_frame_path = os.path.join(input_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        raise RuntimeError(f"Could not load the first frame: {first_frame_path}")

    height, width, _ = first_frame.shape
    print(f"Frame resolution detected: {width}x{height}, {len(frame_files)} frames total")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    written = 0
    for f in frame_files:
        path = os.path.join(input_folder, f)
        frame = cv2.imread(path)

        if frame is None:
            print(f"Skipping unreadable frame: {f}")
            continue

        if frame.shape[0] != height or frame.shape[1] != width:
            print(f"Skipping mismatched resolution: {f}")
            continue

        out.write(frame)
        written += 1

    out.release()

    if written == 0:
        raise RuntimeError("No valid frames written to video.")

    print(f"Video successfully created at: {output_video_path} with {written} frames.")
   

def iterate_images(folder_path, verbose):
    """Iterates through all image files in a folder and prints their paths and sizes.

    Args:
        folder_path: The path to the folder containing the images.
    """
    num_imgs = 0
    all_imgs = []
    for filename in os.listdir(folder_path):
        num_imgs+=1
        all_imgs+=[filename]
    all_imgs.sort(reverse=True)
    correct = 0 
    total = 0

    for filename in all_imgs:
        if filename.lower().endswith(('.png')):#, '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
                img_name = os.path.basename(image_path)
                print(f"Image: {img_name}, Size: {img.size}")
                #display(img)
                result = model(image_path)
                save_path = os.path.join(save_dir, f"pose_{img_name}")

                keypoints_all = result[0].keypoints.xy
                if len(keypoints_all) == 0 or all(p.shape[0] < 6 for p in keypoints_all):
                    print(f"No detections in {img_name}. Saving original image.")
                    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    cv2.putText(img_cv2, "No person detected", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(save_path, img_cv2)
                    continue  # Skip the rest of the logic
                else:
                    # Save annotated image with pose skeleton
                    show_pose_result(result, img_name, save_path=save_path)

                gt_level = extract_ground_truth_from_name(img_name)
                total += 1
                predicted = None

                for person in keypoints_all:
                    keypoints = person.cpu().numpy()
                    if keypoints.shape[0] < 11:
                        print(f"Incomplete keypoints in {img_name}, skipping person")
                        continue

                    l_shoulder = keypoints[5]
                    r_shoulder = keypoints[6]
                    l_elbow = keypoints[7]
                    r_elbow = keypoints[8]
                    l_wrist = keypoints[9]
                    r_wrist = keypoints[10]
                    l_eye = keypoints[1]
                    r_eye = keypoints[2]
                    l_ear = keypoints[3]
                    r_ear = keypoints[4]
                    x = l_shoulder[0]
                    y = l_shoulder[1]
                    if verbose:
                        print(f"Left Shoulder: {l_shoulder}")
                        print(f"Right Shoulder: {r_shoulder}")
                        print(f"Left Elbow: {l_elbow}")
                        print(f"Right Elbow: {r_elbow}")
                        print(f"Left Wrist: {l_wrist}")
                        print(f"Right Wrist: {r_wrist}")
                        print(f"Left Eye: {l_eye}")
                        print(f"Right Eye: {r_eye}")
                        print(f"Left Ear: {l_ear}")
                        print(f"Right Ear: {r_ear}")
                        print("x, y: ", x, " ", y)

                    one_hand = False
                    pistol_hand = "LEFT"
                    l_elbow_exists = (l_elbow[0] != 0 or l_elbow[1] != 0)
                    r_elbow_exists = (r_elbow[0] != 0 or r_elbow[1] != 0)
                    r_wrist_exists = (r_wrist[0] != 0 or r_wrist[1] != 0)
                    l_wrist_exists = (l_wrist[0] != 0 or l_wrist[1] != 0)
                    l_eye_exists = (l_eye[0] != 0 or l_eye[1] != 0)
                    r_eye_exists = (r_eye[0] != 0 or r_eye[1] != 0)
                    l_ear_exists = (l_ear[0] != 0 or l_ear[1] != 0)
                    r_ear_exists = (r_ear[0] != 0 or r_ear[1] != 0)

                    l_arm_exists = l_elbow_exists and l_wrist_exists
                    r_arm_exists = r_elbow_exists and r_wrist_exists

                    l_angle = None
                    r_angle = None
                    if l_arm_exists:
                        l_angle = arm_angle(l_shoulder, l_elbow, l_wrist)
                    else:
                        one_hand = True
                    if r_arm_exists:
                        r_angle = arm_angle(r_shoulder, r_elbow, r_wrist)
                    else:
                        one_hand = True

                    if verbose:
                        print("l_arm_exists: ", l_arm_exists, " l_angle: ", l_angle)
                        print("r_arm_exists: ", r_arm_exists, " r_angle: ", r_angle)
                        print("pre one_hand: ", one_hand)
                    
                    if img_name[0] == "p": #if detected gun is a pistol
                        #print("PISTOL")
                        # Differentiating between 1 hand and 2 hands:  if one arm is obtuse angle (>135), or if arm (wrist and elbow) doesnt exist

                        if (l_arm_exists and not r_arm_exists):
                            one_hand = True
                            pistol_hand = "LEFT"
                        elif (r_arm_exists and not l_arm_exists):
                            one_hand = True
                            pistol_hand = "RIGHT"
                        elif (not l_arm_exists) and (not r_arm_exists):
                            #print("THREAT LEVEL 1")
                            predicted = 1
                            continue

                        # below code will work mostly for one hand?????
                        if (l_arm_exists and l_angle > 120):
                            if (r_arm_exists and r_angle > 150):
                                one_hand = False
                                #print("THREAT LEVEL 1")
                                predicted = 1
                                #continue
                            elif (r_arm_exists and r_angle < 100):
                                one_hand = True
                                pistol_hand = "RIGHT"
                        if (r_arm_exists and r_angle > 120):
                            if (l_arm_exists and l_angle > 150):
                                one_hand = False
                                #print("THREAT LEVEL 1")
                                predicted = 1
                                #continue
                            elif (l_arm_exists and l_angle < 100):
                                one_hand = True
                                pistol_hand = "LEFT"
                        print("post one_hand: ", one_hand)
                        print("pistol_hand: ", pistol_hand)


                        # P, 1 hand, no  aim: wrist is not aligned with shoulder
                        #     wrist is certain distance away from side of eye, the wrist near eye is away from person
                        #     Code: Wrist is near opposite shoulder OR Wrist is "greater than" opposite shoulder, OR wrist is outside of same shoulder (by significant amount)
                        #                            img 8, 5, 4, 1            img 2, 7 (find better imgs)              img 1, 9, 10

                        # P, 1 hand, yes aim: wrist is aligned with shoulder (difficult, get more imgs)
                        #     Code: wrist holding gun is between shoulder and opposite eye?  or midpoint of shoulders?
                        l_mod_wrist = l_wrist[0]
                        if l_arm_exists and l_angle < 10:
                            l_mod_wrist = l_shoulder[0]
                        r_mod_wrist = r_wrist[0]
                        if r_arm_exists and r_angle < 10:
                            r_mod_wrist = r_shoulder[0]
                        if one_hand:
                            if pistol_hand == "LEFT":

                                if within_circle(r_shoulder, l_wrist, 20) or l_mod_wrist < r_shoulder[0] or l_mod_wrist > l_shoulder[0] + 20:
                                    #print("THREAT LEVEL 1")
                                    predicted = 1
                                    continue
                                elif l_mod_wrist <= l_shoulder[0] or within_circle(l_shoulder, l_wrist, 20) or (r_eye_exists and (l_mod_wrist >= r_eye[0])): #use r_ear?
                                    #print("THREAT LEVEL 3") #yes aim
                                    predicted = 3
                                    continue
                                else:
                                    #print("One Left Hand: Couldn't Determine")
                                    continue
                            elif pistol_hand == "RIGHT":

                                if within_circle(l_shoulder, r_wrist, 20) or r_mod_wrist > l_shoulder[0] or r_mod_wrist < max(r_shoulder[0] - 20, 0):
                                    #print("THREAT LEVEL 1")
                                    predicted = 1
                                    continue
                                elif r_mod_wrist >=r_shoulder[0] or within_circle(r_shoulder, r_wrist, 20) or (l_eye_exists and (r_mod_wrist <= l_eye[0])): #use l_ear?
                                    print("THREAT LEVEL 3") # yes aim
                                    predicted = 3
                                    continue
                                else:
                                    print("One Right Hand: Couldn't Determine")
                                    continue


                        # P, 2 hand, no  aim: one or both wrists (x-val) are outside of shoulders OR
                        #     midpoint of wrists is not within outer eye or ear distance AND the wrist that is outside of the eye/ear distance,
                        #     that wrist is past midpoint of shoulder (x-val)

                        # P, 2 hand, yes aim: midpoint of wrists is between eye or ear distance AND both wrists are between shoulders (x-val),
                        #     measure angle?  (should be acute for both arms)
                        else: # both hands used to hold gun
                            wrist_midpoint = r_mod_wrist + ((l_mod_wrist - r_mod_wrist)//2)
                            leftmost_facial_ftr = max(l_eye[0], l_ear[0])
                            if leftmost_facial_ftr == 0:
                                leftmost_facial_ftr = l_shoulder[0]
                            if (r_ear_exists):
                                rightmost_facial_ftr = r_ear[0]
                            elif (r_eye_exists):
                                rightmost_facial_ftr = r_eye[0]
                            else:
                                rightmost_facial_ftr = r_shoulder[0]

                            if (wrist_midpoint > leftmost_facial_ftr) or (wrist_midpoint < rightmost_facial_ftr):
                                print("THREAT LEVEL 1")
                                predicted = 1
                                continue
                            else:
                                print("THREAT LEVEL 3")
                                predicted = 3
                                continue


                            if l_angle > 120 and r_angle > 120: # ??120?? 110?  90?
                                print("THREAT LEVEL 2")
                            if l_wrist[0] < r_shoulder[0] or r_wrist[0] > l_shoulder[0] or \
                              (r_wrist[0] < max(r_shoulder[0] - 20, 0) and r_angle > 10) or (l_wrist[0] > l_shoulder[0] + 20 and l_angle > 10):
                                print("THREAT LEVEL 2")
                                print("pistol both hands, out of shoulder width")
                                continue
                            print("both hands, nothing decided????????")


                    elif img_name[0] == "r": #if detected gun is a rifle
                        print("RIFLE")
                        # Check if both wrists overlap, if one wrist doesn't exist, check if closer arm's wrist overlaps with opposite shoulder
                        # one or both wrists must be within shoulder
                        # R, 2 hand, no  aim: If both wrists are past the shoulder or if wrists are too far apart (0.6? of shoulders width)

                        # R, 2 hand, yes aim: If wrists are less than 0.6 shoulder apart
                        #    (both wrists converge to certain shoulder, both wrists are near the circle?)
                        #     Compare angles created by shoulder-elbow and elbow-wrist.  Use arm with greater angle.
                        #     if line created by wrist-elbow and shoulders intersect near shoulder, then good.
                        #  if angle created by other arm is very small (less than 30 degrees) (means wrist is close to shoulder)
                        wrists_width = np.linalg.norm(np.abs(l_wrist - r_wrist))
                        shoulders_width = np.linalg.norm(np.abs(l_shoulder - r_shoulder))
                        if verbose:
                            print("wrists_width: ", wrists_width)
                            print("shoulders_width: ", shoulders_width)
                        if wrists_width > 0.7 * shoulders_width:
                            print("THREAT LEVEL 1, r1")
                            predicted = 1
                            continue
                        if (l_wrist[0] < r_shoulder[0] and r_wrist[0] < max(r_shoulder[0] - 10, 0)) or (r_wrist[0] > l_shoulder[0] and l_wrist[0] > l_shoulder[0] + 10):
                            print("THREAT LEVEL 1, r2") # both wrists are away from shoulder,  -20? -15?
                            predicted = 1
                            # check arm angle?????
                            continue
                        if (l_wrist[0] > l_shoulder[0] + 0.4*shoulders_width) or (r_wrist[0] < max(r_shoulder[0] - 0.4*shoulders_width, 0)): #0.5? 0.6?
                            print("THREAT LEVEL 1, r3") # both wrists are away from shoulder
                            predicted = 1
                            # check arm angle?????
                            continue


                        print("Testing for threat level 3")
                        l_mod_wrist = l_wrist[0]
                        if l_arm_exists and r_arm_exists:
                            if l_angle > r_angle:
                                support_arm = "LEFT"
                            elif l_angle < r_angle:
                                support_arm = "RIGHT"

                        r_mod_wrist = r_wrist[0]
                        if l_arm_exists and l_angle < 15:
                            l_mod_wrist = l_shoulder[0]
                        r_mod_wrist = r_wrist[0]
                        if r_arm_exists and r_angle < 15:
                            r_mod_wrist = r_shoulder[0]
                        shoulder_m, shoulder_b = find_line(l_shoulder[0], -l_shoulder[1], r_shoulder[0], -r_shoulder[1])
                        # print("shoulder_m", shoulder_m, " shoulder_b: ", shoulder_b)
                        if support_arm == "LEFT":
                            support_m, support_b = find_line(l_elbow[0], -l_elbow[1], l_wrist[0], -l_wrist[1])
                        elif support_arm == "RIGHT":
                            support_m, support_b = find_line(r_elbow[0], -r_elbow[1], r_wrist[0], -r_wrist[1])
                        # print("support_m: ", support_m, " support_b: ", support_b)
                        if shoulder_b is not None and support_b is not None:
                            x, y = find_intersection(shoulder_m, shoulder_b, support_m, support_b)
                            x = abs(x)
                            y = abs(y)
                            # print("intersection: ", x, " ", y)
                            if x is not None and y is not None:
                                if support_arm == "LEFT" and within_circle(r_shoulder, (x, y), 0.33 * shoulders_width) or \
                                   support_arm == "RIGHT" and within_circle(l_shoulder, (x, y), 0.33 * shoulders_width):  #0.2?, 0.1?
                                    print("THREAT LEVEL 3")
                                    predicted = 3
                                    continue
                                elif (support_arm == "LEFT" and within_circle(r_shoulder, (x, y), 0.5 * shoulders_width) \
                                      and l_angle >= 30 and l_angle <= 70) or \
                                     (support_arm == "RIGHT" and within_circle(l_shoulder, (x, y), 0.5 * shoulders_width) \
                                      and r_angle >= 30 and r_angle <= 70):  #0.2?, 0.1?
                                    print("THREAT LEVEL 3")
                                    predicted = 3
                                    continue
                                else:
                                    print("THREAT LEVEL 1")
                                    predicted = 1
                                    continue
                correct += (predicted == gt_level)                    
                img.close()
            except Exception as e:
                print(f"Error opening {image_path}: {e}")

    accuracy_percent = (correct / total) * 100
    print(f"\n\nTotal images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"\n\nAccuracy: {accuracy_percent:.2f}%")




# Example usage:
folder_path = "test_imgs/"
save_dir = "results"
agumented_dir = "augmented_imgs"

input_video = "test_vid.mp4"
frame_dir = "frames"
final_output = "output_video.mp4"

os.makedirs(save_dir, exist_ok=True)

'''
    mode == "none"
    mode == "upscale":
    mode == "downscale":
    mode == "darken":
    mode == "lighten":
    mode == "contrast_up":
    mode == "contrast_down":
    mode == "rotate15":
    mode == "rotate-15":
    mode == "sharpen":
    mode == "blur":
    mode == "desaturate":
    mode == "saturate":
    mode == "multiple":
'''
if video:
    # Step 1: Extract
    print("Extracting frames from video...")
    extract_frames(input_video, frame_dir)

    # Step 2: Analyze pose and threat
    print("Going through the frames")
    iterate_images(frame_dir, 0)

    # Step 3: Compile into video
    print("Creating video from frames...")
    create_video_from_frames(save_dir, final_output, fps=30)
else:
    if agument:
        augmentation_mode = "none" # Change this to the desired augmentation mode
        augment_and_save_all(folder_path, agumented_dir, mode=augmentation_mode)
        iterate_images(agumented_dir, 0)
    else:
        iterate_images(folder_path, 0)