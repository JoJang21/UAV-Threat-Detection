import os
import shutil
import random
from pathlib import Path


def organize_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """
    Organize dataset into YOLOv8 datapath format
    
    Args:
        images_dir: images
        labels_dir: txt annotations
        output_dir: output folder
        train_ratio: number of training data/total data (rest is val)
    """
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg'))]
    
    random.shuffle(images)
    train_files = images
    count = 0 
    # Copy training files/annotations
    for img_file in train_files:
        base_name = os.path.splitext(img_file)[0]
        txt_file = f"{base_name}.txt"
        
        label_path = os.path.join(labels_dir, txt_file)

        with open(label_path, 'r') as file:
            lines = file.readlines()

        skip = False
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            
            if parts and parts[0] == '0':
                skip = True
                break
            elif parts and (parts[0] == '1' or parts[0] == '2'): #pistol
                parts[0] = '0'
            elif parts and (parts[0] == '3' or parts[0] == '4'): #rifle
                parts[0] = '1'

            new_lines.append(' '.join(parts) + '\n')

        #skip if knife
        if skip:
            continue

        with open(label_path, 'w') as file:
            file.writelines(new_lines)
        count += 1

        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(output_dir, 'train', 'images', img_file)
        )
        shutil.copy(
                label_path,
                os.path.join(output_dir, 'train', 'labels', txt_file)
        )
    
    print(f"{count} training data")

organize_dataset(
    images_dir='./test/images', 
    labels_dir='./test/labels', 
    output_dir='./testset'
)
