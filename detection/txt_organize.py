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
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg'))]
    
    random.shuffle(images)
    split = int(len(images) * train_ratio)
    train_files = images[:split]
    val_files = images[split:]
    
    # Copy training files/annotations
    for img_file in train_files:
        base_name = os.path.splitext(img_file)[0]
        txt_file = f"{base_name}.txt"
        
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(output_dir, 'train', 'images', img_file)
        )
        
        label_path = os.path.join(labels_dir, txt_file)
        shutil.copy(
                label_path,
                os.path.join(output_dir, 'train', 'labels', txt_file)
        )
    
    # Copy validation 
    for img_file in val_files:
        base_name = os.path.splitext(img_file)[0]
        txt_file = f"{base_name}.txt"
        
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(output_dir, 'val', 'images', img_file)
        )
        
        label_path = os.path.join(labels_dir, txt_file)
        shutil.copy(
            label_path,
            os.path.join(output_dir, 'val', 'labels', txt_file)
        )
    
    print(f"{len(train_files)} training and {len(val_files)} validation data")

organize_dataset(
    images_dir='./train/images', 
    labels_dir='./train/labels', 
    output_dir='./dataset2'
)
