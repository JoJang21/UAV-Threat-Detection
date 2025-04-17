from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Nano model (fastest, less accurate)
# model = YOLO('yolov8s.pt')  # Small model
# model = YOLO('yolov8m.pt')  # Medium model
# model = YOLO('yolov8l.pt')  # Large model

# Train the model
results = model.train(
    data='./dataset/data.yaml',
    epochs=70,
    imgsz=640,
    batch=16,
    patience=50,  # Early stopping
    name='gun_detection'
)

metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")

# Make predictions on test images
image_path = './testset/train/images'  
results = model.predict(image_path, save=True, conf=0.5)
