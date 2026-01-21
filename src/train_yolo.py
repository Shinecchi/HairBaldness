import os
from ultralytics import YOLO


def train_yolo():
    # --- 1. SETUP PATHS ---
    base_dir = os.path.join(os.getcwd(), "dataset_v1_new")

    # Rename 'valid' to 'val' if needed
    valid_path = os.path.join(base_dir, "valid")
    val_path = os.path.join(base_dir, "val")

    if os.path.exists(valid_path) and not os.path.exists(val_path):
        print("Renaming 'valid' folder to 'val' for YOLO...")
        os.rename(valid_path, val_path)

    # --- 2. LOAD STRONGER MODEL ---
    # We are switching from 'n' (Nano) to 's' (Small).
    # This is slightly slower but MUCH smarter at distinguishing similar classes.
    print("--- Loading YOLOv8 Small Model ---")
    model = YOLO('yolov8s-cls.pt')

    # --- 3. TRAIN WITH HIGH RESOLUTION ---
    print(f"--- Starting Improved Training on {base_dir} ---")

    results = model.train(
        data=base_dir,
        epochs=50,  # Increased from 30 to 50
        imgsz=320,  # Increased from 224 to 320 (Helps see hair density)
        batch=16,  # Keep 16 (Safe for CPU)
        patience=10,  # Stop early if it stops learning for 10 epochs
        augment=True,  # Force extra data augmentation
        name='hair_loss_yolo_improved',
        device='cpu',
        exist_ok=True
    )

    # --- 4. VALIDATE ---
    print("--- Validating Model ---")
    metrics = model.val(device=0)  # Force GPU usage here too
    print(f"Top-1 Accuracy: {metrics.top1:.2f}")


if __name__ == '__main__':
    train_yolo()