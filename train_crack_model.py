"""
Generate realistic synthetic training data for crack vs no-crack classification.
Creates images that mimic:
- no_crack: Smooth fields, uniform ground, sky+land compositions
- crack: Dense crack networks on dried/parched ground

Then retrains the YOLOv8-cls model.
"""
import os
import sys
import shutil
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "surface_crack")

EPOCHS = 25
IMG_SIZE = 224
BATCH_SIZE = 16

os.makedirs(MODELS_DIR, exist_ok=True)


def generate_crack_image(size=224):
    """Generate a realistic cracked earth image."""
    # Base: dry earth color (tan/brown tones)
    base_r = np.random.randint(160, 210)
    base_g = np.random.randint(140, 180)
    base_b = np.random.randint(110, 150)
    img = np.full((size, size, 3), [base_b, base_g, base_r], dtype=np.uint8)
    
    # Add surface texture variation
    noise = np.random.normal(0, 12, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Generate crack network using random walks
    num_main_cracks = np.random.randint(8, 25)
    crack_mask = np.zeros((size, size), dtype=np.uint8)
    
    for _ in range(num_main_cracks):
        # Start from random point
        x, y = np.random.randint(0, size, 2)
        length = np.random.randint(40, size)
        thickness = np.random.randint(1, 4)
        
        points = [(x, y)]
        angle = np.random.uniform(0, 2 * np.pi)
        
        for step in range(length):
            # Random walk with momentum
            angle += np.random.normal(0, 0.3)
            dx = int(np.cos(angle) * np.random.randint(1, 4))
            dy = int(np.sin(angle) * np.random.randint(1, 4))
            x = max(0, min(size-1, x + dx))
            y = max(0, min(size-1, y + dy))
            points.append((x, y))
            
            # Branch occasionally
            if np.random.random() < 0.08:
                bx, by = x, y
                b_angle = angle + np.random.uniform(-1.2, 1.2)
                b_len = np.random.randint(10, 40)
                b_points = [(bx, by)]
                for _ in range(b_len):
                    b_angle += np.random.normal(0, 0.2)
                    bx = max(0, min(size-1, bx + int(np.cos(b_angle) * 2)))
                    by = max(0, min(size-1, by + int(np.sin(b_angle) * 2)))
                    b_points.append((bx, by))
                if len(b_points) > 1:
                    pts = np.array(b_points, dtype=np.int32)
                    cv2.polylines(crack_mask, [pts], False, 255, max(1, thickness-1))
        
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(crack_mask, [pts], False, 255, thickness)
    
    # Add polygon-style cracks (mud crack pattern)
    num_polygons = np.random.randint(3, 10)
    for _ in range(num_polygons):
        cx, cy = np.random.randint(20, size-20, 2)
        radius = np.random.randint(15, 50)
        num_sides = np.random.randint(4, 8)
        angles = np.sort(np.random.uniform(0, 2*np.pi, num_sides))
        pts = []
        for a in angles:
            r = radius + np.random.randint(-10, 10)
            px = int(cx + r * np.cos(a))
            py = int(cy + r * np.sin(a))
            pts.append([max(0, min(size-1, px)), max(0, min(size-1, py))])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(crack_mask, [pts], True, 255, np.random.randint(1, 3))
    
    # Apply cracks to image (darken where mask is active)
    darkness = np.random.randint(40, 90)
    for c in range(3):
        channel = img[:,:,c].astype(np.int16)
        channel[crack_mask > 0] -= darkness
        img[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
    
    # Slight blur to make it look natural
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img


def generate_safe_image(size=224):
    """Generate a realistic safe (no crack) ground image."""
    img_type = np.random.choice(["field", "smooth_ground", "green_field", "sky_ground"])
    
    if img_type == "field":
        # Brown/tan agricultural field - uniform
        base_r = np.random.randint(160, 220)
        base_g = np.random.randint(140, 190)
        base_b = np.random.randint(100, 150)
        img = np.full((size, size, 3), [base_b, base_g, base_r], dtype=np.uint8)
        # Gentle gradient from bottom to top
        for y in range(size):
            factor = 1.0 + 0.15 * (y / size)
            img[y] = np.clip(img[y] * factor, 0, 255).astype(np.uint8)
        noise = np.random.normal(0, 5, (size, size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
    elif img_type == "smooth_ground":
        # Smooth concrete/asphalt - gray tones
        base = np.random.randint(140, 200)
        img = np.full((size, size, 3), base, dtype=np.uint8)
        noise = np.random.normal(0, 8, (size, size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        
    elif img_type == "green_field":
        # Green vegetation field
        base_r = np.random.randint(60, 100)
        base_g = np.random.randint(120, 180)
        base_b = np.random.randint(40, 80)
        img = np.full((size, size, 3), [base_b, base_g, base_r], dtype=np.uint8)
        noise = np.random.normal(0, 15, (size, size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
    elif img_type == "sky_ground":
        # Sky on top, smooth ground on bottom (like the user's safe image)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        horizon = np.random.randint(size//3, size//2)
        
        # Sky (blue gradient)
        for y in range(horizon):
            blue = int(200 + 55 * (y / horizon))
            green = int(180 + 30 * (y / horizon))
            red = int(160 + 20 * (y / horizon))
            img[y] = [min(255, blue), min(255, green), min(255, red)]
        
        # Ground (tan/brown, smooth)
        base_r = np.random.randint(170, 210)
        base_g = np.random.randint(150, 185)
        base_b = np.random.randint(110, 145)
        for y in range(horizon, size):
            factor = 1.0 - 0.1 * ((y - horizon) / (size - horizon))
            img[y] = [int(base_b * factor), int(base_g * factor), int(base_r * factor)]
        
        noise = np.random.normal(0, 4, (size, size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img


def create_dataset():
    """Create the full training dataset."""
    splits = {"train": 500, "val": 100}
    
    for split, count in splits.items():
        for cls, generator in [("crack", generate_crack_image), ("no_crack", generate_safe_image)]:
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            
            # Clear old data
            for f in os.listdir(cls_dir):
                os.remove(os.path.join(cls_dir, f))
            
            print(f"  Generating {count} images for {split}/{cls}...")
            for i in range(count):
                img = generator(IMG_SIZE)
                cv2.imwrite(os.path.join(cls_dir, f"{cls}_{i:04d}.jpg"), img)
            
            print(f"    ✓ {count} images saved")


# ─── Main ───
print("=" * 60)
print("STEP 1: Generating Realistic Training Dataset")
print("=" * 60)
create_dataset()

print("\n" + "=" * 60)
print("STEP 2: Training YOLOv8-cls Crack Classifier")
print(f"  Epochs: {EPOCHS} | ImgSize: {IMG_SIZE}")
print("=" * 60)

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=DATASET_DIR,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=MODELS_DIR,
    name="crack_classifier",
    exist_ok=True,
    verbose=True,
    patience=10,
    lr0=0.001,
    augment=True,
)

# Copy best weights
best_weights = os.path.join(MODELS_DIR, "crack_classifier", "weights", "best.pt")
final_path = os.path.join(MODELS_DIR, "crack_classifier_best.pt")

if os.path.exists(best_weights):
    shutil.copy2(best_weights, final_path)
    print(f"\n{'=' * 60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"Best weights: {final_path}")
    print(f"{'=' * 60}")
else:
    last_weights = os.path.join(MODELS_DIR, "crack_classifier", "weights", "last.pt")
    if os.path.exists(last_weights):
        shutil.copy2(last_weights, final_path)
        print(f"Used last.pt fallback: {final_path}")
