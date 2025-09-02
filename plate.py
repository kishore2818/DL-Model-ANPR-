
import os
from ultralytics import YOLO
import cv2

# --- Settings ---
MODEL_PATH = r"D:\ANPR\ANPR.pt"  # path to your trained model
# IMAGE_PATH = r"D:D:\ANPR\test-1.png"  # full path to your test image
IMAGE_PATH = r"D:\ANPR\test-1.png"   # ✅ correct

OUTPUT_DIR = r"D:\ANPR\images"  # folder to save results

# --- Create output folder if not exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load model ---
model = YOLO(MODEL_PATH)

# --- Run prediction ---
results = model.predict(source=IMAGE_PATH, imgsz=640, conf=0.25)

# --- Process and save output ---
for i, r in enumerate(results):
    im_bgr = r.plot()  # image with bounding boxes
    # Save with the same name as input
    filename = os.path.basename(IMAGE_PATH)
    output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
    cv2.imwrite(output_path, im_bgr)
    print(f"✅ Saved result at: {output_path}")
