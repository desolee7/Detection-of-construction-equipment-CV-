from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\Ева\Desktop\Хакатон\Yolo_26\runs_yolo26\trial_0005\weights\best.onnx"
)

IMAGE_DIR = Path(
    r"C:\Users\Ева\Desktop\Хакатон\dataset\images\val"
)

OUTPUT_DIR = Path(
    r"C:\Users\Ева\Desktop\Хакатон\Yolo_26\onnx_test"
)

IMG_SIZE = 768
CONF = 0.25


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ONNX model...")
model = YOLO(str(MODEL_PATH))

print(f"Model: {MODEL_PATH}")
print(f"Images: {IMAGE_DIR}")
print(f"Output: {OUTPUT_DIR}")


# ============================================================
# GET IMAGES
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

# Берём первые 5 изображений
images = images[:5]

if not images:
    raise FileNotFoundError(
        f"Не найдено изображений в {IMAGE_DIR}"
    )

print(f"\nFound {len(images)} images:")

for image in images:
    print(f"  {image}")


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning inference...\n")

results = model.predict(
    source=[str(x) for x in images],
    imgsz=IMG_SIZE,
    conf=CONF,
    save=True,
    project=str(OUTPUT_DIR),
    name="results",
    exist_ok=True,
    verbose=True,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for image, result in zip(images, results):

    if result.boxes is None:
        count = 0
    else:
        count = len(result.boxes)

    print(
        f"{image.name}: {count} detections"
    )

print("\nDone!")
print(f"Results saved to:")
print(OUTPUT_DIR / "results")