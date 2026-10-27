from pathlib import Path
from ultralytics import YOLO
import torch
import ultralytics


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\Ева\Desktop\Хакатон\Yolo_26\runs_yolo26\trial_0005\weights\best.pt"
)

IMG_SIZE = 768

# Dynamic input sizes
DYNAMIC = True

# NMS outside ONNX
NMS = False

# ONNX opset
OPSET = 17


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print(f"Ultralytics : {ultralytics.__version__}")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"CUDA        : {torch.version.cuda}")

print()


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print("=" * 70)
print("MODEL")
print("=" * 70)
print(f"Input : {MODEL_PATH}")
print()


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print(f"Task: {model.task}")
print()


# ============================================================
# EXPORT
# ============================================================

print("=" * 70)
print("EXPORTING TO ONNX")
print("=" * 70)

onnx_path = model.export(
    format="onnx",
    imgsz=IMG_SIZE,
    dynamic=DYNAMIC,
    nms=NMS,
    opset=OPSET,
    simplify=True,
)

print()
print("=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)
print(f"ONNX model: {onnx_path}")

if Path(onnx_path).exists():
    size_mb = Path(onnx_path).stat().st_size / (1024 ** 2)
    print(f"Size      : {size_mb:.2f} MB")
else:
    print("WARNING: ONNX file was not found.")

print()
print("Next step:")
print("Run:")
print("python verify_onnx.py")