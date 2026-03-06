"""
Fine-tune YOLOv8n on merged organic data to boost ORGANIC class accuracy.

Strategy:
  - Start from existing best.pt (already knows all 4 classes)
  - Train on pure organic images → reinforces ORGANIC detection
  - Low LR to avoid catastrophic forgetting of other classes
  - RTX 4060 Laptop 8GB VRAM → batch 32, imgsz 640

Usage:
    .\.venv\Scripts\python training\train_organic_boost.py
"""

import torch
import gc
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE         = Path(__file__).parent                      # training/
PROJECT_ROOT = BASE.parent

MERGED_DATA  = BASE / "merged_organic" / "data.yaml"
BEST_PT      = BASE / "runs" / "waste_classifier" / "weights" / "best.pt"
YOLOV8N_PT   = BASE / "yolov8n.pt"                       # pretrained fallback

# Choose starting weights: fine-tune from our model; fall back to YOLOv8n
if BEST_PT.exists():
    START_WEIGHTS = str(BEST_PT)
    print(f"✅ Fine-tuning from our best model: {BEST_PT}")
else:
    START_WEIGHTS = str(YOLOV8N_PT) if YOLOV8N_PT.exists() else "yolov8n.pt"
    print(f"⚠️  best.pt not found — using pretrained: {START_WEIGHTS}")

OUTPUT_DIR   = str(BASE / "runs")
RUN_NAME     = "organic_boost_v1"


def check_gpu():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU : {name}  ({vram:.1f} GB VRAM)")
        torch.cuda.empty_cache()
        gc.collect()
        return True
    else:
        print("⚠️  No CUDA GPU found — will train on CPU (very slow).")
        return False


def main():
    print("=" * 60)
    print("ORGANIC BOOST TRAINING  —  YOLOv8n + RTX 4060")
    print("=" * 60)

    if not MERGED_DATA.exists():
        raise FileNotFoundError(
            f"Merged dataset not found: {MERGED_DATA}\n"
            "Run  merge_organic_datasets.py  first!"
        )

    has_gpu = check_gpu()
    device  = 0 if has_gpu else "cpu"

    # RTX 4060 Laptop 8 GB — safe settings
    BATCH  = 32    # ~6 GB VRAM at imgsz 640
    IMGSZ  = 640
    EPOCHS = 60

    print(f"\n📋 Config: batch={BATCH}, imgsz={IMGSZ}, epochs={EPOCHS}, device={device}")
    print(f"📂 Data  : {MERGED_DATA}")
    print(f"🏋️  Model : {START_WEIGHTS}\n")

    model = YOLO(START_WEIGHTS)

    results = model.train(
        data=str(MERGED_DATA),

        # Training duration
        epochs=EPOCHS,
        patience=15,          # early stop if no improvement for 15 epochs

        # Hardware
        device=device,
        batch=BATCH,
        imgsz=IMGSZ,
        workers=4,            # parallel data loading
        amp=True,             # mixed precision → saves VRAM, faster
        cache=False,          # set True if you have >16 GB RAM

        # Optimizer — low LR for fine-tuning (don't erase other-class knowledge)
        optimizer="AdamW",
        lr0=0.002,            # low start LR (default is 0.01)
        lrf=0.01,             # final LR ratio
        weight_decay=0.0005,
        cos_lr=True,          # cosine LR decay

        # Augmentation — helps learn from varied organic shots
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.15,
        scale=0.6,
        shear=3.0,
        flipud=0.2,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=10,      # disable mosaic for last 10 epochs

        # Output
        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    best = Path(OUTPUT_DIR) / RUN_NAME / "weights" / "best.pt"
    print(f"🏆 Best model: {best}")

    # Quick validation
    print("\nRunning validation on merged organic data...")
    val_model = YOLO(str(best))
    val = val_model.val(
        data=str(MERGED_DATA),
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=4,
    )
    print(f"\n📊 Organic-only mAP50 : {val.box.map50 * 100:.1f}%")
    if val.box.maps is not None and len(val.box.maps) > 1:
        print(f"   ORGANIC class mAP50: {val.box.maps[1] * 100:.1f}%")  # index 1

    print(f"\n📌 To use the new model, copy:")
    print(f"   {best}")
    print(f"   → backend/model/best.pt")


if __name__ == "__main__":
    main()
