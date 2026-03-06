"""
Focused Retraining for ORGANIC, HAZARDOUS, GENERAL
===================================================
Resumes from best.pt and uses stronger augmentation + more epochs
to boost weak classes (ORGANIC: 19.5%, GENERAL: 18.0%)

Strategy:
- Resume from best.pt (keeps RECYCLABLE 91.6% knowledge)
- More epochs (100 total)
- Stronger augmentation (helps minority classes)
- Higher mosaic/mixup (forces model to learn smaller objects)
- Lower learning rate (fine-tune, don't overwrite)
"""

import torch
import gc
from ultralytics import YOLO

def train():
    print("=" * 60)
    print("FOCUSED RETRAINING: ORGANIC + HAZARDOUS + GENERAL")
    print("=" * 60)
    
    # GPU check
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu} ({vram:.1f} GB)")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load from YOUR trained model (not pretrained)
    model = YOLO('runs/waste_classifier/weights/best.pt')
    print(f"Loaded model with classes: {model.names}")
    
    # Train with aggressive augmentation for weak classes
    results = model.train(
        data='D:/Hackthon-garbage/training/dataset/data.yaml',
        
        # More epochs - resume learning
        epochs=80,
        
        # Memory safe for RTX 3050 4GB
        batch=4,           # Smaller batch = more gradient updates
        imgsz=512,         # Slightly larger images for better features
        
        # Device
        device=0,
        workers=0,
        cache=False,
        amp=True,
        
        # Lower LR for fine-tuning (don't destroy RECYCLABLE knowledge)
        lr0=0.005,         # Half the default
        lrf=0.01,          # Final LR ratio
        optimizer='AdamW',
        weight_decay=0.001,
        
        # AGGRESSIVE AUGMENTATION (helps minority classes)
        mosaic=1.0,        # Full mosaic augmentation
        mixup=0.3,         # Mix images together (helps learn varied features)
        copy_paste=0.2,    # Copy-paste augmentation
        degrees=15.0,      # Rotation
        translate=0.2,     # Translation
        scale=0.7,         # Scale variation (important!)
        shear=5.0,         # Shear
        flipud=0.3,        # Vertical flip
        fliplr=0.5,        # Horizontal flip
        hsv_h=0.02,        # Hue variation
        hsv_s=0.8,         # Saturation variation
        hsv_v=0.5,         # Value variation
        
        # Keep mosaic ON longer (don't disable at end)
        close_mosaic=5,    # Only disable mosaic last 5 epochs
        
        # Training config
        patience=20,       # More patience
        cos_lr=True,       # Cosine LR schedule
        
        # Save config
        project='D:/Hackthon-garbage/training/runs',
        name='waste_v2_focused',
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model: runs/waste_v2_focused/weights/best.pt")
    
    # Validate
    print("\nValidating new model...")
    model_new = YOLO('D:/Hackthon-garbage/training/runs/waste_v2_focused/weights/best.pt')
    val = model_new.val(data='D:/Hackthon-garbage/training/dataset/data.yaml', imgsz=512, batch=4, workers=0)
    
    print(f"\nOverall mAP50: {val.box.map50*100:.1f}%")
    print(f"Per-class mAP50:")
    for i, name in model_new.names.items():
        print(f"  {name}: {val.box.ap50[i]*100:.1f}%")

if __name__ == '__main__':
    train()
