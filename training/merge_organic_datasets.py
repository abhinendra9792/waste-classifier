"""
Merge both downloaded organic datasets into one unified dataset.
Remaps ALL classes → ORGANIC (class index 1, matching our system).

Our 4 classes:
  0: RECYCLABLE
  1: ORGANIC   ← all organic dataset labels will map here
  2: HAZARDOUS
  3: GENERAL
"""

import os
import shutil
from pathlib import Path


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE        = Path(__file__).parent                            # training/
DATASET1    = Path(r"C:\Users\saga\Downloads\organic waste detection.v1i.yolov8")
DATASET2    = Path(r"C:\Users\saga\Downloads\Organic Waste.v1i.yolov8")
OUT_DIR     = BASE / "merged_organic"                          # output folder
ORGANIC_IDX = 1                                                # class index in our system

SPLITS = ["train", "valid", "test"]


def remap_label_file(src: Path, dst: Path, remap_all_to: int):
    """Read a YOLO .txt label, remap all class IDs → remap_all_to, write to dst."""
    if not src.exists():
        return
    lines = src.read_text().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[0] = str(remap_all_to)          # overwrite class ID
            new_lines.append(" ".join(parts))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(new_lines))


def copy_dataset(src_root: Path, out_root: Path, prefix: str, remap_to: int):
    """Copy images + remapped labels from one dataset into the merged output."""
    copied_imgs = 0
    copied_lbls = 0
    for split in SPLITS:
        img_src = src_root / split / "images"
        lbl_src = src_root / split / "labels"
        img_dst = out_root / split / "images"
        lbl_dst = out_root / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            continue

        for img_file in img_src.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            # Unique name to avoid collisions between datasets
            new_name = f"{prefix}_{img_file.name}"
            shutil.copy2(img_file, img_dst / new_name)
            copied_imgs += 1

            # Corresponding label
            lbl_file = lbl_src / (img_file.stem + ".txt")
            new_lbl_name = f"{prefix}_{img_file.stem}.txt"
            remap_label_file(lbl_file, lbl_dst / new_lbl_name, remap_to)
            if lbl_file.exists():
                copied_lbls += 1

    print(f"  [{prefix}] copied {copied_imgs} images, {copied_lbls} labels")
    return copied_imgs


def write_data_yaml(out_root: Path):
    yaml_content = f"""# Merged Organic Datasets - ready for fine-tuning with our 4-class system
train: {(out_root / 'train' / 'images').as_posix()}
val:   {(out_root / 'valid' / 'images').as_posix()}
test:  {(out_root / 'test'  / 'images').as_posix()}

nc: 4
names: ['RECYCLABLE', 'ORGANIC', 'HAZARDOUS', 'GENERAL']

# NOTE: All images in this dataset are ORGANIC (class 1).
# Use for fine-tuning from best.pt to improve organic detection.
"""
    (out_root / "data.yaml").write_text(yaml_content)
    print(f"  Written data.yaml → {out_root / 'data.yaml'}")


def count_images(out_root: Path):
    for split in SPLITS:
        p = out_root / split / "images"
        if p.exists():
            n = len(list(p.iterdir()))
            print(f"  {split}: {n} images")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("MERGING ORGANIC DATASETS")
    print("=" * 55)

    if OUT_DIR.exists():
        print(f"Output dir already exists: {OUT_DIR}")
        ans = input("Delete and re-merge? [y/N]: ").strip().lower()
        if ans == "y":
            shutil.rmtree(OUT_DIR)
        else:
            print("Skipped. Using existing merged dataset.")
            count_images(OUT_DIR)
            raise SystemExit(0)

    print(f"\nDataset 1 ({DATASET1.name}):")
    n1 = copy_dataset(DATASET1, OUT_DIR, "ds1", ORGANIC_IDX)

    print(f"\nDataset 2 ({DATASET2.name}):")
    n2 = copy_dataset(DATASET2, OUT_DIR, "ds2", ORGANIC_IDX)

    print(f"\nWriting data.yaml...")
    write_data_yaml(OUT_DIR)

    print(f"\n{'='*55}")
    print(f"DONE — Total: {n1 + n2} images")
    print("Final split counts:")
    count_images(OUT_DIR)
    print(f"\nMerged dataset path: {OUT_DIR}")
    print("Next step: run  train_organic_boost.py")
