# Industrial Surface Defect Detection with YOLO

A project-based computer vision engineering repository for inspecting the NEU-DET dataset and developing a YOLO defect detector. The first stage focuses on understanding and validating ground-truth annotations before training.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the NEU-DET dataset in `data/raw/NEU-DET/`, with `train/` and `validation/` directories, each containing `images/` and `annotations/`. Dataset files are intentionally excluded from version control.

## Dataset inspection utilities

```bash
# Inspect one annotated image (defaults to train/crazing_1.jpg)
python scripts/visualize_dataset.py

# View a different image, using its same-split XML
python scripts/visualize_dataset.py --image data/raw/NEU-DET/train/images/crazing/crazing_2.jpg

# Explicitly supply an XML file (useful when investigating cross-split anomalies)
python scripts/visualize_dataset.py \
  --image data/raw/NEU-DET/train/images/crazing/crazing_240.jpg \
  --xml data/raw/NEU-DET/validation/annotations/crazing_240.xml

# Generate two reproducible contact sheets of 10 images each
python scripts/visualize_samples.py

# Audit both splits, their image/XML pairing, metadata, boxes, and class labels
python scripts/audit_annotations.py

# Run tests (Python standard-library unittest)
python -m unittest discover -s tests -v
```

Audit reports and contact sheets are saved in `outputs/inspection/`, which is excluded from Git. These tools are read-only with respect to `data/raw/`.

## Preliminary local findings (October 2026)

The inspected local copy contains 1,440 training images and 360 validation images. A cross-split pairing anomaly was identified: `crazing_240.jpg` is in training, while `crazing_240.xml` is in validation. The file's annotations visually appear compatible, but raw files must remain unchanged until the data preparation plan is agreed.

Object labels may differ from image folder names; for example, `crazing_104.xml` includes a `patches` object. Always use individual XML object labels for object-detection training. A separate filename audit identified 174 XML `<filename>` values lacking the `.jpg` extension; these are metadata-format differences, not different basenames. Re-run the consolidated audit to verify those counts with the current code.

## Project scope

The dataset inspection scripts support the first project issue. Model configuration, experiments, training, and evaluation will be added in later milestones.
