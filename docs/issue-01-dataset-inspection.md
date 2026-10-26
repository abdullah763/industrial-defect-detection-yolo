# Issue #1 — NEU-DET Dataset Inspection Findings

**Project:** Industrial Surface Defect Detection with YOLO  
**Issue:** [#1 — Inspect NEU Dataset and Define Success Metrics](https://github.com/abdullah763/industrial-defect-detection-yolo/issues/1)  
**Date:** 2026-10-27  
**Status:** Inspection and audit complete; one cross-split annotation incident must be handled during preprocessing.

## 1. Objective

Inspect the NEU-DET industrial surface-defect dataset before preparing YOLO labels. Understand the dataset splits, categories, image format, bounding-box annotations, and evaluation metrics. Preserve the downloaded raw dataset unchanged.

## 2. Dataset structure and inspection

- **Training:** `data/raw/NEU-DET/train/images/<class>/*.jpg` and `train/annotations/*.xml`
- **Validation:** `data/raw/NEU-DET/validation/images/<class>/*.jpg` and `validation/annotations/*.xml`
- Images examined with the visualization script are **200 × 200 pixels**, RGB. The project audit and sample inspection should not be treated as a proof that every file in the dataset has identical dimensions unless a full dimensional summary is separately verified.
- Annotation format: Pascal VOC-style XML, with an `<object>` entry and pixel coordinates (`xmin`, `ymin`, `xmax`, `ymax`) for each object.
- Twenty training samples spanning the six categories were selected with a fixed seed and saved to two contact sheets (10 images each) under `outputs/inspection/`.

### Counts from consolidated audit

| Split | Images | XML files | Annotated objects |
|---|---:|---:|---:|
| Train | 1,440 | 1,439 | 3,332 |
| Validation | 360 | 361 | 857 |
| **Total** | **1,800** | **1,800** | **4,189** |

### Object distribution by class

| Class | Train objects | Validation objects | Total objects |
|---|---:|---:|---:|
| `crazing` | 524 | 165 | 689 |
| `inclusion` | 852 | 159 | 1,011 |
| `patches` | 688 | 193 | 881 |
| `pitted_surface` | 345 | 87 | 432 |
| `rolled-in_scale` | 496 | 132 | 628 |
| `scratches` | 427 | 121 | 548 |
| **Total** | **3,332** | **857** | **4,189** |

The training *image folders* each contain 240 images, but the number of annotated *objects* differs across classes. Therefore, image-level balance does not imply object-level balance.

## 3. Annotation-quality findings

### A. Cross-split image/XML pairing: one incident

- Image: `data/raw/NEU-DET/train/images/crazing/crazing_240.jpg`
- Corresponding XML: `data/raw/NEU-DET/validation/annotations/crazing_240.xml`
- Audit issue categories: `cross_split_annotation: 1`, `missing_annotation: 1`, `unmatched_annotation: 1`.

These three audit entries describe **one underlying cross-split pairing incident**, not three separate annotation defects. The visualization script skipped the image when using strict same-split lookup. Previous visual inspection suggested the off-split XML matches the image, but this is not a replacement for a verified preprocessing policy.

**Next action:** During data conversion, associate that XML with its training image **in a derived dataset only**, after explicit verification. Keep the original train/validation image split and leave `data/raw/` untouched. Never silently discard the image or treat the XML as belonging to a validation image.

### B. XML filename metadata: 174 observations

The audit found 174 XML `<filename>` values with the `.jpg` extension omitted. A dedicated earlier check identified **zero different basenames**. This is a metadata-format observation, not evidence of missing images. Pair files by verified basename/stem, with ambiguity and duplicate checks.

### C. Class label versus image-folder name: 188 object observations

The audit detected 188 object-level instances where the XML `<object><name>` differs from the parent image folder. Example: `crazing_104.xml` contains a `patches` object. This is **not automatically an error**: a single image can contain multiple defect categories. Use **each object's XML label** when creating YOLO annotations; do not infer the label for every bounding box from the image folder.

### D. Other issue categories

The consolidated audit output did not report malformed coordinates, unknown categories, or out-of-bounds boxes as issues. This means no such issues were reported by the implemented checks, **not** that all possible dataset-quality problems have been ruled out.

## 4. Evaluation metrics and interpretation

- **IoU:** `intersection_area / union_area`; checks localization overlap. A correct-class, one-to-one match must also meet the evaluation IoU threshold.
- **TP / FP / FN:** Correct matches / unmatched predictions / unmatched ground-truth objects, for a specified IoU threshold, confidence selection, and class-matching rule.
- **Precision:** `TP / (TP + FP)`; how many retained positive predictions are correct.
- **Recall:** `TP / (TP + FN)`; how many ground-truth objects are found.
- **F1-score:** `2PR / (P + R)`; harmonic mean of precision and recall.
- **AP:** Area-summary of a class's precision–recall curve over confidence thresholds at a specified IoU threshold, using the evaluation implementation's interpolation convention.
- **mAP@0.5:** Average AP across classes at IoU 0.50.
- **mAP@0.5:0.95:** Average AP across classes and IoU thresholds 0.50, 0.55, ..., 0.95; more demanding of box localization.

**Planned primary metric:** mAP@0.5:0.95 on the fixed validation split. Report mAP@0.5, per-class precision/recall/AP, confusion/error examples, and inference latency as secondary measurements. These are **metric definitions and a plan**, not trained-model results.

## 5. Reproduction

From the repository root, with the Python environment activated and the dataset placed in `data/raw/NEU-DET/`:

```bash
python -m unittest discover -s tests -v
python scripts/audit_annotations.py
python scripts/visualize_dataset.py
python scripts/visualize_samples.py
```

At this checkpoint:
- Audit unit tests: **9 passed**.
- `visualize_dataset.py`: reported `crazing_1.jpg` (200 × 200), `1/1` valid boxes.
- `visualize_samples.py`: saved `outputs/inspection/training_samples_1.png` and `_2.png`, skipping the known `crazing_240.jpg` same-split mismatch.
- Full audit report: `outputs/inspection/annotation_audit.txt` (generated locally; ignored by Git).
- Both images were generated, though a final **visual review of the displayed images/contact sheets** is not independently confirmed by terminal output.
- Inspection implementation commit: [`6e22eeb`](https://github.com/abdullah763/industrial-defect-detection-yolo/commit/6e22eeb).

## 6. Limitations and handoff

1. Resolve the `crazing_240` pairing via an explicit, tested exception in derived preprocessing output, or document an alternative decision; do not alter raw files.
2. Preserve all XML object-level labels when converting Pascal VOC boxes to YOLO's normalized `class x_center y_center width height` format.
3. Check complete image/XML pairing, class IDs, box bounds, split integrity, and object counts after conversion.
4. Inspect per-class detector performance, especially classes with fewer annotated objects.
5. A trained model, baseline scores, and measured deployment latency **do not exist at this stage**.

See [`issue-01-initial-experiment-plan.md`](issue-01-initial-experiment-plan.md) for the proposed next steps and success criteria.
