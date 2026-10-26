# Issue #1 — Initial YOLO Experiment Plan

**Project:** Industrial Surface Defect Detection with YOLO  
**Issue:** [#1 — Inspect NEU Dataset and Define Success Metrics](https://github.com/abdullah763/industrial-defect-detection-yolo/issues/1)  
**Date:** 2026-10-27  
**Status:** Initial plan only; no model trained yet.

## 1. Goal

Build a reproducible six-class industrial surface-defect detector from the inspected NEU-DET dataset, then evaluate localization, detection coverage, class-level error patterns, and practical inference speed. Prefer learning, correct experiments, and traceable evaluation over optimizing a single headline score.

## 2. Ground truth and dataset policy

- Source: `data/raw/NEU-DET/`, left **read-only**.
- Original splits: **1,440 training images / 360 validation images**; do not silently reshuffle them.
- Six target classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- Use one version-controlled class-to-ID mapping for both splits. A proposed initial mapping is the alphabetical list above, IDs 0–5; verify against the training framework and save it in dataset configuration.
- Parse **each XML object's label**; do not assign all boxes based on the image folder name.
- Convert Pascal VOC pixel `(xmin, ymin, xmax, ymax)` to normalized YOLO `(class_id, x_center, y_center, width, height)` using actual image dimensions.
- Handle `crazing_240` explicitly: its JPG is in train and its corresponding XML is in validation. Verify association, then pair in the *derived training labels* without moving raw files or moving the image to validation. Fail noisily if no valid resolution is selected.
- Treat XML `<filename>` entries missing `.jpg` as an allowed format variant only when stem pairing is unique and validated.
- Validate converted file pairing, allowed class IDs, normalized coordinate bounds, positive box sizes, original split integrity, and expected object counts; report any unresolvable annotations.

## 3. Baseline experiment

1. **Prepare and validate labels:** Produce a deterministic derived YOLO dataset and data configuration. Preserve conversion logs and write focused synthetic tests for coordinate conversion and pairing.
2. **Choose a small baseline:** Start with a lightweight, currently supported Ultralytics YOLO *nano-scale* detection checkpoint; record exact package version and checkpoint ID when implemented. Use a free GPU environment if available; local MacBook CPU/MPS may be used for sanity checks.
3. **Initial image size:** Try a standard supported detector input size (for example 416 or 640) with a small batch appropriate to available memory. NEU-DET source images are 200 × 200, so document resizing and letterboxing rather than treating upscaling as new detail.
4. **Reproducibility:** Fix random seed, log train/validation split, data version, hyperparameters, input size, checkpoint, device, and library versions.
5. **Sanity test:** Run a small training/inference smoke test before a longer training run. Confirm predictions use six class IDs and boxes overlay sensibly on samples.
6. **Train and validate:** Train the baseline, evaluate using the unchanged validation split, save metrics and a modest set of visual detection examples.
7. **Analyze errors:** Compare per-class performance and inspect false positives, missed defects, and badly localized boxes. Note image-folder versus object-label ambiguities when explaining samples.
8. **Iterate only after baseline:** Consider input size, epochs, augmentation, and confidence/IoU operating points one variable at a time. Keep baseline results for comparison.

## 4. Success metrics and acceptance criteria

### Primary metric

- **mAP@0.5:0.95:** Main localization-sensitive validation metric.

### Secondary metrics

- **mAP@0.5:** More permissive localization reference.
- **Per-class AP, precision, and recall:** Identify weak defect categories, including lower-object-count classes.
- **F1-score at a specified operating confidence threshold:** Report the threshold used rather than presenting F1 in isolation.
- **Qualitative error analysis:** Show at least representative true positives, false positives, false negatives, and localization mistakes.
- **Inference latency:** Benchmark on a named device with batch size and image size recorded, distinguishing model-only time from preprocessing/postprocessing where possible.

### Definition of a useful first baseline

- **Data correctness:** No unresolved cross-split pairing is silently skipped, no invalid labels, no class-ID mismatch, and reproducible conversion counts.
- **Reproducible run:** Training/evaluation commands and exact configuration recorded; model and result artifacts clearly located.
- **Report completeness:** Overall and per-class metrics plus annotated examples and an honest limitations section.
- **Quantitative targets:** Do **not** claim an achieved accuracy target before training. Use the first measured baseline to propose evidence-based improvement targets. A score alone does not determine readiness for deployment.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `crazing_240` XML stored in wrong split | Explicit verified pairing during conversion, test that it remains a training sample |
| Mixed-class objects in folder-grouped images | Trust each XML object's name, not parent folder |
| Imbalance in number of objects per class | Report per-class AP/recall; inspect error distributions |
| Inconsistent XML `<filename>` metadata | Validate image/XML pairing by unique stem, log exceptions |
| No separate held-out test set in inspected copy | Clearly name the existing split **validation**; do not present validation results as untouched test-set performance |
| Limited compute budget | Begin with small model and short smoke run; use free GPU where available |
| Metric/threshold confusion | Record evaluation IoU thresholds and prediction confidence threshold separately |

## 6. Next implementation handoff

**First action after Issue #1:** Implement and test **VOC XML → YOLO label conversion**, beginning with a single annotated image and a one-box unit test. Demonstrate normalized coordinates on paper and in a reproducible assertion before batch conversion.

Do not run full training until the derived dataset passes conversion checks. Keep the inspection scripts and raw files intact.

See [`issue-01-dataset-inspection.md`](issue-01-dataset-inspection.md) for the measured findings that motivate this plan.
