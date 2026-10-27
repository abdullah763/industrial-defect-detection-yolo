# AI Engineering — Learning Journal

## Day 1 — October 26–27, 2026

**GitHub Issue:** [#1 — Inspect NEU Dataset and Define Success Metrics](https://github.com/abdullah763/industrial-defect-detection-yolo/issues/1)

### Today's Goal
Understand the NEU-DET industrial surface defect dataset, verify the reliability of its annotations, learn core object-detection evaluation metrics, and define an initial YOLO experiment plan.

### Work Completed
- Inspected Pascal VOC XML annotations and 200 × 200 image examples; learned how to read object class labels and bounding-box coordinates.
- Built and used `scripts/visualize_dataset.py` to display an individual image with annotated boxes; `crazing_1.jpg` reported 1/1 valid bounding boxes.
- Ran `scripts/visualize_samples.py` to select 20 reproducible training examples across six classes (seed 42) and save two contact sheets under `outputs/inspection/`.
- Ran a consolidated annotation audit: **1,440 training images, 360 validation images, 4,189 annotated objects**, across **six classes**.
- Identified the `crazing_240` cross-split pairing incident: training image, validation XML. The audit reports one cross-split incident along with related missing/unmatched entries.
- Classified **174 omitted XML filename extensions** as observations rather than missing annotations, and identified **188 object labels differing from image-folder names** for review.
- Added and ran **nine passing synthetic unit tests** for the annotation audit; verified the real dataset audit output.
- Removed redundant exploratory scripts, cleaned the repository, and pushed the inspection code (commit `6e22eeb`).
- Created and pushed `docs/issue-01-dataset-inspection.md` and `docs/issue-01-initial-experiment-plan.md`.
- Practiced IoU, TP/FP/FN, precision, recall, F1-score, Average Precision (AP), and mAP concepts using manual examples.

### What I Learned
- **Pascal VOC annotations:** Each `<object>` has its own class and `(xmin, ymin, xmax, ymax)` box. The parent image folder does not necessarily describe every object in that image.
- **Image balance vs. object balance:** Each training class folder has 240 images, but annotated object counts vary; for example, `inclusion` has 852 training objects and `pitted_surface` has 345.
- **Intersection over Union (IoU):** Intersection area divided by union area. Calculate intersection coordinates with `max()` for lower bounds, `min()` for upper bounds, and clamp width/height to zero if boxes do not overlap. I manually computed IoUs of approximately **0.286** and **0.25** for two practice cases.
- **Precision and recall:** Precision is `TP / (TP + FP)` (reliability of positive predictions); recall is `TP / (TP + FN)` (fraction of actual defects found). I corrected an early recall mistake and solved a later example with both values equal to **0.80**.
- **F1-score:** The harmonic mean `2PR / (P + R)` penalizes an imbalance between precision and recall.
- **AP and mAP:** AP summarizes a class's precision–recall curve across confidence thresholds at a given matching IoU. `mAP@0.5` averages AP at IoU 0.50; `mAP@0.5:0.95` averages across IoU thresholds 0.50 through 0.95 and classes.
- **Confidence threshold vs. IoU threshold:** Confidence filtering determines which predictions are kept; matching IoU determines whether a retained prediction is sufficiently localized.
- **Git workflow:** Run tests, check staged diffs, commit meaningful changes, push, and confirm `git status` is clean.

### Problems Encountered
- `crazing_240.jpg` had no XML in its own training annotation folder; its matching XML was in the validation annotation folder.
- Initial filename checks incorrectly treated XML `<filename>` values lacking `.jpg` as mismatches.
- Some object classes differed from their image folder, which could lead to mislabeled YOLO training data if folder names were used as labels.
- An early inspection implementation included duplicate/redundant scripts; the first attempt to extract a cleanup ZIP from `~/Downloads/` failed because it was not available there.
- During the evaluation exercises, I initially mixed up false positives and false negatives and made a recall arithmetic mistake.

### How I Solved Them
- Compared annotation locations across both splits; documented the cross-split incident rather than changing raw data.
- Matched filenames carefully by verified stems and treated omitted extensions as informational observations.
- Chose each XML object's own `<name>` value as the authoritative class label for future conversion.
- Consolidated the functionality into three scripts plus one test module, extracted the ZIP from the project directory, and ran all nine tests before committing.
- Reworked the precision/recall examples by separately counting actual defects, successful detections, missed defects, and false alarms.

### Engineering Decisions and Why
- **Keep `data/raw/` unchanged:** The original download should remain reproducible. Handle the misplaced XML with an explicit, tested policy in derived preprocessing output.
- **Use object-level labels:** Images can contain multiple defect classes, so every object's `<name>` field matters.
- **Separate audit issues from observations:** An omitted filename extension is not automatically a missing or corrupt annotation.
- **Use a reproducible random seed for contact sheets:** Makes inspection repeatable.
- **Keep generated images, raw data, and audit outputs out of Git:** Version source code and concise findings, not local/derived files.
- **Use mAP@0.5:0.95 as the planned primary model metric:** It evaluates localization at several IoU thresholds; supplement with per-class AP, precision, recall, error analysis, and measured latency after training.

### Interview Questions
1. **Why can a class-balanced folder structure still give an object-imbalanced detector dataset?**  A detection image may contain multiple boxes, potentially from different classes. Image-folder counts do not equal per-class annotated-object counts.
2. **What is the difference between precision and recall in defect inspection?**  Precision measures how many positive detections are correct; recall measures how many real defects are found. Missing a serious defect is a false negative.
3. **Why do mAP@0.5 and mAP@0.5:0.95 differ?**  The first evaluates AP at IoU 0.50; the second also tests stricter matching thresholds up to 0.95, so it better reflects tight localization.

### Tomorrow's First Action
- Start Issue #2 only after Issue #1 is closed: design the Pascal VOC XML → YOLO label converter and its synthetic tests. Begin by specifying normalized bounding-box formulas, split-pairing rules, and explicit handling of `crazing_240` without editing the raw dataset.

### Time Spent
- Not tracked precisely; no estimate recorded.

### Final Verification / Handoff
- Audit tests: **9/9 passed**.
- Both sample contact sheets were generated; the terminal verified file creation, but a final manual visual inspection should be confirmed separately.
- Inspection code and Issue #1 documentation have been pushed to `main`; the latest user-provided `git status` was clean and synchronized with `origin/main`.
- No trained YOLO model or measured validation mAP exists yet; those belong to later issues.

---

## Day 2 — October 27, 2026 — Issue #2

**GitHub Issue:** [#2 — Prepare dataset and validate train/val/test splits](https://github.com/abdullah763/industrial-defect-detection-yolo/issues/2)

### Goal
Prepare a reproducible six-class YOLO dataset without changing the original NEU-DET source files, validate it, and prevent known data leakage.

### Work Completed
- Practiced VOC pixel-to-YOLO normalized box conversion on paper and in `scratch/voc_to_yolo_practice.py`, including positive-area and image-boundary validation.
- Parsed actual XML objects, converted all per-object class names to IDs 0–5, and verified mixed-class `crazing_104.xml` output.
- Added production VOC conversion, image/XML identity matching, SHA-256 duplicate checks, controlled duplicate resolution, fixed-seed dataset splitting, cross-split dHash screening, synthetic tests, and final export/integrity verification.
- Detected a misplaced `crazing_240.xml` and retained training membership based on the image; raw data stayed unchanged.
- Detected identical image bytes for `patches_101` and `patches_105`, observed slightly different boxes, and excluded `patches_105` from derived data only.
- Generated **1,799 YOLO images**, with **1,151 train / 288 val / 360 held-out test**, and reconciled **4,186 retained + 3 excluded + 0 rejected = 4,189 original objects**.
- Reviewed cross-split near-duplicate screening (1,156 dHash candidates, zero strict-heuristic priority flags) and manually reviewed generated six-class YOLO overlays for each split (18 displayed examples total).
- Ran the complete automated test suite: **27/27 passed** after a portability fix to the macOS path assertion.

### Problems and Debugging
- Forgot to save a Python file before running tests, causing an import error; saving and rerunning resolved it.
- An initially empty test file reported `Ran 0 tests — OK`; adding the tests made test discovery meaningful.
- A misplaced indentation block repeated duplicate resolution inside a loop; moved it outside the loop and returned selected images explicitly.
- A synthetic export test compared `/var` against `/private/var` on macOS; comparing resolved paths fixed the assertion.
- Raw NEU-DET contained one exact duplicate pair with slightly inconsistent annotations and many broad perceptual-hash matches. Avoided blanket deletion or interpreting every candidate as a duplicate.

### Engineering Decisions
- Preserve `data/raw` and keep derived images, manifest, and similarity CSV out of Git.
- Use the XML object's true label rather than the image folder to label every box.
- Preserve original validation images as untouched **test**, splitting only the original training images into training and validation with seed 42.
- Make duplicate and cross-split decisions explicit and logged; fail rather than silently overwrite ambiguous identities or discard invalid annotations.
- Defer deeper exercises on `unittest`, SHA-256, perceptual hashing, Python collection operations, and error handling until after Issue #2 is shipped.

### What I Can Explain Now
- VOC `(xmin, ymin, xmax, ymax)` to YOLO `class_id x_center y_center width height` normalization.
- Why duplicate filenames and duplicate file content require different checks.
- Why training/validation leakage and repeatedly tuning on a held-out test set undermine evaluation.
- Why automated tests must execute actual test cases, not just finish with `OK`.

### Follow-Up Learning
Review `unittest` structure and assertions, temporary directories, hashing and Hamming distance, file/path handling, reproducible random splits, and interpretation of class-level counts using small independent examples.

### Issue #3 First Action
Confirm the training-environment path in `data.yaml`; perform a light YOLO dataset/model-loading smoke test without evaluating or tuning on the held-out test set.

### Time Spent
Not tracked precisely; no estimate recorded.
