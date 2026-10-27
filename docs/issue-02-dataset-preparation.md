# Issue #2 — NEU-DET Dataset Preparation and Validation

**Repository:** `abdullah763/industrial-defect-detection-yolo`

**GitHub issue:** [#2 — Prepare dataset and validate train/val/test splits](https://github.com/abdullah763/industrial-defect-detection-yolo/issues/2)

**Completed local review:** October 27, 2026

**Status:** Local preparation and validation completed; GitHub commit/Technical Lead acceptance pending.

## Objective and source data

Convert NEU-DET Pascal VOC XML annotations into six-class YOLO detection labels and establish reproducible, leakage-conscious train/validation/test partitions. Keep `data/raw/NEU-DET` unchanged. All derived images, labels, reports, and manifests are local artifacts under `data/processed/NEU-DET-YOLO` and `outputs/inspection/issue-02`.

Original source data:

- 1,800 images: 1,440 original train + 360 original validation.
- 4,189 annotated XML objects: 3,332 objects in the original train annotation directory and 857 in the original validation annotation directory. These source-directory counts are not final split counts.
- Classes and zero-based YOLO IDs: `0 crazing`, `1 inclusion`, `2 patches`, `3 pitted_surface`, `4 rolled-in_scale`, `5 scratches`.
- Each object's XML `<name>` determines its YOLO class; do not infer all object classes from the image's folder name.

## Split policy and reproducibility

- Use seed **42**, reproducible sorting/shuffling, and folder-category stratification within the original training images.
- Retain the original **360 validation images only as an untouched final test set**.
- From the retained original training images, assign 1,151 to train and 288 to validation (approximately 80/20 after one known duplicate-image exclusion).
- Use the derived validation set for model selection, early stopping, and controlled experiments. Do **not** use the held-out test set to choose hyperparameters or iterations.
- Folder-level stratification is not the same as multi-label/object-level stratification, so validate per-object class coverage in every split.
- Source identity is based on a unique filename stem, with explicit checks against duplicate/missing/unmatched image/XML identities. Hash checks use image content as well as names.

## Documented source-data decisions

1. **Misplaced XML (`crazing_240`)**: `crazing_240.jpg` is in the original training images, but its XML is in the original validation annotation directory. Match it by unique stem and use its image's source membership when assigning its derived partition. The built manifest records this exception, assigning it to derived train; no raw files are moved.
2. **Exact duplicate image pair**: `patches_101.jpg` and `patches_105.jpg` have identical SHA-256 file hashes but slightly differing three-box annotations. Deterministically keep `patches_101` and exclude `patches_105` **only from derived data**. This excludes its three annotated objects; it does not claim the retained annotation is objectively superior.
3. **Class/folder differences**: The prior inspection counted 188 individual objects whose XML labels differ from the image-folder category. Keep XML object-level labels; these are not automatic annotation errors.
4. **XML filename metadata**: The prior inspection counted 174 missing `.jpg` extensions inside XML filename text. Unique-stem pairing handles these informational variations without silently selecting ambiguous files.

## Final derived dataset results

| Split | Images | YOLO objects |
|---|---:|---:|
| Train | 1,151 | 2,660 |
| Validation | 288 | 672 |
| Test (original validation images) | 360 | 854 |
| **Total** | **1,799** | **4,186** |

| Class | Train objects | Validation objects | Test objects |
|---|---:|---:|---:|
| Crazing | 418 | 109 | 162 |
| Inclusion | 672 | 180 | 159 |
| Patches | 560 | 125 | 193 |
| Pitted surface | 282 | 63 | 87 |
| Rolled-in scale | 390 | 106 | 132 |
| Scratches | 338 | 89 | 121 |

**Object-count reconciliation:** 4,189 original = 4,186 retained + 3 deliberately excluded + **0 rejected**. All six XML object classes occur in train, validation, and test.

## Label conversion and integrity checks

- Convert each VOC box `(xmin, ymin, xmax, ymax)` to normalized YOLO line `class_id x_center y_center width height` using actual image dimensions.
- Validate positive image sizes, coordinate ordering, image boundaries, allowed class IDs, normalized values, image/XML size agreement, and preservation of every selected XML object.
- Validate each written label line; require exactly one derived label file per derived image.
- Save deterministic membership, reconciliation, and exceptions in `split_manifest.json`, and dataset configuration in `data.yaml`.
- Generated files: `data/processed/NEU-DET-YOLO/{images,labels}/{train,val,test}`, `data.yaml`, `split_manifest.json`, `integrity_report.md`, and three visual previews.

## Leakage screening and limitations

- SHA-256 found one exact duplicate group in source data; the known pair was explicitly resolved prior to splitting.
- Cross-split dHash screening (distance <= 4) yielded **1,156 candidate pairs**. A stricter pixel-MAE/correlation heuristic flagged **zero high-priority pairs** (priority required MAE <= 4.0 and correlation >= 0.995).
- The full candidate report and review collage are in ignored local outputs, not proof that all near-duplicate leakage is absent. Similar steel-surface textures can trigger perceptual-hash matches, and near-duplicates can evade the chosen thresholds. Consider targeted review of residual high-correlation pairs if evaluation looks suspicious.
- The original training image category was used for stratification, so object-level balance is not perfectly guaranteed even though all six classes appear in each split.
- `data.yaml` currently contains an **absolute path specific to the local Mac**. Regenerate/update `path:` to the Kaggle or other training environment's dataset root before Issue #3 training. Preserve the same split manifest and held-out test policy.
- This report covers data preparation, not a trained model. There are **no accuracy, recall, AP, or mAP results yet**.

## Verification evidence

- **27/27 Python unittest tests passed** after fixing a macOS `/var` versus `/private/var` canonical-path assertion in the synthetic export test.
- Real `--preflight` completed; candidate CSV and side-by-side preview inspected.
- Real `--build` completed and generated the target image and label files; the resulting `integrity_report.md` reconciles object counts.
- **Manual visual sample review completed:** six generated YOLO-overlay images per split, **18 previews total**, spanning all six classes. A separate review by the user and coding mentor found no obvious gross box-position or class-mapping errors; some overlay labels are crowded near image boundaries. This is sampled verification, not proof of perfect annotation quality.

## Commands to reproduce on a fresh environment

From the repository root, with the raw dataset in the expected location and dependencies installed:

```bash
python -m unittest discover -s tests -v
python -m scripts.finish_issue2 --preflight
python -m scripts.finish_issue2 --build
```

**Important:** The build intentionally refuses to overwrite an existing `data/processed/NEU-DET-YOLO` directory. Recreating output must be an explicit, controlled decision; never run cleanup commands against `data/raw`.

## Issue #3 handoff

1. Check `data.yaml` path portability and the presence of `numpy`, `Pillow`, and any other required dependencies in the environment.
2. Use **train** for fitting, **val** for tuning/early stopping, and preserve **test** for final evaluation only.
3. Validate class IDs 0–5 and run one model-inference smoke test before a full baseline run.
4. Record exact model checkpoint, library versions, seed, device, image size, batch size, and result directory.
5. Compare first measured baseline against the planned primary mAP@0.5:0.95 and supporting per-class metrics; don't invent benchmark results.

**Planning note:** This derived split policy supersedes the original Issue #1 experiment plan's use of the original validation split for routine model selection. The original validation images are now final held-out test images.
