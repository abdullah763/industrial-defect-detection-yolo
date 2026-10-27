"""Review cross-split image similarity; safely export verified VOC -> YOLO dataset.

Run from repository root: python -m scripts.finish_issue2 --preflight
After review:          python -m scripts.finish_issue2 --build
This never changes data/raw. See --help for explicit exclusions.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.check_near_duplicates import find_cross_split_candidates
from scripts.convert_voc_to_yolo import CLASS_NAMES, convert_annotation
from scripts.prepare_dataset import (
    DATA_ROOT, build_split_manifest, index_files, inspect_pairs,
)

SPLITS = ("train", "val", "test")
SIMILARITY_THRESHOLD = 4  # dHash candidate threshold; NOT a proof of duplication
MAX_REVIEW = 20


def image_vector(path: Path) -> np.ndarray:
    """Downsampled grayscale pixel values used to refine broad dHash candidates."""
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image).resize((64, 64), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float32).reshape(-1)


def compare_vectors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return 8-bit mean absolute error and Pearson correlation."""
    mae = float(np.mean(np.abs(a - b)))
    az = a - a.mean()
    bz = b - b.mean()
    denom = float(np.linalg.norm(az) * np.linalg.norm(bz))
    if denom == 0:
        corr = 1.0 if np.array_equal(a, b) else 0.0
    else:
        corr = float(np.dot(az, bz) / denom)
    return round(mae, 4), round(max(-1.0, min(1.0, corr)), 6)


def priority_review(mae: float, correlation: float) -> bool:
    """Conservative *review flag*, not automatic duplicate declaration."""
    return mae <= 4.0 and correlation >= 0.995


def review_near_duplicates(images, manifest, report_dir: Path, max_distance=SIMILARITY_THRESHOLD):
    """Write full candidate CSV and a representative side-by-side contact sheet."""
    report_dir.mkdir(parents=True, exist_ok=True)
    candidates = find_cross_split_candidates(images, manifest, max_distance=max_distance)
    cache: dict[str, np.ndarray] = {}

    def vector(stem):
        if stem not in cache:
            cache[stem] = image_vector(images[stem][1])
        return cache[stem]

    rows = []
    for distance, split_a, stem_a, split_b, stem_b in candidates:
        mae, corr = compare_vectors(vector(stem_a), vector(stem_b))
        rows.append({
            "dhash_distance": distance,
            "mae_0to255": mae,
            "correlation": corr,
            "priority_review": priority_review(mae, corr),
            "split_a": split_a,
            "image_a": stem_a,
            "split_b": split_b,
            "image_b": stem_b,
        })
    rows.sort(key=lambda r: (not r["priority_review"], r["mae_0to255"], -r["correlation"], r["image_a"], r["image_b"]))

    csv_path = report_dir / "near_duplicate_candidates.csv"
    fields = ["priority_review", "dhash_distance", "mae_0to255", "correlation", "split_a", "image_a", "split_b", "image_b"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    preview_path = report_dir / "near_duplicate_preview.jpg"
    if rows:
        chosen = rows[:MAX_REVIEW]
        tile_w, tile_h, label_h = 180, 180, 34
        canvas = Image.new("RGB", (tile_w * 2, len(chosen) * (tile_h + label_h)), "white")
        painter = ImageDraw.Draw(canvas)
        for idx, row in enumerate(chosen):
            y = idx * (tile_h + label_h)
            for col, stem in enumerate((row["image_a"], row["image_b"])):
                with Image.open(images[stem][1]) as image:
                    thumb = image.convert("RGB").resize((tile_w, tile_h))
                    canvas.paste(thumb, (col * tile_w, y + label_h))
            label = f"dH={row['dhash_distance']}  MAE={row['mae_0to255']:.2f}  corr={row['correlation']:.4f}"
            painter.text((3, y + 1), label, fill="black")
            painter.text((3, y + 16), f"{row['split_a']}/{row['image_a']}", fill="black")
            painter.text((tile_w + 3, y + 16), f"{row['split_b']}/{row['image_b']}", fill="black")
        canvas.save(preview_path, quality=90)

    flagged = sum(row["priority_review"] for row in rows)
    return {"total": len(rows), "priority_review": flagged, "rows": rows, "csv": str(csv_path), "preview": str(preview_path) if rows else None}


def load_exclusions(path: Path | None, images: dict) -> list[str]:
    """Only explicit original-train exclusions allowed; preserve held-out test."""
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    stems = [s for s in entries if s and not s.startswith("#")]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate names in exclusions file")
    for stem in stems:
        if stem not in images:
            raise ValueError(f"Unknown exclusion: {stem}")
        if images[stem][0] != "train":
            raise ValueError(f"Cannot exclude original held-out test image: {stem}")
    return sorted(stems)


def prepare_records(images, manifest, raw_root: Path) -> tuple[list[dict], dict]:
    """Validate every selected image/XML and produce complete in-memory label records."""
    annotations = index_files(raw_root, "annotations", ".xml")
    records = []
    counts = {split: Counter() for split in SPLITS}
    special = []
    for split in SPLITS:
        for stem in manifest[split]:
            source_split, image_path = images[stem]
            xml_source_split, xml_path = annotations[stem]
            with Image.open(image_path) as image:
                actual_w, actual_h = image.size
                image.load()  # fail on unreadable/corrupt source
            root = ET.parse(xml_path).getroot()
            declared_w = int(root.findtext("size/width"))
            declared_h = int(root.findtext("size/height"))
            if (actual_w, actual_h) != (declared_w, declared_h):
                raise ValueError(f"XML/image size disagreement: {stem}")
            label_lines = convert_annotation(xml_path)
            if len(label_lines) != len(root.findall("object")):
                raise ValueError(f"Silent object loss: {stem}")
            for line in label_lines:
                pieces = line.split()
                validate_yolo_line(line)
                counts[split][CLASS_NAMES[int(pieces[0])]] += 1
            if source_split != xml_source_split:
                special.append({"stem": stem, "image_source": source_split, "xml_source": xml_source_split, "derived_split": split})
            records.append({"stem": stem, "split": split, "source_split": source_split, "image": image_path, "xml": xml_path, "labels": label_lines})
    if len(records) != len(images):
        raise ValueError("Source-to-derived image count mismatch")
    missing_classes = [s for s in SPLITS if any(counts[s][name] == 0 for name in CLASS_NAMES)]
    if missing_classes:
        raise ValueError(f"Some splits are missing annotated classes: {missing_classes}")
    return records, {s: dict(counts[s]) for s in SPLITS}, special


def validate_yolo_line(line: str):
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Not five YOLO fields: {line!r}")
    class_str, *values = parts
    try:
        cls = int(class_str)
        x, y, w, h = (float(v) for v in values)
    except ValueError as exc:
        raise ValueError(f"Non-numeric YOLO label: {line!r}") from exc
    if not 0 <= cls < len(CLASS_NAMES):
        raise ValueError(f"Invalid class ID {cls}")
    if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"Invalid normalized box: {line!r}")
    tolerance = 1e-5  # account for label rounding to six decimal places
    if x - w / 2 < -tolerance or x + w / 2 > 1 + tolerance or y - h / 2 < -tolerance or y + h / 2 > 1 + tolerance:
        raise ValueError(f"Normalized box extends outside image: {line!r}")


def source_object_reconciliation(raw_root: Path, records: list[dict]) -> dict:
    """Count objects without silently treating rejected annotations as retained."""
    annotations = index_files(raw_root, "annotations", ".xml")
    total_source = sum(len(ET.parse(path).getroot().findall("object")) for _, path in annotations.values())
    retained = sum(len(record["labels"]) for record in records)
    excluded_stems = sorted(set(annotations) - {r["stem"] for r in records})
    excluded = sum(len(ET.parse(annotations[stem][1]).getroot().findall("object")) for stem in excluded_stems)
    rejected = 0  # Fail-fast conversion never silently skips a bad object.
    if retained + excluded + rejected != total_source:
        raise ValueError("Object-count reconciliation failed")
    return {"source_objects": total_source, "retained_objects": retained, "excluded_objects": excluded,
            "rejected_objects": rejected, "excluded_stems": excluded_stems}


def make_label_previews(tmp: Path, records: list[dict]):
    """Render YOLO boxes from generated label values for six classes in each split."""
    palette = ["#00AEEF", "#E55353", "#42A66D", "#C879E1", "#E5A92A", "#8A9BF0"]
    for split in SPLITS:
        samples = []
        for name in CLASS_NAMES:
            # Prefer source folder of this category; all object labels are drawn from XML.
            hit = next((r for r in records if r["split"] == split and r["image"].parent.name == name), None)
            if hit is not None:
                samples.append(hit)
        if not samples:
            continue
        tile_w, tile_h = 200, 225
        sheet = Image.new("RGB", (tile_w * 3, tile_h * 2), "white")
        for i, record in enumerate(samples[:6]):
            with Image.open(tmp / "images" / split / (record["stem"] + ".jpg")) as im:
                img = im.convert("RGB").resize((200, 200))
            draw = ImageDraw.Draw(img)
            for line in record["labels"]:
                parts = line.split()
                cls = int(parts[0])
                x, y, w, h = [float(v) for v in parts[1:]]
                coords = (round((x - w / 2) * 200), round((y - h / 2) * 200),
                          round((x + w / 2) * 200), round((y + h / 2) * 200))
                draw.rectangle(coords, outline=palette[cls], width=2)
                draw.text((max(0, coords[0]), max(0, coords[1] - 11)), CLASS_NAMES[cls], fill=palette[cls])
            x0, y0 = (i % 3) * tile_w, (i // 3) * tile_h
            sheet.paste(img, (x0, y0 + 25))
            ImageDraw.Draw(sheet).text((x0 + 3, y0 + 5), f"{split}/{record['stem']}", fill="black")
        sheet.save(tmp / f"preview_{split}.jpg", quality=90)


def make_yaml(final_path: Path) -> str:
    # YAML strings are quoted using JSON's compatible double-quoted syntax.
    lines = [f"path: {json.dumps(str(final_path.resolve()))}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    return "\n".join(lines) + "\n"


def export_dataset(output_dir: Path, records: list[dict], manifest: dict, class_counts: dict, special: list, exclusions: list, review_summary: dict, seed=42, reconciliation=None):
    """Write under temporary directory, verify, then atomically publish a new path."""
    output_dir = output_dir.resolve()
    raw_resolved = DATA_ROOT.resolve()
    if output_dir == raw_resolved or raw_resolved in output_dir.parents:
        raise ValueError("Refusing to write derived data inside data/raw")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; not overwriting: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".neu_yolo_staging_", dir=output_dir.parent))
    try:
        for split in SPLITS:
            (tmp / "images" / split).mkdir(parents=True)
            (tmp / "labels" / split).mkdir(parents=True)
        for record in records:
            split, stem = record["split"], record["stem"]
            shutil.copy2(record["image"], tmp / "images" / split / f"{stem}.jpg")
            (tmp / "labels" / split / f"{stem}.txt").write_text("\n".join(record["labels"]) + "\n", encoding="utf-8")
        (tmp / "data.yaml").write_text(make_yaml(output_dir), encoding="utf-8")
        manifest_payload = {
            "seed": seed,
            "partitions": manifest,
            "known_exact_duplicate_exclusion": "patches_105",
            "additional_reviewed_exclusions": exclusions,
            "cross_split_xml_resolutions": special,
            "near_duplicate_screening": {"dhash_threshold": SIMILARITY_THRESHOLD, "total_dhash_candidates": review_summary["total"], "priority_review_count": review_summary["priority_review"]},
            "derived_object_counts": class_counts,
            "object_reconciliation": reconciliation,
        }
        (tmp / "split_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = ["# Issue #2 — Derived NEU-DET dataset validation", "", "All records below describe derived data, not modifications to data/raw.", "", f"Seed: {seed}", f"Derived images: {len(records)}", f"Derived objects: {sum(sum(c.values()) for c in class_counts.values())}", f"Object reconciliation (source / retained / excluded / rejected): {reconciliation}", f"Explicit extra exclusions: {exclusions}", f"Near-duplicate dHash candidates: {review_summary['total']} (not all confirmed duplicates)", f"High-priority review pairs remaining: {review_summary['priority_review']}", "", "| Split | Images | Objects |", "|---|---:|---:|"]
        for split in SPLITS:
            report.append(f"| {split} | {len(manifest[split])} | {sum(class_counts[split].values())} |")
        report.extend(["", "## Per-split object count", "", "| Class | train | val | test |", "|---|---:|---:|---:|"])
        for name in CLASS_NAMES:
            report.append(f"| {name} | {class_counts['train'].get(name, 0)} | {class_counts['val'].get(name, 0)} | {class_counts['test'].get(name, 0)} |")
        report.extend(["", "## Cross-split XML source resolution", "", "```json", json.dumps(special, indent=2), "```", "", "The XML is paired by image stem and assigned to the derived partition determined by its IMAGE source membership.", "The original 360-image validation set is reserved as test data; do not tune models on it.", ""])
        (tmp / "integrity_report.md").write_text("\n".join(report), encoding="utf-8")
        make_label_previews(tmp, records)
        # Verify written files and every actual label line before publication.
        for split in SPLITS:
            images_written = list((tmp / "images" / split).glob("*.jpg"))
            labels_written = list((tmp / "labels" / split).glob("*.txt"))
            if len(images_written) != len(manifest[split]) or len(labels_written) != len(manifest[split]):
                raise ValueError(f"Written image/label counts invalid for {split}")
            for label in labels_written:
                for line in label.read_text(encoding="utf-8").splitlines():
                    validate_yolo_line(line)
        tmp.rename(output_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Screen similarity and create review evidence only (default)")
    parser.add_argument("--build", action="store_true", help="Generate derived data only if no priority near-duplicate pairs remain")
    parser.add_argument("--exclude-file", type=Path, help="Reviewed original-train image stems to omit, one per line; reasons should be documented")
    parser.add_argument("--output", type=Path, default=Path("data/processed/NEU-DET-YOLO"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/inspection/issue-02"))
    args = parser.parse_args()
    if args.build and args.preflight:
        parser.error("Choose either --preflight or --build")
    images = inspect_pairs(DATA_ROOT)
    excluded = load_exclusions(args.exclude_file, images)
    selected_images = {k: v for k, v in images.items() if k not in excluded}
    manifest = build_split_manifest(selected_images, seed=42)
    print("Proposed partitions:", {s: len(v) for s, v in manifest.items()})
    similarity = review_near_duplicates(selected_images, manifest, args.report_dir)
    print(f"dHash candidates: {similarity['total']}; high-priority pixel/correlation review: {similarity['priority_review']}")
    print("Candidate CSV:", similarity["csv"])
    if similarity["preview"]:
        print("Preview:", similarity["preview"])
    if not args.build:
        print("PREFLIGHT ONLY: no dataset generated. Review candidate CSV/preview before --build.")
        return
    if similarity["priority_review"]:
        raise SystemExit("BUILD BLOCKED: high-priority cross-split similarity pairs remain. Review them; do not automatically exclude them. Optionally supply reviewed original-train exclusions via --exclude-file.")
    records, class_counts, special = prepare_records(selected_images, manifest, DATA_ROOT)
    reconciliation = source_object_reconciliation(DATA_ROOT, records)
    print("Object reconciliation:", reconciliation)
    final_dir = export_dataset(args.output, records, manifest, class_counts, special, excluded, similarity, reconciliation=reconciliation)
    print("DATASET BUILT:", final_dir)
    print("Report:", final_dir / "integrity_report.md")
    print("Manifest:", final_dir / "split_manifest.json")
    print("YOLO YAML:", final_dir / "data.yaml")


if __name__ == "__main__":
    main()
