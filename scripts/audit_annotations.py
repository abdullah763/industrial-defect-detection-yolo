"""Read-only quality audit for the NEU-DET training and validation splits."""

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data/raw/NEU-DET"
OUTPUT_PATH = PROJECT_ROOT / "outputs/inspection/annotation_audit.txt"
SPLITS = ("train", "validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_CLASSES = {
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled-in_scale", "scratches",
}


def classify_filename(xml_filename: str | None, image_path: Path) -> str:
    """Distinguish omitted extensions from genuinely different filenames."""
    if not xml_filename or not xml_filename.strip():
        return "missing_filename_field"

    filename = xml_filename.strip().replace("\\", "/").rsplit("/", 1)[-1]
    if filename == image_path.name:
        return "exact_match"
    if filename == image_path.stem:
        return "extension_omitted"
    if Path(filename).stem == image_path.stem and Path(filename).suffix:
        return "different_extension_or_case"
    return "different_basename"


def record(counter: Counter[str], details: list[str], kind: str, message: str) -> None:
    counter[kind] += 1
    details.append(f"[{kind}] {message}")


def read_integer(
    value: str | None,
    field: str,
    location: str,
    issues: Counter[str],
    details: list[str],
    *,
    positive: bool = False,
) -> int | None:
    if value is None or not value.strip():
        record(issues, details, f"missing_{field}", location)
        return None
    try:
        result = int(value)
    except ValueError:
        record(issues, details, f"nonnumeric_{field}", f"{location}: {value!r}")
        return None
    if positive and result <= 0:
        record(issues, details, f"invalid_{field}", f"{location}: {result}")
        return None
    return result


def collect_images(folder: Path) -> dict[str, list[Path]]:
    images: dict[str, list[Path]] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(path.stem, []).append(path)
    return images


def inspect_xml(
    xml_path: Path,
    matched_image: Path | None,
    image_size: tuple[int, int] | None,
    issues: Counter[str],
    issue_details: list[str],
    observations: Counter[str],
    observation_details: list[str],
) -> Counter[str]:
    """Audit one annotation and return object counts grouped by XML label."""
    location = str(xml_path.relative_to(PROJECT_ROOT))
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as error:
        record(issues, issue_details, "invalid_xml", f"{location}: {error}")
        return Counter()

    if matched_image is not None:
        filename_category = classify_filename(root.findtext("filename"), matched_image)
        if filename_category == "extension_omitted":
            record(
                observations, observation_details, filename_category,
                f"{location}: XML omits the extension for {matched_image.name}",
            )
        elif filename_category != "exact_match":
            record(
                issues, issue_details, filename_category,
                f"{location}: XML filename={root.findtext('filename')!r}, "
                f"image={matched_image.name!r}",
            )

    xml_width = read_integer(
        root.findtext("size/width"), "xml_width", location,
        issues, issue_details, positive=True,
    )
    xml_height = read_integer(
        root.findtext("size/height"), "xml_height", location,
        issues, issue_details, positive=True,
    )
    if image_size is not None and xml_width is not None and xml_height is not None:
        if (xml_width, xml_height) != image_size:
            record(
                issues, issue_details, "dimension_mismatch",
                f"{location}: XML={xml_width}x{xml_height}, image={image_size}",
            )

    labels: Counter[str] = Counter()
    for index, obj in enumerate(root.findall("object"), start=1):
        object_location = f"{location} object {index}"
        label = (obj.findtext("name") or "").strip()
        labels[label or "(missing)"] += 1
        if not label:
            record(issues, issue_details, "missing_label", object_location)
        elif label not in EXPECTED_CLASSES:
            record(issues, issue_details, "unexpected_label", f"{object_location}: {label!r}")

        if matched_image is not None and label and label != matched_image.parent.name:
            record(
                observations, observation_details, "label_folder_mismatch",
                f"{object_location}: label {label!r} differs from "
                f"folder {matched_image.parent.name!r}",
            )

        box = obj.find("bndbox")
        if box is None:
            record(issues, issue_details, "missing_bndbox", object_location)
            continue

        coords = [
            read_integer(box.findtext(field), f"bbox_{field}", object_location,
                         issues, issue_details)
            for field in ("xmin", "ymin", "xmax", "ymax")
        ]
        if any(value is None for value in coords):
            continue
        xmin, ymin, xmax, ymax = coords
        if image_size is not None:
            width, height = image_size
            valid = 0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height
        else:
            valid = 0 <= xmin < xmax and 0 <= ymin < ymax
        if not valid:
            record(
                issues, issue_details, "invalid_bbox_coordinates",
                f"{object_location}: {(xmin, ymin, xmax, ymax)}, image={image_size}",
            )
    return labels


def main() -> None:
    issues: Counter[str] = Counter()
    observations: Counter[str] = Counter()
    issue_details: list[str] = []
    observation_details: list[str] = []
    images_by_split: dict[str, dict[str, list[Path]]] = {}
    xmls_by_split: dict[str, dict[str, Path]] = {}

    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        xml_dir = DATASET_DIR / split / "annotations"
        if not image_dir.is_dir() or not xml_dir.is_dir():
            raise FileNotFoundError(f"Missing image or annotation directory: {split}")
        images_by_split[split] = collect_images(image_dir)
        xmls_by_split[split] = {
            path.stem: path for path in sorted(xml_dir.glob("*.xml"))
        }

    seen_cross_split: set[str] = set()
    split_summaries: list[str] = []
    total_objects = 0

    for split in SPLITS:
        other_split = "validation" if split == "train" else "train"
        images = images_by_split[split]
        annotations = xmls_by_split[split]
        other_images = images_by_split[other_split]
        other_annotations = xmls_by_split[other_split]
        image_count = sum(len(paths) for paths in images.values())

        for stem, paths in images.items():
            if stem in annotations:
                continue
            for image_path in paths:
                record(
                    issues, issue_details, "missing_annotation",
                    f"{image_path.relative_to(PROJECT_ROOT)}",
                )
            if stem in other_annotations and stem not in seen_cross_split:
                seen_cross_split.add(stem)
                record(
                    issues, issue_details, "cross_split_annotation",
                    f"{stem}: image in {split}, XML in {other_split}",
                )

        label_counts: Counter[str] = Counter()
        for stem, xml_path in annotations.items():
            matches = images.get(stem, [])
            if not matches:
                record(
                    issues, issue_details, "unmatched_annotation",
                    str(xml_path.relative_to(PROJECT_ROOT)),
                )
                matches = other_images.get(stem, [])
                if matches and stem not in seen_cross_split:
                    seen_cross_split.add(stem)
                    record(
                        issues, issue_details, "cross_split_annotation",
                        f"{stem}: XML in {split}, image in {other_split}",
                    )

            image_path = None
            image_size = None
            if len(matches) == 1:
                image_path = matches[0]
                try:
                    with Image.open(image_path) as image:
                        image_size = image.size
                except (OSError, ValueError) as error:
                    record(
                        issues, issue_details, "unreadable_image",
                        f"{image_path.relative_to(PROJECT_ROOT)}: {error}",
                    )
            else:
                reason = "no image" if not matches else "multiple candidate images"
                if matches:
                    record(issues, issue_details, "ambiguous_image_match", str(xml_path))
                record(
                    issues, issue_details, "unverifiable_annotation",
                    f"{xml_path.relative_to(PROJECT_ROOT)}: {reason}",
                )
            if image_path is not None and image_size is None:
                record(
                    issues, issue_details, "unverifiable_annotation",
                    f"{xml_path.relative_to(PROJECT_ROOT)}: image cannot be opened",
                )
            label_counts.update(
                inspect_xml(
                    xml_path, image_path, image_size, issues, issue_details,
                    observations, observation_details,
                )
            )

        objects = sum(label_counts.values())
        total_objects += objects
        split_summaries.append(
            f"{split}: {image_count} images, {len(annotations)} XMLs, {objects} objects"
        )
        split_summaries.extend(
            f"  {name}: {count} objects" for name, count in sorted(label_counts.items())
        )

    report = [
        "NEU-DET annotation audit", f"Dataset: {DATASET_DIR}", "",
        *split_summaries, "", f"Total objects: {total_objects}", "",
        "Issues by type:",
        *(f"  {key}: {value}" for key, value in sorted(issues.items())),
        *([] if issues else ["  None"]), "",
        "Observations (not automatic errors):",
        *(f"  {key}: {value}" for key, value in sorted(observations.items())),
        *([] if observations else ["  None"]), "",
        "Issue details:", *(issue_details or ["  None"]), "",
        "Observation details:", *(observation_details or ["  None"]),
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(line for line in split_summaries if not line.startswith("  ")))
    print(f"Total objects: {total_objects}")
    print("Issues by type:")
    for name, count in sorted(issues.items()):
        print(f"  {name}: {count}")
    if not issues:
        print("  None")
    print("Observations:")
    for name, count in sorted(observations.items()):
        print(f"  {name}: {count}")
    if not observations:
        print("  None")
    print(f"Report: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
