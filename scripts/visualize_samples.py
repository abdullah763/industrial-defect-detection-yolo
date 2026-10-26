import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGES_DIR = PROJECT_ROOT / "data/raw/NEU-DET/train/images"
TRAIN_ANNOTATIONS_DIR = PROJECT_ROOT / "data/raw/NEU-DET/train/annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs/inspection"
SAMPLE_COUNT = 20
SAMPLES_PER_SHEET = 10
RANDOM_SEED = 42


def draw_contact_sheet(
    samples: list[tuple[Path, Path]], output_path: Path, sheet_number: int
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(15, 7))
    figure.suptitle(f"NEU-DET training samples ({sheet_number}/2)")

    for axis, (image_path, annotation_path) in zip(axes.flat, samples):
        with Image.open(image_path) as image:
            image_width, image_height = image.size
            axis.imshow(image.convert("RGB"))

        annotation = ET.parse(annotation_path).getroot()
        for obj in annotation.findall("object"):
            label = obj.findtext("name")
            box = obj.find("bndbox")
            if label is None or box is None:
                raise ValueError(f"Invalid object annotation in {annotation_path}")

            xmin_text = box.findtext("xmin")
            ymin_text = box.findtext("ymin")
            xmax_text = box.findtext("xmax")
            ymax_text = box.findtext("ymax")
            if (
                xmin_text is None
                or ymin_text is None
                or xmax_text is None
                or ymax_text is None
            ):
                raise ValueError(f"Incomplete bounding box in {annotation_path}")

            xmin, ymin, xmax, ymax = (
                int(xmin_text),
                int(ymin_text),
                int(xmax_text),
                int(ymax_text),
            )

            valid_x = 0 <= xmin < xmax <= image_width
            valid_y = 0 <= ymin < ymax <= image_height

            if not (valid_x and valid_y):
                print(
                    f"WARNING: Invalid bounding box in "
                    f"{annotation_path.name}: "
                    f"({xmin}, {ymin}, {xmax}, {ymax})"
                )
                continue

            axis.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
            )
            axis.text(
                xmin + 2,
                ymin + 2,
                label,
                color="red",
                fontsize=7,
                verticalalignment="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        axis.set_title(image_path.name, fontsize=9)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    annotation_paths = sorted(TRAIN_ANNOTATIONS_DIR.glob("*.xml"))
    annotations_by_stem = {path.stem: path for path in annotation_paths}
    class_candidates = {
        directory.name: []
        for directory in sorted(TRAIN_IMAGES_DIR.iterdir())
        if directory.is_dir()
    }
    missing_annotations = []

    for image_path in sorted(TRAIN_IMAGES_DIR.glob("*/*.jpg")):
        annotation_path = annotations_by_stem.get(image_path.stem)
        if annotation_path is None:
            missing_annotations.append(image_path)
            continue
        class_candidates[image_path.parent.name].append((image_path, annotation_path))

    if missing_annotations:
        print("Skipping images without matching annotations:")
        for image_path in missing_annotations:
            print(f"  {image_path.relative_to(PROJECT_ROOT)}")
    else:
        print("No training images are missing annotations.")

    class_names = sorted(class_candidates)
    if not class_names:
        raise FileNotFoundError(f"No class folders found in {TRAIN_IMAGES_DIR}")

    samples_per_class, extra_samples = divmod(SAMPLE_COUNT, len(class_names))
    if any(
        len(class_candidates[class_name]) < samples_per_class
        for class_name in class_names
    ):
        raise ValueError("Not enough annotated images to sample from every class.")

    generator = random.Random(RANDOM_SEED)
    samples = []
    remaining_candidates = []
    for class_name in class_names:
        candidates = class_candidates[class_name]
        class_samples = generator.sample(candidates, samples_per_class)
        samples.extend(class_samples)
        remaining_candidates.extend(
            candidate for candidate in candidates if candidate not in class_samples
        )

    if len(remaining_candidates) < extra_samples:
        raise ValueError(f"Fewer than {SAMPLE_COUNT} annotated training images found.")
    samples.extend(generator.sample(remaining_candidates, extra_samples))
    samples.sort(key=lambda sample: (sample[0].parent.name, sample[0].name))

    print("Selected filenames:")
    for image_path, _ in samples:
        print(f"  {image_path.name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sheet_index in range(2):
        start = sheet_index * SAMPLES_PER_SHEET
        sheet_samples = samples[start : start + SAMPLES_PER_SHEET]
        output_path = OUTPUT_DIR / f"training_samples_{sheet_index + 1}.png"
        draw_contact_sheet(sheet_samples, output_path, sheet_index + 1)
        print(f"Saved contact sheet: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
