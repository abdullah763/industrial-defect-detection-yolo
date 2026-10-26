"""Display one NEU-DET image with bounding boxes from its XML annotation."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data/raw/NEU-DET"


def visualize(image_path: Path, annotation_path: Path) -> None:
    root = ET.parse(annotation_path).getroot()
    objects = root.findall("object")

    with Image.open(image_path) as image:
        image_size = image.size
        rgb_image = image.convert("RGB")

    xml_size = (
        int(root.findtext("size/width")),
        int(root.findtext("size/height")),
    )
    if xml_size != image_size:
        print(f"Warning: XML size {xml_size} differs from image size {image_size}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image)
    valid_count = 0

    for index, obj in enumerate(objects, start=1):
        label = (obj.findtext("name") or "").strip()
        try:
            xmin, ymin, xmax, ymax = (
                int(obj.findtext(f"bndbox/{key}"))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError) as error:
            print(f"Warning: object {index} has invalid coordinates: {error}")
            continue

        width, height = image_size
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            print(f"Warning: object {index} has an out-of-bounds box")
            continue

        ax.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor="red", linewidth=2,
        ))
        ax.text(
            xmin + 3, ymin + 3, label or "(missing label)",
            color="red", fontsize=9, verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        valid_count += 1

    ax.set_title(f"{image_path.name} — {valid_count}/{len(objects)} boxes")
    ax.axis("off")
    print(f"Image: {image_path.name} ({image_size[0]}x{image_size[1]})")
    print(f"Annotation: {annotation_path.name}")
    print(f"Valid boxes: {valid_count}/{len(objects)}")
    plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path,
        default=DATASET_DIR / "train/images/crazing/crazing_1.jpg",
        help="Path to image (default: train crazing_1.jpg)",
    )
    parser.add_argument(
        "--xml", type=Path, help="Annotation path (defaults to same-split XML)",
    )
    args = parser.parse_args()
    image_path = args.image
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if args.xml is not None:
        xml_path = args.xml if args.xml.is_absolute() else PROJECT_ROOT / args.xml
    else:
        split = next((name for name in ("train", "validation") if name in image_path.parts), None)
        if split is None:
            parser.error("Image is outside known splits; provide --xml")
        xml_path = DATASET_DIR / split / "annotations" / f"{image_path.stem}.xml"

    visualize(image_path, xml_path)


if __name__ == "__main__":
    main()
