from pathlib import Path
import xml.etree.ElementTree as ET


CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

CLASS_TO_ID = {
    name: index for index, name in enumerate(CLASS_NAMES)
}


def voc_to_yolo(xmin, ymin, xmax, ymax, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")

    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid bounding box coordinates")

    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError("Bounding box outside image boundaries")

    return (
        (xmin + xmax) / (2 * width),
        (ymin + ymax) / (2 * height),
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def convert_annotation(xml_path):
    xml_path = Path(xml_path)

    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot read {xml_path}: {exc}") from exc

    try:
        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{xml_path}: invalid image size") from exc

    objects = root.findall("object")
    if not objects:
        raise ValueError(f"{xml_path}: no annotated objects")

    lines = []

    for index, obj in enumerate(objects, start=1):
        class_name = (obj.findtext("name") or "").strip()

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"{xml_path}: object {index}: unknown class {class_name!r}"
            )

        try:
            coords = [
                int(obj.findtext(f"bndbox/{field}"))
                for field in ("xmin", "ymin", "xmax", "ymax")
            ]
            normalized = voc_to_yolo(*coords, width, height)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{xml_path}: object {index}: invalid box: {exc}"
            ) from exc

        values = " ".join(f"{value:.6f}" for value in normalized)
        lines.append(f"{CLASS_TO_ID[class_name]} {values}")

    return lines


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/convert_voc_to_yolo.py FILE.xml")

    for line in convert_annotation(sys.argv[1]):
        print(line)