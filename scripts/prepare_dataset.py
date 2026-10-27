
from pathlib import Path
from collections import defaultdict
from random import Random
from hashlib import sha256

DATA_ROOT = Path("data/raw/NEU-DET")


def index_files(root, folder_name, extension):
    files = {}

    for split in ("train", "validation"):
        folder = root / split / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        for path in sorted(folder.rglob(f"*{extension}")):
            stem = path.stem

            if stem in files:
                raise ValueError(
                    f"Duplicate filename identity: {stem}\n"
                    f"First: {files[stem][1]}\n"
                    f"Second: {path}"
                )

            files[stem] = (split, path)

    return files


def find_exact_duplicates(images):
    hashes = defaultdict(list)

    for stem, (split, image_path) in images.items():
        image_bytes = image_path.read_bytes()
        file_hash = sha256(image_bytes).hexdigest()

        hashes[file_hash].append(stem)

    duplicate_groups = []

    for stems in hashes.values():
        if len(stems) > 1:
            duplicate_groups.append(sorted(stems))

    return sorted(duplicate_groups)


def resolve_known_duplicates(images, duplicate_groups):
    selected = dict(images)

    if not duplicate_groups:
        return selected, []

    expected = [["patches_101", "patches_105"]]

    if duplicate_groups != expected:
        raise ValueError(
            f"Unexpected duplicate groups: {duplicate_groups}"
        )

    if any(images[stem][0] != "train" for stem in expected[0]):
        raise ValueError("Known duplicates are not both in training")

    # Exclude patches_105 only in the derived dataset.
    del selected["patches_105"]

    decisions = [
        {
            "retained": "patches_101",
            "excluded": "patches_105",
            "reason": "Identical image bytes; annotations differ",
        }
    ]

    return selected, decisions


def inspect_pairs(root):
    images = index_files(root, "images", ".jpg")
    annotations = index_files(root, "annotations", ".xml")

    missing = set(images) - set(annotations)
    unmatched = set(annotations) - set(images)

    if missing or unmatched:
        raise ValueError(
            f"Missing XMLs: {sorted(missing)}; "
            f"Unmatched XMLs: {sorted(unmatched)}"
        )

    cross_split = {}

    for stem, (image_split, image_path) in images.items():
        xml_split, xml_path = annotations[stem]

        if image_split != xml_split:
            cross_split[stem] = (image_split, xml_split)

    expected_exception = {
        "crazing_240": ("train", "validation")
    }

    if cross_split != expected_exception:
        raise ValueError(
            f"Unexpected cross-split annotations: {cross_split}"
        )

    print("Images indexed:", len(images))
    print("Annotations indexed:", len(annotations))
    print("Unique image/XML pairs:", len(images))
    print("Cross-split exception:", cross_split)

    duplicate_groups = find_exact_duplicates(images)

    print("Exact duplicate image groups:", len(duplicate_groups))

    for group in duplicate_groups[:10]:
        print("Duplicate group:", group)

    # Resolve duplicates outside the loop.
    selected_images, decisions = resolve_known_duplicates(
        images, duplicate_groups
    )

    print("Selected unique images:", len(selected_images))
    print("Duplicate resolutions:", decisions)

    return selected_images


def build_split_manifest(selected_images, seed=42):
    rng = Random(seed)

    # Group original training images by folder category.
    groups = {}
    test_stems = []

    for stem, (source_split, image_path) in selected_images.items():

        if source_split == "validation":
            # Original validation becomes the held-out test set.
            test_stems.append(stem)

        elif source_split == "train":
            category = image_path.parent.name
            groups.setdefault(category, []).append(stem)

        else:
            raise ValueError(f"Unknown source split: {source_split}")

    train_stems = []
    val_stems = []

    # Process categories in a deterministic order.
    for category in sorted(groups):
        stems = sorted(groups[category])

        if len(stems) < 2:
            raise ValueError(
                f"Not enough images to split category: {category}"
            )

        rng.shuffle(stems)

        val_count = round(len(stems) * 0.20)
        val_count = max(1, min(val_count, len(stems) - 1))

        val_stems.extend(stems[:val_count])
        train_stems.extend(stems[val_count:])

    manifest = {
        "train": sorted(train_stems),
        "val": sorted(val_stems),
        "test": sorted(test_stems),
    }

    # Confirm that no image appears in multiple partitions.
    all_stems = []

    for split_stems in manifest.values():
        all_stems.extend(split_stems)

    if len(all_stems) != len(set(all_stems)):
        raise ValueError("Data leakage: duplicate image identities")

    if set(all_stems) != set(selected_images):
        raise ValueError("Split manifest does not cover selected images")

    return manifest


if __name__ == "__main__":
    selected_images = inspect_pairs(DATA_ROOT)

    manifest = build_split_manifest(
        selected_images,
        seed=42
    )

    print("\nProposed dataset split:")

    for split_name, stems in manifest.items():
        print(f"  {split_name}: {len(stems)} images")
