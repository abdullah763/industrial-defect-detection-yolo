from itertools import combinations

from PIL import Image


def difference_hash(image_path):
    """Return a 64-bit perceptual difference hash for an image."""
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = gray.tobytes()

    value = 0
    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]
            value = (value << 1) | int(left > right)
    return value


def find_cross_split_candidates(images, manifest, max_distance=4):
    """Find similar-looking images belonging to different proposed splits."""
    if not 0 <= max_distance <= 64:
        raise ValueError("max_distance must be between 0 and 64")

    all_stems = [stem for stems in manifest.values() for stem in stems]
    if len(all_stems) != len(set(all_stems)) or set(all_stems) != set(images):
        raise ValueError("Split manifest must cover each selected image exactly once")

    hashes = {
        stem: difference_hash(image_path)
        for stem, (_, image_path) in images.items()
    }
    candidates = []
    for first_split, second_split in combinations(("train", "val", "test"), 2):
        for first_stem in manifest.get(first_split, []):
            for second_stem in manifest.get(second_split, []):
                distance = (hashes[first_stem] ^ hashes[second_stem]).bit_count()
                if distance <= max_distance:
                    candidates.append((distance, first_split, first_stem,
                                       second_split, second_stem))
    return sorted(candidates)


def main():
    from scripts.prepare_dataset import DATA_ROOT, build_split_manifest, inspect_pairs

    images = inspect_pairs(DATA_ROOT)
    manifest = build_split_manifest(images, seed=42)
    print("Proposed splits:", {name: len(stems) for name, stems in manifest.items()})

    candidates = find_cross_split_candidates(images, manifest, max_distance=4)
    print("Cross-split near-duplicate candidates:", len(candidates))
    for distance, split_a, stem_a, split_b, stem_b in candidates[:20]:
        print(f"  dHash distance {distance}: {split_a}/{stem_a} <-> {split_b}/{stem_b}")
    if len(candidates) > 20:
        print(f"  ... {len(candidates) - 20} more candidates not shown")


if __name__ == "__main__":
    main()
