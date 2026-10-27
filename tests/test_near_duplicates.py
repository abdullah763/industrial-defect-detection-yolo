import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.check_near_duplicates import difference_hash, find_cross_split_candidates


def make_gradient(path, reverse=False):
    image = Image.new("L", (9, 8))
    image.putdata([
        (8 - x if reverse else x) * 25
        for y in range(8) for x in range(9)
    ])
    image.save(path)


class NearDuplicateTests(unittest.TestCase):
    def test_identical_images_have_same_hash(self):
        with TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.png", Path(temp) / "b.png"
            make_gradient(first)
            make_gradient(second)
            self.assertEqual(difference_hash(first), difference_hash(second))

    def test_opposite_gradients_have_different_hashes(self):
        with TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.png", Path(temp) / "b.png"
            make_gradient(first)
            make_gradient(second, reverse=True)
            distance = (difference_hash(first) ^ difference_hash(second)).bit_count()
            self.assertEqual(distance, 64)

    def test_flags_cross_split_but_not_within_split(self):
        with TemporaryDirectory() as temp:
            first, second, third = [Path(temp) / f"{x}.png" for x in ("a", "b", "c")]
            for path in (first, second, third):
                make_gradient(path)
            images = {
                "a": ("train", first),
                "b": ("train", second),
                "c": ("validation", third),
            }
            manifest = {"train": ["a", "b"], "val": ["c"], "test": []}
            result = find_cross_split_candidates(images, manifest, max_distance=0)
            self.assertEqual(result, [
                (0, "train", "a", "val", "c"),
                (0, "train", "b", "val", "c"),
            ])


if __name__ == "__main__":
    unittest.main()
