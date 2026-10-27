import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.prepare_dataset import find_exact_duplicates

from scripts.prepare_dataset import index_files, inspect_pairs, resolve_known_duplicates, build_split_manifest


class PairingTests(unittest.TestCase):

    def test_known_cross_split_exception(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)

            image = root / "train/images/crazing/crazing_240.jpg"
            xml = root / "validation/annotations/crazing_240.xml"

            image.parent.mkdir(parents=True)
            xml.parent.mkdir(parents=True)

            image.touch()
            xml.touch()

            # Create the other required folders
            (root / "validation/images").mkdir(parents=True)
            (root / "train/annotations").mkdir(parents=True)

            # This known mismatch is explicitly allowed.
            inspect_pairs(root)

    def test_duplicate_image_identity_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)

            first = root / "train/images/crazing/crazing_1.jpg"
            second = root / "validation/images/crazing/crazing_1.jpg"

            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)

            first.touch()
            second.touch()

            with self.assertRaises(ValueError):
                index_files(root, "images", ".jpg")

    def test_exact_duplicate_content_detected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)

            image1 = root / "crazing_1.jpg"
            image2 = root / "scratches_2.jpg"

            image1.write_bytes(b"same image content")
            image2.write_bytes(b"same image content")

            images = {
                "crazing_1": ("train", image1),
                "scratches_2": ("validation", image2),
            }

            duplicates = find_exact_duplicates(images)

            self.assertEqual(
                duplicates,
                [["crazing_1", "scratches_2"]]
            )


    def test_different_image_content(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)

            image1 = root / "crazing_1.jpg"
            image2 = root / "scratches_2.jpg"

            image1.write_bytes(b"image one")
            image2.write_bytes(b"image two")

            images = {
                "crazing_1": ("train", image1),
                "scratches_2": ("validation", image2),
            }

            duplicates = find_exact_duplicates(images)

            self.assertEqual(duplicates, [])

    def test_known_duplicate_resolution(self):
        images = {
            "patches_101": ("train", Path("101.jpg")),
            "patches_105": ("train", Path("105.jpg")),
        }

        selected, decisions = resolve_known_duplicates(
            images,
            [["patches_101", "patches_105"]]
        )

        self.assertEqual(len(selected), 1)
        self.assertIn("patches_101", selected)
        self.assertNotIn("patches_105", selected)
        self.assertEqual(decisions[0]["excluded"], "patches_105")

        # Original dictionary must remain unchanged.
        self.assertEqual(len(images), 2)


    def test_unknown_duplicate_rejected(self):
        images = {
            "crazing_1": ("train", Path("1.jpg")),
            "crazing_2": ("train", Path("2.jpg")),
        }

        with self.assertRaises(ValueError):
            resolve_known_duplicates(
                images,
                [["crazing_1", "crazing_2"]]
            )

    def test_deterministic_stratified_split(self):
        images = {}

        # Create 10 training images for each category.
        for category in ("crazing", "patches"):
            for index in range(10):
                stem = f"{category}_{index}"

                images[stem] = (
                    "train",
                    Path(f"train/images/{category}/{stem}.jpg"),
                )

        # Original validation image becomes final test image.
        images["scratches_100"] = (
            "validation",
            Path("validation/images/scratches/scratches_100.jpg"),
        )

        manifest1 = build_split_manifest(images, seed=42)

        # Change input dictionary order.
        reversed_images = dict(reversed(list(images.items())))
        manifest2 = build_split_manifest(reversed_images, seed=42)

        # Same seed must produce identical results.
        self.assertEqual(manifest1, manifest2)

        # Expected split counts.
        self.assertEqual(len(manifest1["train"]), 16)
        self.assertEqual(len(manifest1["val"]), 4)
        self.assertEqual(len(manifest1["test"]), 1)

        # Original validation must remain in test.
        self.assertEqual(
            manifest1["test"],
            ["scratches_100"]
        )

        # No image may appear in multiple partitions.
        all_stems = sum(manifest1.values(), [])

        self.assertEqual(len(all_stems), len(set(all_stems)))
        self.assertEqual(len(all_stems), len(images))


if __name__ == "__main__":
    unittest.main()