import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_yolo_dataset import (
    compare_vectors, priority_review, validate_yolo_line,
    load_exclusions, export_dataset, make_yaml,
)


class BuildYoloDatasetTests(unittest.TestCase):
    def test_similarity_identical_and_opposite(self):
        a = np.arange(64, dtype=np.float32)
        mae, corr = compare_vectors(a, a.copy())
        self.assertEqual((mae, corr), (0, 1))
        self.assertTrue(priority_review(mae, corr))
        mae2, corr2 = compare_vectors(a, -a)
        self.assertFalse(priority_review(mae2, corr2))

    def test_yolo_coordinates_valid_and_invalid(self):
        validate_yolo_line("2 0.490000 0.922500 0.200000 0.155000")
        for label in ("9 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.0 0.2", "0 0.95 0.5 0.2 0.3", "0 0.5 0.5 0.2"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_yolo_line(label)

    def test_only_original_train_can_be_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = {"a": ("train", root / "a.jpg"), "b": ("validation", root / "b.jpg")}
            path = root / "exclude.txt"
            path.write_text("# reviewed\na\n")
            self.assertEqual(load_exclusions(path, names), ["a"])
            path.write_text("b\n")
            with self.assertRaises(ValueError):
                load_exclusions(path, names)

    def test_export_dataset_writes_labels_and_manifest_without_source_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jpg"
            Image.new("RGB", (20, 20), color="gray").save(source)
            before = source.read_bytes()
            output = root / "derived"
            records = [{"stem": "crazing_1", "split": "train", "source_split": "train", "image": source, "xml": root / "source.xml", "labels": ["0 0.5 0.5 0.2 0.2"]}]
            manifest = {"train": ["crazing_1"], "val": [], "test": []}
            counts = {"train": {"crazing": 1}, "val": {}, "test": {}}
            evidence = {"total": 0, "priority_review": 0}
            result = export_dataset(output, records, manifest, counts, [], [], evidence)
            self.assertEqual(result.resolve(), output.resolve())
            self.assertEqual((output / "labels/train/crazing_1.txt").read_text(), "0 0.5 0.5 0.2 0.2\n")
            self.assertTrue((output / "data.yaml").is_file())
            self.assertTrue((output / "split_manifest.json").is_file())
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                export_dataset(output, records, manifest, counts, [], [], evidence)


if __name__ == "__main__":
    unittest.main()
