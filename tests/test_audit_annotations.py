"""Small synthetic-dataset tests for the read-only annotation audit."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts import audit_annotations as audit


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "data/raw/NEU-DET"
        for split in audit.SPLITS:
            (self.dataset / split / "images/crazing").mkdir(parents=True)
            (self.dataset / split / "annotations").mkdir(parents=True)

        self.original = audit.PROJECT_ROOT, audit.DATASET_DIR, audit.OUTPUT_PATH
        self.addCleanup(self.restore_paths)
        audit.PROJECT_ROOT = self.root
        audit.DATASET_DIR = self.dataset
        audit.OUTPUT_PATH = self.root / "outputs/inspection/annotation_audit.txt"

    def restore_paths(self) -> None:
        audit.PROJECT_ROOT, audit.DATASET_DIR, audit.OUTPUT_PATH = self.original

    def image(self, split: str, stem: str, size: tuple[int, int] = (8, 8)) -> None:
        Image.new("RGB", size).save(
            self.dataset / split / "images/crazing" / f"{stem}.jpg"
        )

    def xml(
        self,
        split: str,
        stem: str,
        *,
        filename: str | None = None,
        image_size: tuple[int, int] = (8, 8),
        box: tuple[int, int, int, int] = (0, 0, 8, 8),
        label: str = "crazing",
    ) -> None:
        name = filename if filename is not None else f"{stem}.jpg"
        xmin, ymin, xmax, ymax = box
        xml_text = f"""<annotation>
<filename>{name}</filename>
<size><width>{image_size[0]}</width><height>{image_size[1]}</height></size>
<object><name>{label}</name><bndbox>
<xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax>
</bndbox></object>
</annotation>"""
        (self.dataset / split / "annotations" / f"{stem}.xml").write_text(
            xml_text, encoding="utf-8"
        )

    def report(self) -> str:
        with contextlib.redirect_stdout(io.StringIO()):
            audit.main()
        return audit.OUTPUT_PATH.read_text(encoding="utf-8")

    def test_normal_pair_has_no_issues(self) -> None:
        self.image("train", "sample")
        self.xml("train", "sample")
        report = self.report()
        self.assertIn("train: 1 images, 1 XMLs, 1 objects", report)
        self.assertIn("Issues by type:\n  None", report)
        self.assertIn("Observations (not automatic errors):\n  None", report)

    def test_cross_split_counted_once_and_validated(self) -> None:
        self.image("train", "crossed")
        self.xml("validation", "crossed", filename="wrong.jpg", image_size=(9, 8), box=(0, 0, 9, 8))
        report = self.report()
        for finding in (
            "cross_split_annotation", "missing_annotation", "unmatched_annotation",
            "different_basename", "dimension_mismatch", "invalid_bbox_coordinates",
        ):
            self.assertIn(f"  {finding}: 1", report)
        self.assertNotIn("  unverifiable_annotation: 1", report)
        self.assertEqual(report.count("[cross_split_annotation]"), 1)

    def test_omitted_extension_is_observation_not_issue(self) -> None:
        self.image("train", "patches_104")
        self.xml("train", "patches_104", filename="patches_104")
        report = self.report()
        self.assertIn("  extension_omitted: 1", report)
        self.assertIn("Issues by type:\n  None", report)

    def test_wrong_explicit_extension_is_issue(self) -> None:
        self.image("train", "sample")
        self.xml("train", "sample", filename="sample.png")
        self.assertIn("  different_extension_or_case: 1", self.report())

    def test_invalid_bounding_box(self) -> None:
        self.image("train", "sample")
        self.xml("train", "sample", box=(0, 0, 9, 8))
        self.assertIn("  invalid_bbox_coordinates: 1", self.report())

    def test_missing_bounding_box_field(self) -> None:
        self.image("train", "sample")
        self.xml("train", "sample")
        p = self.dataset / "train/annotations/sample.xml"
        p.write_text(p.read_text().replace("<xmin>0</xmin>", ""))
        self.assertIn("  missing_bbox_xmin: 1", self.report())

    def test_no_image_is_unverifiable(self) -> None:
        self.xml("validation", "orphan")
        report = self.report()
        self.assertIn("  unmatched_annotation: 1", report)
        self.assertIn("  unverifiable_annotation: 1", report)

    def test_mixed_class_is_review_not_issue(self) -> None:
        self.image("train", "sample")
        self.xml("train", "sample", label="patches")
        report = self.report()
        self.assertIn("  label_folder_mismatch: 1", report)
        self.assertIn("Issues by type:\n  None", report)

    def test_issue_details_are_not_duplicated(self) -> None:
        self.image("train", "sample")
        self.xml("train", "sample", filename="wrong.jpg")
        report = self.report()
        details = report.split("Issue details:\n", 1)[1].split("Observation details:", 1)[0]
        self.assertEqual(details.count("[different_basename]"), 1)


if __name__ == "__main__":
    unittest.main()
