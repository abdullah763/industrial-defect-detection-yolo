import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.convert_voc_to_yolo import convert_annotation, voc_to_yolo


class ConverterTests(unittest.TestCase):

    def test_normalization(self):
        result = voc_to_yolo(40, 50, 120, 150, 200, 200)
        expected = (0.4, 0.5, 0.4, 0.5)

        for actual, wanted in zip(result, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_invalid_box(self):
        with self.assertRaises(ValueError):
            voc_to_yolo(150, 50, 40, 120, 200, 200)

    def test_boundary_box(self):
        result = voc_to_yolo(78, 169, 118, 200, 200, 200)
        self.assertAlmostEqual(result[3], 0.155)

    def test_multiple_classes(self):
        xml = """
        <annotation>
            <size><width>200</width><height>200</height></size>
            <object>
                <name>crazing</name>
                <bndbox>
                    <xmin>4</xmin><ymin>4</ymin>
                    <xmax>110</xmax><ymax>136</ymax>
                </bndbox>
            </object>
            <object>
                <name>patches</name>
                <bndbox>
                    <xmin>78</xmin><ymin>169</ymin>
                    <xmax>118</xmax><ymax>200</ymax>
                </bndbox>
            </object>
        </annotation>
        """

        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.xml"
            path.write_text(xml, encoding="utf-8")
            lines = convert_annotation(path)

        self.assertEqual(len(lines), 2)
        self.assertEqual(
            lines[0],
            "0 0.285000 0.350000 0.530000 0.660000"
        )
        self.assertEqual(
            lines[1],
            "2 0.490000 0.922500 0.200000 0.155000"
        )


if __name__ == "__main__":
    unittest.main()