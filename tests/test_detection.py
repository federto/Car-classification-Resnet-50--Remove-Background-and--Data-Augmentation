import unittest

import cv2

from utils.detection import get_vehicle_coordinates


class TestDataAug(unittest.TestCase):
    def test_get_vehicle_coordinates(self):
        # Test no vehicle is detected
        im = cv2.imread("tests/test_data/cat.jpeg")
        h, w = im.shape[:2]
        det = get_vehicle_coordinates(im)
        self.assertEqual(det, [0, 0, w, h])

        # Test car is detected
        im = cv2.imread("tests/test_data/012310.jpg")
        x1, y1, x2, y2 = get_vehicle_coordinates(im)
        self.assertAlmostEqual(x1, 72, delta=5)
        self.assertAlmostEqual(y1, 106, delta=5)
        self.assertAlmostEqual(x2, 572, delta=5)
        self.assertAlmostEqual(y2, 359, delta=5)

        # Test truck is detected
        im = cv2.imread("tests/test_data/005652.jpg")
        x1, y1, x2, y2 = get_vehicle_coordinates(im)
        self.assertAlmostEqual(x1, 7, delta=5)
        self.assertAlmostEqual(y1, 21, delta=5)
        self.assertAlmostEqual(x2, 234, delta=5)
        self.assertAlmostEqual(y2, 155, delta=5)

        # Test multiple objects
        im = cv2.imread("tests/test_data/008773.jpg")
        x1, y1, x2, y2 = get_vehicle_coordinates(im)
        self.assertAlmostEqual(x1, 110, delta=5)
        self.assertAlmostEqual(y1, 181, delta=5)
        self.assertAlmostEqual(x2, 543, delta=5)
        self.assertAlmostEqual(y2, 408, delta=5)
