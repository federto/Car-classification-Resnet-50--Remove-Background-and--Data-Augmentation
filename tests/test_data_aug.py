import unittest

from tensorflow import keras

from utils.data_aug import create_data_aug_layer


class TestDataAug(unittest.TestCase):
    def test_empty_config(self):
        data_aug_layer = {}
        layers = create_data_aug_layer(data_aug_layer)
        # Check is a keras.Sequential model
        self.assertTrue(
            isinstance(layers, keras.Sequential),
            msg="Invalid type for model data augmentation layer",
        )

        # Check it has no layers
        self.assertEqual(len(layers.layers), 0)

    def test_full_config(self):
        data_aug_layer = {
            "random_flip": {
                "mode": "horizontal",
                "name": "test_random_flip_123",
            },
            "random_rotation": {
                "factor": 0.5,
                "name": "test_random_rotation_123",
            },
            "random_zoom": {
                "height_factor": 0.25,
                "width_factor": 0.3,
                "name": "test_random_zoom_123",
            },
        }
        layers = create_data_aug_layer(data_aug_layer)

        # Check is a keras.Sequential model
        self.assertTrue(
            isinstance(layers, keras.Sequential),
            msg="Invalid type for model data augmentation layer",
        )

        # Check number of layers created is ok
        self.assertEqual(len(layers.layers), 3)

        # Check RandomFlip layer exits and parameters are ok
        random_flip_l = layers.get_layer("test_random_flip_123")
        self.assertEqual(
            random_flip_l.mode,
            "horizontal",
            msg="Incorrect RandomFlip parameters",
        )

        # Check RandomRotation layer exits and parameters are ok
        random_flip_l = layers.get_layer("test_random_rotation_123")
        self.assertAlmostEqual(
            random_flip_l.factor,
            0.5,
            places=4,
            msg="Incorrect RandomRotation parameters",
        )

        # Check RandomZoom layer exits and parameters are ok
        random_zoom_l = layers.get_layer("test_random_zoom_123")
        self.assertAlmostEqual(
            random_zoom_l.height_factor,
            0.25,
            places=4,
            msg="Incorrect RandomZoom 'height_factor' parameter",
        )
        self.assertAlmostEqual(
            random_zoom_l.width_factor,
            0.3,
            places=4,
            msg="Incorrect RandomZoom 'width_factor' parameter",
        )
