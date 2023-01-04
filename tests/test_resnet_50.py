import unittest

from tensorflow import keras

from models.resnet_50 import create_model


class TestResnet50(unittest.TestCase):
    def test_create_model(self):
        weights = "imagenet"
        input_shape = (128, 128, 3)
        dropout_rate = 0.1
        classes = 25
        model = create_model(
            weights=weights,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            classes=classes,
        )

        # Validate output model is a keras.Model
        self.assertTrue(
            isinstance(model, keras.Model),
            msg="Output model is not a keras.Model",
        )

        # Validate first layer is ok
        input_layer = model.layers[0]
        self.assertTrue(
            isinstance(input_layer, keras.layers.InputLayer),
            msg="Invalid type for model input layer",
        )
        self.assertEqual(
            input_layer.input_shape[0][1:],
            input_shape,
            msg="Input layer shape is invalid",
        )

        # No data augmentation should be applied
        # 4th layer must be resnet50 base model
        resnet50 = model.layers[3]
        self.assertEqual(resnet50.name, "resnet50")

        # Check output is applying GlobalAveragePooling2D
        self.assertTrue(
            isinstance(
                resnet50.layers[-1], keras.layers.GlobalAveragePooling2D
            ),
            msg="GlobalAveragePooling2D must used as Resnet50 model output",
        )
        # Check output shape is ok
        self.assertEqual(resnet50.output.shape.as_list(), [None, 2048])

        # Check Droupout layer is present in the model
        dropout_layer = model.layers[-2]
        self.assertTrue(
            isinstance(dropout_layer, keras.layers.Dropout),
            msg="Dropout layer not found or incorrectly placed",
        )
        #  Validate dropout rate is ok
        self.assertAlmostEqual(
            dropout_layer.rate,
            dropout_rate,
            places=4,
            msg="Dropout rate not being applied",
        )

        # Check full model output
        output_layer = model.layers[-1]
        self.assertTrue(
            isinstance(output_layer, keras.layers.Dense),
            msg="Output layer not found or incorrectly placed",
        )
        # Check output shape is ok
        self.assertEqual(
            output_layer.output.shape.as_list(),
            [None, classes],
            msg="Number of output model classes is incorrect",
        )
        # Check output activation function
        self.assertTrue(
            output_layer.activation == keras.activations.softmax,
            msg="Model output activation must be Softmax",
        )

    def test_create_model_data_aug(self):
        weights = "imagenet"
        input_shape = (128, 128, 3)
        dropout_rate = 0.5
        classes = 30
        data_aug_layer = {
            "random_flip": {
                "mode": "horizontal",
                "name": "test_random_flip_1",
            },
            "random_rotation": {
                "factor": 0.5,
                "name": "test_random_rotation_1",
            },
        }
        model = create_model(
            weights=weights,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            data_aug_layer=data_aug_layer,
            classes=classes,
        )

        # Get data augmentation layer, must be the second model layer
        aug_model_layer = model.layers[1]
        # Check is a keras.Sequential model
        self.assertTrue(
            isinstance(aug_model_layer, keras.Sequential),
            msg="Invalid type for model data augmentation layer",
        )

        # Check RandomFlip layer exits and parameters are ok
        random_flip_l = aug_model_layer.get_layer("test_random_flip_1")
        self.assertEqual(
            random_flip_l.mode,
            "horizontal",
            msg="Incorrect RandomFlip parameters",
        )

        # Check RandomRotation layer exits and parameters are ok
        random_flip_l = aug_model_layer.get_layer("test_random_rotation_1")
        self.assertAlmostEqual(
            random_flip_l.factor,
            0.5,
            places=4,
            msg="Incorrect RandomRotation parameters",
        )


if __name__ == "__main__":
    unittest.main()
