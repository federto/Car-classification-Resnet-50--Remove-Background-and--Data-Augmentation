import unittest

from utils.utils import load_config


class TestUtils(unittest.TestCase):
    def test_load_config(self):
        config = {
            "seed": 123,
            "data": {
                "directory": "/bla/data",
                "label_mode": "categorical",
                "validation_split": 0.2,
                "image_size": [224, 224],
                "batch_size": 32,
            },
            "model": {
                "weights": "imagenet",
                "input_shape": [224, 224, 3],
                "classes": 56,
                "droput_rate": 0.8,
                "data_aug_layer": {
                    "random_flip": {"mode": "horizontal"},
                    "random_rotation": {"factor": 0.1},
                },
            },
            "compile": {
                "optimizer": {"adam": {"learning_rate": 0.1}},
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"],
            },
            "fit": {
                "epochs": 100,
                "callbacks": {
                    "model_checkpoint": {
                        "filepath": "/bla/model",
                        "save_best_only": True,
                    },
                    "tensor_board": {"log_dir": "/bla/log"},
                },
            },
        }

        loaded_config = load_config("tests/test_data/config_test.yml")

        self.assertDictEqual(loaded_config, config)
