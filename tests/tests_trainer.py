import os
import unittest

import tensorflow as tf
from yaket.schema.schema import Accelerator
from yaket.trainer import Trainer


class CModel(tf.keras.Model):
    """Simple custom Model to test Trainer"""

    def __init__(self):
        super(CModel, self).__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)


class CLayer(tf.keras.layers.Layer):
    """Simple custom Layer to test Trainer"""

    def __init__(self):
        super(CLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)


class TestTrainer(unittest.TestCase):

    optimizers = Trainer.list_available_tf_modules("optimizers")
    losses = Trainer.list_available_tf_modules("losses")
    metrics = Trainer.list_available_tf_modules("metrics")

    simple_model = tf.keras.Sequential()
    simple_model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

    train_dataset = tf.data.Dataset.from_tensor_slices((range(10), range(10))).batch(1)
    dev_dataset = tf.data.Dataset.from_tensor_slices((range(10), range(10))).batch(1)

    simple_config = dict(
        optimizer="Adam",
        loss="BinaryCrossentropy",
        metrics=["Accuracy"],
        epochs=1,
        verbose=2,
        shuffle=True,
        accelerator="cpu",
    )

    def test_wrong_type_config_params(self):
        """Test inputs config params/file is valid"""

        params = [2, (3.0, 2), list([1, 2])]

        for param in params:
            with self.subTest(msg=param):
                with self.assertRaises(TypeError):
                    Trainer(
                        param,
                        self.simple_model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                    )

    def test_wrong_values_config_params(self):
        """Test whether a not-existing file has been passed or a wrong-type dictionary"""

        params = ["random_path.yaml", dict(a=1, b=2)]

        with self.subTest():
            with self.assertRaises(FileNotFoundError):
                Trainer(
                    params[0],
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                )
        with self.subTest():
            with self.assertRaises(ValueError):
                Trainer(
                    params[1],
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                )

    def test_wrong_keras_optimizer(self):
        """Test whether a wrong keras optimizer has been passed."""

        params = ["random_optimizer", "adam", "AdamW"]

        for param in params:
            with self.subTest(msg=param):
                with self.assertRaises(ValueError):
                    d = self.simple_config
                    d["optimizer"] = param
                    Trainer(
                        d,
                        self.simple_model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                    )

    def test_wrong_keras_loss(self):
        """Test whether a wrong keras loss has been passed."""

        params = ["random_loss", "binary_crossentropy", "mse"]

        for param in params:
            with self.subTest(msg=param):
                with self.assertRaises(ValueError):
                    d = self.simple_config
                    d["loss"] = param
                    Trainer(
                        d,
                        self.simple_model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                    )

    def test_wrong_keras_metric(self):
        """Test whether a wrong keras metric has been passed."""

        params = ["prc", "truepositive", "mse"]

        for param in params:
            with self.subTest(msg=param):
                with self.assertRaises(ValueError):
                    d = self.simple_config
                    d["metrics"] = param
                    Trainer(
                        d,
                        self.simple_model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                    )

    def test_import_custom_module(self):
        """Test whether a wrong custom module can be loaded."""

        custom_module = "custom_module.py"
        with self.assertRaises(FileNotFoundError):
            Trainer(
                self.simple_config,
                self.simple_model,
                train_dataset=self.train_dataset,
                val_dataset=self.dev_dataset,
                custom_modules_path=custom_module,
            )

    def test_accelerator(self):
        """Test whether accelerator config param is converted to right Enum member.
        Test only for CPU because it throws error if GPU is not found"""

        params = ["cpu"]
        acceleator_expected = [
            Accelerator.cpu,
            Accelerator.gpu,
            Accelerator.mgpu,
            Accelerator.tpu,
        ]
        for param, expected in zip(params, acceleator_expected):
            with self.subTest(msg=param):
                d = self.simple_config
                d["accelerator"] = param
                trainer = Trainer(
                    d,
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                )
                self.assertEqual(trainer._accelerator, expected)

    def test_default_strategy(self):
        """Test whether strategy is correctly initialized.

        TODO: tpu is not implemented yet"""

        params = ["cpu"]
        expected_strategies = [tf.distribute.MirroredStrategy]
        strategy = None

        if tf.config.list_physical_devices("GPU"):
            params.extend(["gpu", "mgpu"])
            expected_strategies.extend(
                [tf.distribute.OneDeviceStrategy, tf.distribute.MirroredStrategy]
            )

        for param, expected in zip(params, expected_strategies):
            with self.subTest(msg=param):
                d = self.simple_config
                d["accelerator"] = param
                trainer = Trainer(
                    d,
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                    strategy=strategy,
                )

                self.assertIsNone(trainer.strategy)

                strategy_out = trainer._get_strategy()
                self.assertIsInstance(strategy_out, expected)

    def test_compile_model(self):
        """Test whether model is compiled correctly."""

        params = ["cpu"]
        expected_strategies = [tf.distribute.MirroredStrategy]
        strategy = None

        if tf.config.list_physical_devices("GPU"):
            params.extend(["gpu", "mgpu"])
            expected_strategies.extend(
                [tf.distribute.OneDeviceStrategy, tf.distribute.MirroredStrategy]
            )

        for param, expected in zip(params, expected_strategies):
            with self.subTest(msg=param):
                d = self.simple_config
                d["accelerator"] = param
                trainer = Trainer(
                    d,
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                    strategy=strategy,
                )

                trainer._compile_model()
                self.assertTrue(trainer.model._is_compiled)

    def test_custom_model_clone_model(self):
        """Test whether custom model is not cloned ."""

        model = CModel()
        trainer = Trainer(
            self.simple_config,
            model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )
        with self.assertRaises(Exception):
            trainer._clone_model()

    def test_custom_layer_clone_model(self):
        """Test whether model with custom layer is not cloned correctly."""

        model = tf.keras.Sequential([CLayer()])
        trainer = Trainer(
            self.simple_config,
            model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )
        with self.assertRaises(Exception):
            trainer._clone_model()

    def test_weights_save_model(self):
        """Test whether vanilla model is saved with weights and it can be loaded back"""

        model = CModel()
        trainer = Trainer(
            self.simple_config,
            model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )

        trainer._save_model()

        self.assertTrue(os.path.exists(trainer._out_path + ".index"))
        trainer.model.load_weights(trainer._out_path)

    def test_savedmodel_save_model(self):
        """Test whether vanilla model is saved in SavedFormat and it can be loaded back"""

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )

        trainer._save_model()

        self.assertTrue(os.path.exists(trainer._out_path))
        tf.keras.models.load_model(trainer._out_path)


if __name__ == "__main__":

    unittest.main()
