import os

import platform
import unittest
import numpy as np
import tensorflow as tf
from yaket.schema.schema import Accelerator, TrainingModel
from yaket.trainer import Trainer
import psutil

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

    boring_trainer = Trainer(
        simple_config,
        simple_model,
        train_dataset=train_dataset,
        val_dataset=dev_dataset,
    )



    def test_create_new_trainer(self):
        """
        GIVEN a Trainer initialized with valid config params, model, and train/dev dataset
        WHEN a new Trainer instance is created
        THEN a new Trainer instance exists with same attributes
        """

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )

        assert trainer.model == self.simple_model
        assert trainer.train_dataset == self.train_dataset
        assert trainer.val_dataset == self.dev_dataset
        assert isinstance(trainer.config, TrainingModel)

    def test_train_history(self):
        """Test whether training is working.
        GIVEN a correct initialized Trainer
        WHEN train() is called
        THEN test the train history exists
        """

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )
        history = trainer.train()
        assert history is not None

    def test_train_saved(self):
        """Test whether training is working and model is saved
        GIVEN a correct initialized Trainer
        WHEN train() is called
        THEN test the model is saved
        """

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )
        trainer.train()
        assert os.path.exists(trainer._out_path)

    def test_convert_model_with_no_trained_model(self):
        """Test whether convert_model() is called without a trained model
        GIVEN a correct initialized Trainer
        WHEN convert_model() is called
        THEN an error is raised if the model has not been saved"""

        with self.assertRaises(Exception):
            self.boring_trainer.convert_model("onnx")

    def test_convert_model_wrong_format_model(self):
        """GIVEN a wrong format model
        WHEN convert_model() is called
        THEN an error is raised
        """

        format_list = [2,'random_string', (1,2), [1,2]]

        trainer = Trainer(
                    self.simple_config,
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                )
        trainer.train()

        for format_type in format_list:
            with self.subTest(msg=format_type):
                with self.assertRaises(Exception):
                    trainer.convert_model(format_type)

    def test_convert_model_wrong_opset_onnx(self):
        """GIVEN a wrong opset onnx
        WHEN convert_model() is called
        THEN an error is raised
        """

        opset_list = ['random_string', -1, [1,2]]

        trainer = Trainer(
                    self.simple_config,
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                )
        trainer.train()

        for opset in opset_list:
            with self.subTest(msg=opset):
                with self.assertRaises(TypeError):
                    trainer.convert_model(opset_onnx = opset)

    


    def test_init_wrong_model_format(self):
        """GIVEN a wrong format model
        WHEN Trainer is initialized
        THEN an error is raised
        """

        model_list = [2,'random_string', (1,2), [1,2]]

        for model in model_list:
            with self.subTest(msg=model):
                with self.assertRaises(Exception):
                    Trainer(
                        self.simple_config,
                        model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                    )

    def test_init_long_tuple_train_dataset(self):
        """GIVEN a train dataset with more than 3 elements
        WHEN Trainer is initialized
        THEN an error is raised
        """

        train_dataset = tf.data.Dataset.from_tensor_slices((range(10), range(10), range(10), range(10))).batch(1)

        with self.assertRaises(Exception):
            Trainer(
                self.simple_config,
                self.simple_model,
                train_dataset=train_dataset,
                val_dataset=self.dev_dataset,
            )
        
    
    def test_init_list_train_dataset(self):
        """GIVEN a train dataset with a list
        WHEN Trainer is initialized
        THEN an error is raised
        """

        train_dataset = [range(10), range(10)]

        with self.assertRaises(Exception):
            Trainer(
                self.simple_config,
                self.simple_model,
                train_dataset=train_dataset,
                val_dataset=self.dev_dataset,
            )
    def test_init_list_dev_dataset(self):
        """GIVEN a dev dataset with a list
        WHEN Trainer is initialized
        THEN an error is raised
        """

        dev_dataset = 'string'

        
        with self.assertRaises(TypeError):
            Trainer(
                self.simple_config,
                self.simple_model,
                train_dataset=self.train_dataset,
                val_dataset=dev_dataset,
            )

    def test_init_wrong_strategy_format(self):
        """GIVEN a wrong format strategy
        WHEN Trainer is initialized
        THEN an error is raised
        """

        strategy_list = [2,'random_string', (1,2), [1,2]]

        for strategy in strategy_list:
            with self.subTest(msg=strategy):
                with self.assertRaises(Exception):
                    Trainer(
                        self.simple_config,
                        self.simple_model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                        strategy=strategy
                    )
            
    def test_init_validate_yaml_wrong_format(self):
        """GIVEN a wrong format yaml file
        WHEN Trainer is initialized
        THEN an error is raised
        """

        validate_yaml_list = [2,'random_string', (1,2), [1,2]]

        for yaml in validate_yaml_list:
            with self.subTest(msg=yaml):
                with self.assertRaises(Exception):
                    Trainer(
                        self.simple_config,
                        self.simple_model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                        validate_yaml=yaml
                    )
    def test_init_random_seed_wrong_format(self):
        """GIVEN a wrong format random seed
        WHEN Trainer is initialized
        THEN an error is raised
        """

        random_seed_list = ['random_string', (1,2), [1,2]]

        for random_seed in random_seed_list:
            with self.subTest(msg=random_seed):
                with self.assertRaises(Exception):
                    Trainer(
                        self.simple_config,
                        self.simple_model,
                        train_dataset=self.train_dataset,
                        val_dataset=self.dev_dataset,
                        random_seed=random_seed
                    )

    

    def test_convert_model_wrong_output_path(self):
        """GIVEN a wrong output path
        WHEN convert_model() is called
        THEN an error is raised
        """

        output_path_list = [2, 1.2, [1,2]]

        trainer = Trainer(
                    self.simple_config,
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                )
        trainer.train()

        for output_path in output_path_list:
            with self.subTest(msg=output_path):
                with self.assertRaises(TypeError):
                    trainer.convert_model(output_path = output_path)
    
    def test_convert_model_wrong_from_command_line(self):
        """GIVEN a wrong from command line
        WHEN convert_model() is called
        THEN an error is raised
        """

        from_command_line_list = [2, 1.2, [1,2],'random_string']

        trainer = Trainer(
                    self.simple_config,
                    self.simple_model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.dev_dataset,
                )
        trainer.train()

        for from_command_line in from_command_line_list:
            with self.subTest(msg=from_command_line):
                with self.assertRaises(TypeError):
                    trainer.convert_model(from_command_line = from_command_line)

    def test_get_sample_weight_mode_none(self):
        """Test whether get_sample_weight_mode() is called
        GIVEN a correct initialized Trainer without sample_weight_mode
        WHEN get_sample_weight_mode() is called
        THEN test the sample weight mode is None
        """

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )
        sample_weight_mode = trainer._get_sample_weight_mode()
        self.assertIsNone(sample_weight_mode)
    
    def test_get_sample_weight_mode_temporal(self):
        """GIVEN a correct initialized Trainer with sample_weight_mode = 'temporal'
        WHEN get_sample_weight_mode() is called
        THEN test the sample weight mode is 'temporal'"""
        d = self.simple_config
        d['sample_weight_mode'] = 'temporal'
        trainer = Trainer(d,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,)
        sample_weight_mode = trainer._get_sample_weight_mode()
        self.assertEqual(sample_weight_mode, 'temporal')

    def test_get_input_shape_tf_data(self):
        """GIVEN a correct initialized Trainer with tf.data.Dataset
        WHEN get_input_shape() is called
        THEN test the input shape is correct
        """


        train_dataset_numpy= (np.random.random((2,10)), np.random.random((2,10)))
        dev_dataset_numpy = (np.random.random((2,10)), np.random.random((2,10)))

        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset_numpy).batch(1)
        dev_dataset = tf.data.Dataset.from_tensor_slices(dev_dataset_numpy).batch(1)

        expected_input_shape = (None, 10)

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=train_dataset,
            val_dataset=dev_dataset,
        )
        input_shape = trainer._get_input_shape()
        np.testing.assert_array_equal(input_shape, expected_input_shape)

    def test_get_x_y_train_output_type(self):
        """GIVEN a correct initialized Trainer with tf.data.Dataset or numpy array
        WHEN get_x_y_val() is called
        THEN test the output type is tf.data.Dataset
        """

        train_dataset_numpy= (np.random.random((2,10)), np.random.random((2,10)))
        

        trainer_numpy = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=train_dataset_numpy,
        )
        trainer_tf = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
        )

        x_np = trainer_numpy._get_x_y_train(1)
        x_tf = trainer_tf._get_x_y_train(1)
        self.assertIsInstance(x_np, tf.data.Dataset)
        self.assertIsInstance(x_tf, tf.data.Dataset)

    def test_get_x_y_val_output_type(self):
        """GIVEN a correct initialized Trainer with tf.data.Dataset, numpy array, or None
        WHEN get_x_y_val() is called
        THEN test the output type is tf.data.Dataset or None if val_dataset is None
        """

        dev_dataset_numpy= (np.random.random((2,10)), np.random.random((2,10)))
        

        trainer_numpy = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=dev_dataset_numpy,
        )
        trainer_tf = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )
        trainer_none = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
        )


        x_np = trainer_numpy._get_x_y_val(1)
        x_tf = trainer_tf._get_x_y_val(1)
        x_none = trainer_none._get_x_y_val(1)
        self.assertIsInstance(x_np, tf.data.Dataset)
        self.assertIsInstance(x_tf, tf.data.Dataset)
        self.assertIsNone(x_none)

    def test_validate_config(self):
        """GIVEN a correct initialized Trainer
        WHEN a config parameter is changed with a wrong type and validate_config() is called
        THEN test the config is validated by throwing an error
        """

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )

        trainer.config.epochs = 'wrong_type'
        with self.assertRaises(Exception):
            trainer.validate_config()
    
    def test_validate_config_no_error(self):
        """GIVEN a correct initialized Trainer
        WHEN a config parameter is changed with the correct type and validate_config() is called
        THEN test the config is validated without throwing an error and the new value is set
        """

        new_value_config_epochs = 10

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
            val_dataset=self.dev_dataset,
        )

        trainer.config.epochs = 10

        trainer.validate_config()
        self.assertEqual(trainer.config.epochs, new_value_config_epochs)
    

    
    def test_get_input_shape_numpy_data(self):
        """GIVEN a correct initialized Trainer with numpy array
        WHEN get_input_shape() is called
        THEN test the input shape is correct
        """

        train_dataset_numpy= (np.random.random((2,10)), np.random.random((2,10)))
        dev_dataset_numpy = (np.random.random((2,10)), np.random.random((2,10)))

        expected_input_shape = (None, 10)

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=train_dataset_numpy,
            val_dataset=dev_dataset_numpy,
        )
        input_shape = trainer._get_input_shape()
        np.testing.assert_array_equal(input_shape, expected_input_shape)

    def test_get_free_gpu_idx_linux_no_gpu(self):
        """GIVEN a Trainer class
        WHEN get_free_gpu_idx() is called on a linux machine without GPU 
        THEN test an error is thrown
        """
        with self.assertRaises(Exception):
            Trainer.get_free_gpu_idx()
    
    def test_get_free_gpu_idx_gpu(self):
        """GIVEN a Trainer class
        WHEN get_free_gpu_idx() is called on a linux machine with GPU 
        THEN test the first GPU is returned
        """
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            self.assertEqual(Trainer.get_free_gpu_idx(), 0)
        
    def test_get_free_gpu_idx_windows(self):
        """GIVEN a Trainer class
        WHEN get_free_gpu_idx() is called on a windows machine 
        THEN test an error is thrown
        """
        if platform.system == 'Windows':
            with self.assertRaises(Exception):
                Trainer.get_free_gpu_idx()

    def test_list_available_tf_modules_type_error(self):
        """GIVEN a Trainer class
        WHEN list_available_tf_modules() is called with a wrong type
        THEN an error is thrown
        """

        random_options = ['callabacks','tf_addons','netric']
        for option in random_options:
            with self.assertRaises(AssertionError):
                Trainer.list_available_tf_modules(option)
    
        
    
    def test_set_randomness_seed_wrong(self):
        """GIVEN a Trainer class
        WHEN _set_randomness() is called with a wrong seed
        THEN test an error is thrown
        """
        seeds = ['wrong_type', -1,[1,2]]
        for seed in seeds:
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    self.boring_trainer._set_randomness(seed)

    def test_clear_ram(self):
        """GIVEN a Trainer class correctly initialized and trained
        WHEN clear_ram() is called
        THEN test the RAM is cleared
        """

        trainer = Trainer(
            self.simple_config,
            self.simple_model,
            train_dataset=self.train_dataset,
        )
        
        history = trainer.train(100)

        ram_before_clearing = psutil.virtual_memory().percent

        trainer.clear_ram()
        ram_after_clearing = psutil.virtual_memory().percent

        self.assertGreaterEqual(ram_before_clearing, ram_after_clearing)

    def test_summary_model_no_initialized(self):
        """GIVEN a Trainer class
        WHEN summary_model() is called before the model is initialized
        THEN test an error is thrown
        """
        with self.assertRaises(Exception):
            Trainer.summary_model()


    def test_wrong_type_config_params(self):
        """GIVEN a Trainer class 
        WHEN it is initialized  with a wrong param_config type
        THEN test an error is thrown
        """

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
        """Test whether a not-existing file has been passed or a wrong-type dictionary
        GIVEN a Trainer class
        WHEN it is initialized with a wrong param_config value
        THEN test an error is thrown
        """

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
        """Test whether a wrong keras optimizer has been passed.
        GIVEN a Trainer class
        WHEN it is initialized with a wrong keras optimizer
        THEN test an error is thrown
        """

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
        """Test whether a wrong keras loss has been passed.
        GIVEN a Trainer class
        WHEN it is initialized with a wrong keras loss
        THEN test an error is thrown
        """

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
        """Test whether a wrong keras metric has been passed.
        GIVEN a Trainer class
        WHEN it is initialized with a wrong keras metric
        THEN test an error is thrown
        """

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
        """Test whether a wrong custom module can be loaded.
        GIVEN a Trainer class
        WHEN it is initialized with a wrong custom module
        THEN test an error is thrown
        """

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
        Test only for CPU because it throws error if GPU is not found
        GIVEN a Trainer class
        WHEN it is initialized with a wrong accelerator
        THEN test an error is thrown"""

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
        GIVEN a Trainer class
        WHEN it is initialized with a wrong accelerator
        THEN test an error is thrown

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
        """Test whether model is compiled correctly.
        GIVEN a correctly initialized Trainer class
        WHEN it _compile_model is called
        THEN test the model is compiled

        """

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
        """Test whether custom model is not cloned.
        GIVEN a correctly initialized Trainer class with a custom model
        WHEN it _clone_model is called
        THEN test the model is not cloned and error is thrown
        """

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
        """Test whether model with custom layer is not cloned correctly.
        GIVEN a correctly initialized Trainer class with a custom layer
        WHEN it _clone_model is called
        THEN test the model is not cloned and error is thrown"""

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
        """Test whether vanilla model is saved with weights and it can be loaded back
        
        GIVEN a correctly initialized Trainer class with a vanilla model
        WHEN it _save_model is called
        THEN test the model is saved with weights and it can be loaded back
        """

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
        """Test whether vanilla model is saved in SavedFormat and it can be loaded back
       
        GIVEN a correctly initialized Trainer class with a vanilla model
        WHEN it _save_model is called
        THEN test the model is saved in SavedFormat and it can be loaded back

        """

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
