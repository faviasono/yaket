from dataclasses import dataclass

from typing import List, Optional, Tuple, Union, Any, Dict, Callable
import numpy as np
import tensorflow as tf
import gc
from yaket.schema.schema import TrainingModel, yaml_to_pydantic, Accelerator
from yaket.converter.converter import Converter
import importlib
import mlflow
import os
import time
import subprocess as sp
import sys


@dataclass
class Trainer:
    config_path: str
    model: tf.keras.Model
    train_dataset: Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset]
    val_dataset: Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset] = None
    strategy: Optional[tf.distribute.Strategy] = None
    random_seed: int = 1234
    validate_yaml: bool = True
    custom_modules_path: Optional[str] = None

    # internals
    _config: TrainingModel = None
    _input_shape: Tuple[int, ...] = None
    _metrics = List[Union[tf.keras.metrics.Metric, Callable]]
    _callbacks: List[tf.keras.callbacks.Callback] = None
    _optimizer: tf.keras.optimizers.Optimizer = None
    _loss: Union[tf.keras.losses.Loss, Callable] = None
    _custom_module: Callable = None
    _history: Dict[str, Any] = None
    _model_checkpoint: Optional[str] = None
    _accelerator: Optional[Accelerator] = None
    _log: bool = False
    _out_path: str = None

    def _init_trainer(self) -> None:
        """Initialize the trainer"""

        if not isinstance(self.model, tf.keras.models.Model):
            raise Exception("model must be keras model")
        if not isinstance(self.config_path, str) or not os.path.isfile(
            self.config_path
        ):
            raise Exception("Config path must be a valid file path")
        if self.strategy is not None and not isinstance(
            self.strategy, tf.distribute.Strategy
        ):
            raise Exception("Strategy must be keras strategy object")
        if not isinstance(self.random_seed, int):
            raise Exception("Random seed must be an integer")
        if not isinstance(self.validate_yaml, bool):
            raise Exception("Validate yaml must be a boolean value")
        if self.custom_modules_path is not None:
            if not isinstance(self.custom_modules_path, str) or not os.path.isfile(
                self.custom_modules_path
            ):
                raise Exception("Costum modules path must be a valid path string")

        if self.custom_modules_path:
            self._import_custom_model(self.custom_modules_path)
        self._config = self._parse_config()
        self._accelerator = Accelerator[self.config.accelerator]
        self._validate_config_file()

        self._callbacks = self._get_callbacks()
        self._input_shape = self._get_input_shape()

    def _train(self, train_dataset, val_dataset, epochs: int = None):
        """Train the model"""

        self._compile_model()
        history = self.model.fit(
            x=train_dataset,
            y=None,
            epochs=int(self.config.epochs) if epochs is None else epochs,
            validation_data=val_dataset,
            batch_size=None,
            callbacks=self._callbacks,
            steps_per_epoch=int(self.config.steps_per_epoch) if self.config.steps_per_epoch is not None else None,
            class_weight=None,  # TODO: add class_weight,
            verbose=int(self.config.verbose),
        )
        return history

    def train(self, epochs: int = None):
        """Train the model. Main function to call.

        Args
        ----
        epochs: int, optional
            Number of epochs to train. If None, will train with the number of epochs specified in the config file.

        """

        self._init_trainer()
        self._autolog()

        if self.strategy is None and self._accelerator is Accelerator.cpu:
            train_dataset = self._get_x_y_train(self.config.batch_size)
            val_dataset = self._get_x_y_val(self.config.batch_size)
            history = self._train(train_dataset, val_dataset, epochs)
        else:
            strategy = self._get_strategy()
            batch_size = strategy.num_replicas_in_sync * self.config.batch_size
            train_dataset = self._get_x_y_train(batch_size)
            val_dataset = self._get_x_y_val(batch_size)
            with strategy.scope():
                history = self._train(train_dataset, val_dataset, epochs)

        self._save_model()
        self._history = history.history
        self._clean_workspace()
        return history

    def convert_model(
        self,
        format_model: str = "onnx",
        opset_onnx: int = 15,
        output_path: str = "model",
        from_command_line: bool = True,
    ):

        """Convert the model to a different format. Available formats:
        1. ONNX
        2. TensorFlow Lite

        Args
        ----
        format: str
            Format to convert the model to. Available formats: onnx, tflite
        model_path: str
            Path to the model to convert.
        output_path: str
            Path to the output file.
        from_command_line: bool
            Whether or not to convert the model using the command line.
            It might be not available with some platforms.

        """
        if from_command_line and self._out_path is None:
            raise Exception("You need to have a saved model before converting using command line")
        model_to_convert = self.model if not from_command_line else None
        converter = Converter(
            out_format=format_model.lower(),
            opset_onnx=opset_onnx,
            model_path=self._out_path,
            model=model_to_convert,
            out_path = output_path,
        )
        if converter.convert():
            print(f"Successfully converted to format {format} ")

    def _get_input_shape(self):
        """Get the input shape of input dataset"""
        if isinstance(self.train_dataset, tf.data.Dataset):
            for x, y in self.train_dataset.take(1):
                self.input_shape = (None, *x.shape[1:])
        else:
            self.input_shape = (None, *self.train_dataset[0].shape[1:])
        return self.input_shape

    def _save_model(self):
        """Save the model by loading best checkpoint if available and saving it to mlflow or local path"""
        if self._model_checkpoint is not None:
            self.model.load_weights(self._model_checkpoint)
            if self._log:
                self.model.save("/tmp/best_model")
                run = mlflow.last_active_run()
                idx = 7  # TODO: check is always the same
                artifact_path = run.info.artifact_uri[idx:]
                self._out_path = artifact_path + f"/best_model"
                self.model.save(self._out_path)
            else:
                os.makedirs(os.getcwd() + "/models", exist_ok=True)
                t = int(time.time())
                self._out_path = os.getcwd() + f"/models/{t}_best_model"
                self.model.save(self._out_path)
        else:
            os.makedirs(os.getcwd() + "/models", exist_ok=True)
            t = int(time.time())
            self._out_path = os.getcwd() + f"/models/{t}_best_model"
            self.model.save(self._out_path)

    def _clone_model(self):
        """Clone the model so that it works within tf.distribute.Strategy
        It works only for models not using custom objects
        """
        self.model = tf.keras.models.clone_model(self.model)

    def _get_x_y_val(self, batch_size):
        """Get the x and y for training based on the format of the dataset"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )

        if self.val_dataset is None:
            return None
        if isinstance(self.val_dataset, tf.data.Dataset):
            return self.val_dataset.with_options(options)
        else:
            val = (
                tf.data.Dataset.from_tensor_slices(self.val_dataset)
                .batch(batch_size)
                .with_options(options)
            )
            return val

    def _get_x_y_train(self, batch_size):
        """Get the x and y for training based on the format of the dataset"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )

        if isinstance(self.train_dataset, tf.data.Dataset):
            x = self.train_dataset.with_options(options)
        else:
            x = tf.data.Dataset.from_tensor_slices(self.train_dataset)
            if self.config.shuffle:
                x = x.shuffle(self.train_dataset[0].shape[0])
            x = x.batch(batch_size).prefetch(1).with_options(options)
        return x

    @property
    def config(self):
        return self._config

    def _compile_model(self) -> None:
        """Compile the model"""

        self._optimizer = self._get_optimizer()
        self._loss = self._get_loss()
        self._metrics = self._get_metrics()

        self.model.compile(
            optimizer=self._optimizer, loss=self._loss, metrics=self._get_metrics()
        )

    def _get_strategy(self):
        if self.strategy is None:
            if self._accelerator is None:
                return tf.distribute.MirroredStrategy()
            if self._accelerator is Accelerator.gpu:
                index = Trainer.get_free_gpu_idx()
                return tf.distribute.OneDeviceStrategy(f"/gpu:{index}")
            if (
                self._accelerator is Accelerator.cpu
                or self._accelerator is Accelerator.mgpu
            ):
                # If GPUs are not available, it will use CPUs
                return tf.distribute.MirroredStrategy()
            if self._accelerator is Accelerator.tpu:
                # TODO: check configuration for tpu strategy
                return tf.distribute.TPUStrategy()
        else:
            return self.strategy

    @staticmethod
    def get_free_gpu_idx():
        """Get the index of the freer GPU"""
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = (
            sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
        )
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return int(np.argmin(memory_free_values))

    def _parse_config(self) -> Any:
        return yaml_to_pydantic(self.config_path, self.validate_yaml)

    def _validate_config_file(self):
        "Validate existence of the loss, optimizer and callbacks defined in the config file"
        try:
            self._get_optimizer()
            self._get_metrics()
            self._get_loss()
        except Exception as e:
            raise TypeError(
                f"You are using a module not defined in either keras or in the custom script\n Details: {e}"
            )

    def _import_custom_model(self, module_name: str):
        try:
            custom_dirpath = os.path.dirname(module_name)
            sys.path.append(custom_dirpath)
            module = module_name.split("/")[-1].split(".")[0]
            self._custom_module = importlib.import_module(module)
        except Exception as e:
            raise ImportError(f"Error importing {module_name}: {e}")

    def _load_custom_module(
        self, module_name: str, params: Optional[Dict] = None
    ) -> Callable:
        try:
            if params is None:
                return getattr(self._custom_module, module_name)
            else:
                return getattr(self._custom_module, module_name)(**params)
        except Exception as e:
            raise ImportError(f"Error importing {module_name}: It does not exist")

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Get the optimizer from the config file"""
        default_value = "not_found"
        opt_pars = dict()
        opt = self.config.optimizer
        if isinstance(opt, List):
            if len(opt) == 1 and isinstance(opt[0], Dict):
                k = list(opt[1].keys())[0]
                v = list(opt[1].values())[0]
                scheduler = getattr(tf.keras.optimizers, k, default_value)
                if optimizer != default_value:
                    return optimizer(**opt_pars)
                else:
                    return self._load_custom_module(optimizer, opt_pars)
            else:
                opt_name = opt[0]
                optimizer = getattr(
                    tf.keras.optimizers, f"{opt_name}", default_value
                )
                if optimizer == default_value:
                    optimizer = self._load_custom_module(optimizer)
                if isinstance(opt[1], dict):
                    k = list(opt[1].keys())[0]
                    v = list(opt[1].values())[0]
                    scheduler = getattr(tf.keras.optimizers.schedules, k, default_value)
                    if scheduler == default_value:
                        raise ValueError(f"{v} is not a valid scheduler. Only available from keras")
                    else:
                        opt_pars['learning_rate'] = scheduler(**v)
                        return optimizer(**opt_pars)
        elif isinstance(opt, str):
            optimizer = getattr(
            tf.keras.optimizers, f"{self.config.optimizer}", default_value
            )
            if optimizer != default_value:
                return optimizer(**opt_pars)
            else:
                    return self._load_custom_module(optimizer, opt_pars)

    def _get_loss(self) -> Union[tf.keras.losses.Loss, Callable]:
        """Get the loss from the config file"""

        loss_config = self.config.loss
        default_value = "not_found"

        if isinstance(loss_config, str):
            loss = getattr(tf.keras.losses, loss_config, default_value)
            if loss != default_value:
                return loss()
            else:  # it's a custom loss
                return self._load_custom_module(loss_config)
        elif isinstance(loss_config, Dict):
            loss_name = list(loss_config.keys())[0]
            loss_params = list(loss_config.values())[0]
            loss = getattr(tf.keras.losses, loss_name, default_value)
            if loss != default_value:
                return loss(**loss_params)
            else:
                return self._load_custom_module(loss_config, loss_params)


    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Get the callbacks from the config file"""
        if self.config.callbacks is None:
            return None
        callbacks = []
        default_value = "not_found"

        for name_callback in self.config.callbacks:
            if isinstance(name_callback, Dict):
                key = list(name_callback.keys())[0]
                args = list(name_callback.values())[0]

                # Track filepath if it's a ModelCheckpoint
                self._model_checkpoint = (
                    args["filepath"] if key == "ModelCheckpoint" else None
                )

                callback_value = getattr(tf.keras.callbacks, key, default_value)
            else:
                callback_value = getattr(
                    tf.keras.callbacks, name_callback, default_value
                )
                args = None

            if callback_value != default_value:
                callbacks.append(
                    callback_value(**args) if args is not None else callback_value()
                )
            else:
                if args is not None:
                    if args.get("dataset") is not None:
                        dataset = (
                            self.val_dataset
                            if args["dataset"] in ["val", "dev", "validation"]
                            else self.train_dataset
                        )
                        args["dataset"] = dataset
                    callbacks.append(self._load_custom_module(key, args))
                else:
                    callbacks.append(self._load_custom_module(name_callback, args))

        return callbacks

    def _get_metrics(self) -> List[Union[tf.keras.metrics.Metric, Callable]]:
        """Get the metrics"""
        if self.config.metrics is None:
            return None

        list_metrics = []
        default_value = "not_found"
        for metric in self.config.metrics:
            if metric is None:
                continue
            if isinstance(metric, str):
                args = None
                metric_value = getattr(tf.keras.metrics, f"{metric}", default_value)()
            else:
                m, args = list(metric.items())[0]
                metric_value = getattr(tf.keras.metrics, f"{m}", default_value)(**args)

            if metric_value != default_value:
                list_metrics.append(metric_value)
            else:
                list_metrics.append(self._load_custom_module(metric, args))
        return list_metrics

    @staticmethod
    def list_available_tf_modules(option: str):
        """List available optimizers, losses, and metrics in tf.keras"""
        options_func = {
            "optimizers": tf.keras.optimizers,
            "losses": tf.keras.losses,
            "metrics": tf.keras.metrics,
        }
        assert option in list(options_func.keys())
        modules = [value for value in dir(options_func[option]) if value[0].isupper()]
        return modules

    def _clean_workspace(self) -> None:
        """Clean the workspace"""
        tf.keras.backend.clear_session()

    def _autolog(self) -> None:
        """Autolog the model using MLFlow"""
        if self.config.autolog:
            mlflow.tensorflow.autolog(log_models=True, disable=False)
            self._log = True

    def _set_randomness(self, random_seed: Optional[int] = None) -> None:
        """Set the randomness"""
        if random_seed is not None:
            if tf.__version__ >= "2.9.0":
                tf.keras.set_random_seed(random_seed)
                tf.config.experimental.enable_op_determinism()
            else:
                tf.random.set_seed(random_seed)
                np.random.seed(random_seed)

    def clear_ram(self):
        "Delete model and all datasets saved in the Trainer class"
        del self.model
        del self.train_dataset
        del self.val_dataset
        del self.test_dataset
        gc.collect()

    def summary_model(self):
        """Summary of the model"""
        self.model.summary()


if __name__ == "__main__":
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

    model.summary()

    path = "/root/project/yaket/examples/files/trainer.yaml"

    trainer = Trainer(
        config_path=path,
        train_dataset=(x_train, y_train),
        val_dataset=(x_test, y_test),
        model=model,
        strategy=strategy,
    )
    trainer.train()
