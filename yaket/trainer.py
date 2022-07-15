from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any, Dict, Callable
import numpy as np
import tensorflow as tf
import gc
from schema.schema import TrainingModel, yaml_to_pydantic
import importlib
import mlflow


@dataclass
class Trainer:
    config_path: str
    model: tf.keras.Model
    train_dataset: Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset]
    dev_dataset: Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset]
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

    def _init_trainer(self) -> None:
        """Initialize the trainer
        TODO: Add checks + exceptions"""
        self._config = self._parse_config()
        self._validate_config_file()
        if self.custom_modules_path:
            self._import_custom_model(self.custom_modules_path)
        self._callbacks = self._get_callbacks()
        self._optimizer = self._get_optimizer()
        self._loss = self._get_loss()
        self._metrics = self._get_metrics()
        self._input_shape = self._get_input_shape()

    def train(self):
        """Train the model. Main function to call
        
        TODO: 
        1. Add save_model option based on checkpoint 
        2. Add tf.distribute.Strategy
        """
        self._init_trainer()
        self._compile_model()
        self._autolog()

        x, y, batch_size = self._get_x_y_train()  # handle the format of the dataset
        val_dataset = self._get_x_y_val()

        history = self.model.fit(
            x=x,
            y=y,
            epochs=int(self.config.training.epochs),
            validation_data=val_dataset,
            batch_size=batch_size,
            callbacks=self._callbacks,
            class_weight=self.class_weight_values,
            verbose=int(self.config.training.verbose),
        )

        self._history = history.history
        self._clean_workspace()
        return history

    def _get_input_shape(self):
        """Get the input shape of input dataset"""
        if isinstance(self.train_dataset, tf.data.Dataset):
            for x, y in self.train_dataset.take(1):
                self.input_shape = (None, *x.shape[1:])
        else:
            self.input_shape = (None, *self.train_dataset[0].shape[1:])
        return self.input_shape

    def _get_x_y_val(self):
        if self.val_dataset is None:
            return None
        if isinstance(self.val_dataset, tf.data.Dataset):
            return self.val_dataset
        else:
            val = tf.data.Dataset.from_tensor_slices(self.val_dataset).batch(1)
            return val

    def _get_x_y_train(self):
        """Get the x and y for training based on the format of the dataset"""
        y, batch_size = None, None
        if isinstance(self.train_dataset, tf.data.Dataset):
            x = self.train_dataset
        else:
            x = tf.data.Dataset.from_tensor_slices(self.train_dataset)
            if self.config.training.shuffle:
                x = x.shuffle(self.train_dataset[0].shape[0])
            x = x.batch(self.config.training.batch_size).prefetch(1)
        return x, y, batch_size

    @property
    def config(self):
        return self._config

    def _compile_model(self) -> None:
        """Compile the model"""
        self.model.compile(
            optimizer=self._optimizer, loss=self._loss, metrics=self._get_metrics()
        )

    def _get_strategy(self):
        if self.strategy is None:
            return tf.distribute.MirroredStrategy()
        else:
            return self.strategy

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
            self._custom_module = importlib.import_module(module_name)
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
        opt_pars = self.config.training.optimizer_params
        default_value = "not_found"
        optimizer = getattr(
            tf.keras.optimizers, f"{self.config.training.optimizer}", default_value
        )
        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            return optimizer(opt_pars)
        else:
            return self._load_custom_module(optimizer, opt_pars)

    def _get_loss(self) -> Union[tf.keras.losses.Loss, Callable]:
        loss_name = self.config.training.loss
        loss = getattr(tf.keras.losses, loss_name, "not_found")
        if isinstance(loss, tf.keras.losses.Loss):
            return loss
        else:  # it's a custom loss
            return self._load_custom_module(loss_name)

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        for name_callback in self.config.training.callbacks:
            key = list(name_callback.keys())[0]
            args = list(name_callback.values())[0]

            callback_value = getattr(tf.keras.callbacks, key, "not_found")
            if isinstance(callback_value, tf.keras.callbacks.Callback):
                callbacks.append(callback_value(**args))
            else:
                callbacks.append(self._load_custom_module(key, args))

        return callbacks

    def _get_metrics(self) -> List[Union[tf.keras.metrics.Metric, Callable]]:
        """Get the metrics"""
        list_metrics = []
        for metric in self.config.training.metrics:
            if metric is None:
                continue
            metric_value = getattr(tf.keras.metrics, f"{metric}", "not_found")
            if isinstance(metric_value, tf.keras.metrics.Metric):
                list_metrics.append(metric_value())
            else:
                list_metrics.append(self._load_custom_module(metric))
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
        """Autolog the model"""
        if self.config.training.autolog:
            mlflow.tensorflow.autolog(log_models=False, disable=False)

    def _set_randomness(self, random_seed: Optional[int] = None) -> None:
        """Set the randomness"""
        if random_seed is not None:
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
