"""
Yaket Trainer.

This package requires the following modules:
    - `tensorflow`: https://www.tensorflow.org/install
    - `mlflow`: https://mlflow.org/docs/latest/index.html
    - `yaket.converter.converter` : Prodive a module to convert models to onnx or tflite
    - `yaket.trainer.schema` : Provide a Pydantic module for the Trainer class
    - `numpy`: https://numpy.org/
"""

from .trainer import Trainer

__version__ = "1.3.4"
