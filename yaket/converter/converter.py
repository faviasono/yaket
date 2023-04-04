import tf2onnx
import tensorflow as tf
from typing import  Optional, Union
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass
class Converter:
    """
    This class is used to convert a keras model to onnx or tflite.

    Parameters:
        model_path (str|Path):  Path to the model to convert. If None, the model must be provided.
        model (tf.keras.Model):  If None, the model path must be provided.
        out_path (str|Path): Path to the output file.
        out_format (str|Path): Output format. Can be either "onnx" or "tflite".
        opset_onnx (int):  Opset version for the onnx model. Default is 11.

    Methods:
        convert() - Convert the model to the specified format.
    """

    out_format: Union[str, Path]
    model: Optional[tf.keras.Model] = None
    model_path: Optional[Union[str, Path]] = None
    out_path: Union[str, Path] = "model.onnx"  # model.tflite
    opset_onnx: int = 15

    def __post_init__(self) -> None:
        if not isinstance(self.opset_onnx, int):
            raise TypeError("opset_onnx must be an integer")
        if self.model_path is not None and not (isinstance(self.model_path, Path) or isinstance(self.model_path, str)):
            raise TypeError("model_path must be a string or a Path object")
        if self.opset_onnx < 1:
            raise ValueError("opset_onnx must be a positive integer")
        if not (isinstance(self.out_format, str) or isinstance(self.out_path, Path)):
            raise TypeError("out_format must be a string")
        if self.out_format not in ["onnx", "tflite"]:
            raise ValueError(f"Unknown output format: {self.out_format}")
        if self.model is not None and not isinstance(self.model, tf.keras.Model):
            raise TypeError(f"Model must be a tf.keras.Model, got {type(self.model)}")
        if not (isinstance(self.out_path, str) or isinstance(self.out_path, Path)):
            raise TypeError(
                f"Output path must be a string or Path type, got {type(self.out_path)}"
            )

    def convert(self) -> bool:
        "Convert the model to the specified format"

        if self.out_format == "onnx":
            self._convert_to_onnx()
            return True
        elif self.out_format == "tflite":
            self._convert_to_tflite()
            return True

    def _convert_to_onnx(self):
        "Function to convert a keras model to onnx"

        if not self.out_path.endswith(".onnx"):
            self.out_path = self.out_path + ".onnx"
        try:
            if self.model is None:
                opset_string = (
                    f"--opset {self.opset_onnx}" if self.opset_onnx is not None else ""
                )
                python_version = 3 if sys.version_info.major == 3 else ""
                command = f"python{python_version} -m tf2onnx.convert {opset_string} --saved-model {self.model_path} --output {self.out_path}"
                print(command)
                subprocess.run(command.split(), shell=False)
            else:
                specs = [
                    tf.TensorSpec(
                        input_model.shape, input_model.dtype, name=f"input_{i}"
                    )
                    for i, input_model in enumerate(self.model.inputs)
                ]
                model_proto, _ = tf2onnx.convert.from_keras(
                    self.model, input_signature=specs, output_path=self.out_path
                )
                output_names = [n.name for n in model_proto.graph.output]
                print("Specs:", specs)
                print("Output names:", output_names)
        except Exception as e:
            raise e

    def _convert_to_tflite(self):
        "Function to convert a keras model to tflite"

        if self.model is None:
            tf_converter = tf.lite.TFLiteConverter.from_saved_model(
                self.model_path
            )  # path to the SavedModel directory
            tflite_model = tf_converter.convert()
        else:
            tf_converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = tf_converter.convert()

        if not self.out_path.endswith(".tflite"):
            self.out_path = self.out_path + ".tflite"
        # save model
        with open(self.out_path, "wb") as f:
            f.write(tflite_model)
