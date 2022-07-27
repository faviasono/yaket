import tf2onnx
import tensorflow as tf
from typing import List, Optional, Tuple, Union, Any, Dict, Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass
class Converter:
    out_format: Union[str, Path]
    model: Optional[tf.keras.Model] = None
    model_path: Optional[Union[str, Path]] = None
    out_path: Union[str, Path] = "model.onnx" # model.tflite
    opset_onnx: int = 15

    def _init_converter(self) -> None:
        if self.out_format not in ["onnx", "tf-lite"]:
            raise ValueError(f"Unknown output format: {self.out_format}")
        if self.model is not None and not isinstance(self.model, tf.keras.Model):
            raise ValueError(f"Model must be a tf.keras.Model, got {type(self.model)}")
        if not isinstance(self.out_path, str) or not isinstance(self.out_path, Path):
            raise ValueError(
                f"Output path must be a string or Path type, got {type(self.out_path)}"
            )

    def convert(self) -> bool:
        if self.out_format == "onnx":
            self._convert_to_onnx()
            return True
        elif self.out_format == "tflite":
            self._convert_to_tflite()
            return True
        else:
            raise ValueError(f"Unknown output format: {self.out_format}")

    def _convert_to_onnx(self):
        "Function to convert a keras model to onnx using command line"
        # submit command to command line

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

        if self.model is None:
            tf_converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path) # path to the SavedModel directory
            tflite_model = tf_converter.convert()
        else:
            tf_converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = tf_converter.convert()

        if not self.out_path.endswith(".tflite"):
            self.out_path = self.out_path + ".tflite"
        # save model
        with open(self.out_path, 'wb') as f:
            f.write(tflite_model)
