import os

import numpy as np
import tensorflow as tf
from yaket.schema.schema import Accelerator, TrainingModel
from yaket.trainer import Trainer
from yaket.converter.converter import Converter


class TestConverter(tf.test.TestCase):

    boring_model = tf.keras.Sequential(
        [
        tf.keras.layers.Dense(10, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]           
    )

    def test_init_converter_wrong_out_path(self):
        """
        GIVEN an out_path invalid parameter
        WHEN the converter is initialized
        THEN the converter should raise an exception
        """ 
        out_format = 'onnx'
        out_paths = [[1,2], (3,2), 1]

        for out_path in out_paths:
            with self.subTest(out_path=out_path):
                with self.assertRaises(TypeError):
                    Converter(out_format=out_format, out_path=out_path)
    
        
    def test_init_converter_wrong_out_format(self):
        """
        GIVEN an out_format invalid parameter
        WHEN the _init_converter method is called
        THEN the converter should raise an exception
        """ 
        out_format_list = [ 1, 1.0, ['string']]
        
        for out_format in out_format_list:
            with self.subTest(out_format=out_format):
                with self.assertRaises(TypeError):
                    c = Converter(out_format=out_format)

    

    def test_init_convert_wrong_type_onnx(self):
        """
        GIVEN a Converter correctly initialized with onnx out_format
        WHEN the convert method is called with a wrong type
        THEN the model should raise an exception
        """ 
        out_format = 'onnx'
        opset_onnx = 'ciao'
        
        with self.assertRaises(TypeError):
            Converter(out_format=out_format, opset_onnx=opset_onnx)

    def test_init_converter_wrong_opset_onnx(self):
        """
        GIVEN an opset_onnx invalid parameter
        WHEN the converter is initialized
        THEN the converter should raise an exception
        """ 
        out_format = 'onnx'
        opset_onnx = 0
        
        with self.assertRaises(ValueError):
            Converter(out_format=out_format, opset_onnx=opset_onnx)

    def test_init_converter_wrong_model(self):
        """
        GIVEN a model invalid parameter
        WHEN the converter is initialized
        THEN the converter should raise an exception
        """ 
        out_format = 'onnx'
        model_list = [[1,2], (3,2), 1, 'string']

        for model in model_list:
            with self.subTest(model=model):
                with self.assertRaises(TypeError):
                    Converter(out_format=out_format, model=model)

    def test_init_converter_wrong_model_path(self):
        """
        GIVEN a model_path invalid parameter
        WHEN the converter is initialized
        THEN the converter should raise an exception
        """ 
        out_format = 'onnx'
        model_paths = [[1,2], (3,2), 1]

        for model_path in model_paths:
            with self.subTest(model_path=model_path):
                with self.assertRaises(TypeError):
                    Converter(out_format=out_format, model_path=model_path)
            

    def test_init_converter_default(self):
        """
        GIVEN an out_format valid parameter
        WHEN the converter is initialized
        THEN the converter should be initialized with the correct default attributes
        """ 
        out_format = 'onnx'
        
        converter = Converter(out_format=out_format)
    

        self.assertEqual(converter.out_format, out_format)
        self.assertIsNone(converter.model)
        self.assertIsNone(converter.model_path)
        self.assertEqual(converter.out_path, 'model.onnx')
        self.assertEqual(converter.opset_onnx, 15)

    def test_convert_onnx_format(self):
        """
        GIVEN a Converter correctly initialized with onnx out_format
        WHEN the convert method is called
        THEN the model should be converted to onnx format
        """ 
        out_format = 'onnx'
        converter = Converter(out_format=out_format, model=self.boring_model)
        converter.convert()
        self.assertTrue(os.path.exists(converter.out_path))


    def test_convert_onnx_format_with_cli(self):
        """
        GIVEN a Converter correctly initialized with onnx out_format and without model
        WHEN the convert method is called with the cli
        THEN the model should be converted to onnx format
        """ 
        out_format = 'onnx'

        out_path = 'mymodel'

        self.boring_model.save(out_path)
        converter = Converter(out_format=out_format, model_path=out_path)
        converter.convert()

        self.assertTrue(os.path.exists(converter.out_path))
        

    
    
    def test_convert_tflite_format(self):
        """
        GIVEN a Converter correctly initialized with tflite out_format
        WHEN the convert method is called
        THEN the model should be converted to tflite format
        """ 
        out_format = 'tflite'
        converter = Converter(out_format=out_format, model=self.boring_model)
        converter.convert()
        self.assertTrue(os.path.exists(converter.out_path))

    def test_convert_tflite_format_with_cli(self):
        """
        GIVEN a Converter correctly initialized with tflite out_format and without model
        WHEN the convert method is called with the cli
        THEN the model should be converted to tflite format
        """ 
        out_format = 'tflite'

        out_path = 'mymodel'

        self.boring_model.save(out_path)
        converter = Converter(out_format=out_format, model_path=out_path)
        converter.convert()

        self.assertTrue(os.path.exists(converter.out_path))

    def test_convert_onnx_wrong_name_model(self):
        """
        GIVEN a Converter correctly initialized with onnx out_format
        WHEN the convert method is called with a out_name not ending with .onnx
        THEN the model should anyways save the model with the correct extension
        """ 
        out_format = 'onnx'
        converter = Converter(out_path = 'model', out_format=out_format, model=self.boring_model)
        converter.convert()
        self.assertEqual(converter.out_path, 'model.onnx')

    def test_convert_tflite_wrong_name_model(self):
        """
        GIVEN a Converter correctly initialized with tflite out_format
        WHEN the convert method is called with a out_name not ending with .tflite
        THEN the model should anyways save the model with the correct extension
        """ 
        out_format = 'tflite'
        converter = Converter(out_path = 'model', out_format=out_format, model=self.boring_model)
        converter.convert()
        self.assertEqual(converter.out_path, 'model.tflite')
        

    
        


    



    