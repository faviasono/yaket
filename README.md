# YAKET: Yaml Keras Trainer (or Yet Another Keras Trainer)

[![pipeline status](https://gitlab.com/andreafavia/yaket/badges/main/pipeline.svg)](https://gitlab.com/andreafavia/yaket/-/commits/main) 
[![coverage report](https://gitlab.com/andreafavia/yaket/badges/main/coverage.svg)](https://gitlab.com/andreafavia/yaket/-/commits/main) 
[![Latest Release](https://gitlab.com/andreafavia/yaket/-/badges/release.svg)](https://gitlab.com/andreafavia/yaket/-/releases)

## Installation 💻
    pip install yaket

## Description 🔥
Yaket is a lightweight and simple module to train Keras modules by defining parameters directly using YAML file. 

YAML parameters are validated using Pydantic, hence typos or not allowed parameters will throw errors at the beginning of the execution.
This allows developer to focus uniquely on what matters: data and model development.

Data Scientists and ML Engineer won't need to add manually all training parameters, such as optimizer, callbacks, schedulers, thus reducing the
likelihood of human-induced code bugs.

## Features 🎊

1. Train models with tensorflow default optimizers, metrics, callbacks, and losses.
2. Convert the saved model to **ONNX** or **Tensorflow-Lite** for on edge-deploymnet or faster inference.
3. Quickly use distributed multi-gpu and TPU training with `tf.distributed.strategy` *(Experimental)*
4. Train models with custom modules defined in python script.
5. Log training parameters, models, and results using `mlflow.tensorflow.autolog()` module. The run will be saved in `mlruns` folder. 
6. Save the model in a particular folder and particular format (i.e., SavedModel,H5, or .pb)
7. Train with `sample_weight_mode = 'temporal'` when training sequence models.

8. More to come!

## Visuals 📖

The YAML file contains most of the parameters used in Keras model.fit, such as epochs, verbose, callbacks. Below an example:

```yaml
    autolog: False
    optimizer: 
    - Adam:
       learning_rate: 0.001
    batch_size: 64 
    loss: 
    SparseCategoricalCrossentropy: 
        from_logits: True
    callbacks:
        - EarlyStopping:
            monitor: val_accuracy
            patience: 2
            restore_best_weights: True  
    verbose: 1 
    epochs: 100
    shuffle: True
    accelerator: mgpu 
```

The usage is very simple using python:

```python
    
    model = ... # define your tf.keras.Model

    # Define path to yaml file
    path = "/yaket/examples/files/trainer.yaml"

    # Initialize trainer
    trainer = Trainer(
        config_path=path,
        train_dataset=(x_train, y_train),
        val_dataset=(x_test, y_test),
        model=model, 
    )
    trainer.train() # train based on the parameters defined in the yaml file
    trainer.clear_ram() # clear RAM after training

    trainer.convert_model(format_model = 'onnx') # Convert to ONNX

    
```

Other scenarios are visible in **examples** folder.

## License
MIT License

