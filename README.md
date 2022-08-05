# YAKET: Yaml Keras Trainer (or Yet Another Keras Trainer)

## Project status 🚧
Work In Progress. 

## Installation 💻
    pip install yaket

## Description 🔥
Yaket is a lightweight and simple module to train Keras modules by defining parameters directly using YAML file. YAML parameters are validated using Pydantic, hence typos or not allowed parameters will throw errors at the beginning of the execution.
This allows developer to focus uniquely on what matters: data and model development.
By having parameters defined in a human-readable format, it is possible to have an holistic view of training procedure without opening the code.
Morevoer, Data Scientists and ML Engineer won't need to add manually all training parameters, such as optimizer, callbacks, schedulers, thus reducing the
likelihood of human-induced code bugs.

## Features 🎊

1. Train models with tensorflow default optimizers, metrics, callbacks, and losses.
2. Train models with custom modules that can be defined in a python script whose path is used as argument to Trainer class.
3. Quickly use distributed multi-gpu and TPU training with `tf.distributed.strategy` (Experimental)
4. Log training parameters, models, and results using `mlflow.tensorflow.autolog()` module. The run will be saved in `mlruns` folder. 
5. Save the model in a particular folder and particular format (i.e., SavedModel,H5, or .pb)
6. Convert the saved model to ONNX/Tensorflow-Lite for on edge-deploymnet or faster inference.
7. More to come!

## Badges ✅
TODO: Tests are not covering the code YET.

## Visuals 📖

The YAML file contains most of the parameters used in Keras model.fit, such as epochs, verbose, callbacks. Below an example:

```yaml
    autolog: False
    optimizer: 
    - Adam
    - PiecewiseConstantDecay: 
        boundaries: [200, 300, 400]
        values:  [0.003, 0.0003,0.00003,0.000003]
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
    epochs: 1
    shuffle: False
    class_weights: 
        - False
    accelerator: cpu 
    steps_per_epoch: 1
```

The usage is very simple using python:

```python
    ...
    # Define path to yaml file
    path = "/yaket/examples/files/trainer.yaml"

    trainer = Trainer(
        config_path=path,
        train_dataset=(x_train, y_train), # Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset]
        val_dataset=(x_test, y_test), # Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset]
        model=model, # Keras Model
    )
    trainer.train()
```


## License
MIT License

