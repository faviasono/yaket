from typing import Dict, Optional, Any, Union
import yaml
import os
from pydantic import (
    BaseModel,
    validator,
    Extra,
    PositiveInt,
    conint,
    conlist,
    constr,
)
from enum import Enum, auto


class Accelerator(Enum):
    cpu = auto()
    gpu = auto()
    mgpu = auto()
    tpu = auto()
    @classmethod
    def list(cls):
        return list(map(lambda c: c.name, cls))
    @classmethod
    def list_gpu(cls):
        return list(map(lambda c: c.name if 'gpu' in c.name else '', cls))



class TrainingModel(BaseModel, extra=Extra.allow):
    autolog: bool = False
    optimizer: Union[constr(strict=True), conlist(item_type=Any,min_items = 1, max_items = 2)] = 'Adam'
    optimizer_params: Optional[Dict[str, Any]] = None
    epochs: PositiveInt = 1
    batch_size: PositiveInt = 1  # if format is numpy
    loss: Union[constr(strict=True), Dict[str, Any]]
    callbacks: Optional[conlist(item_type=Union[str,Dict[str, Any]], min_items=0)] 
    metrics: Optional[conlist(item_type=Union[str,Dict], min_items=1, unique_items=True)]
    verbose: conint(ge=1, le=2) = 1
    shuffle: bool = True
    accelerator: Optional[constr(strict=True)] 
    steps_per_epoch: Optional[PositiveInt] = None
    sample_weight_mode: Optional[constr(strict=True)] = None

    @validator('sample_weight_mode')
    def sample_weight_mode_validator(cls, v):
        if v is not None:
            assert v in ['temporal']
        return v

    @validator('accelerator')
    def accelerator_validator(cls, v):
        if v is None:
            return None
        if v not in Accelerator.list():
            raise ValueError(f'{v} is not a valid accelerator.\nPlease use: {Accelerator.list()}')
        if v in Accelerator.list_gpu() and not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('ERROR: No GPU has been detected. Change accelerator.')
        return v


def yaml_to_pydantic(path: str, validate: bool) -> TrainingModel:
    if not os.path.exists(path=path):
        raise FileNotFoundError(f"{path} not found")
    if os.path.isfile(path) and path.endswith(".yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return (
            TrainingModel(**data)
            if validate
            else TrainingModel.construct(_fields_set=None, **data)
        )
    else:
        raise ValueError(f"{path} is not a yaml file")
