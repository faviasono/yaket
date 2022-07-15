from typing import Dict, Optional, Any
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
    FilePath,
    DirectoryPath,
    Field,
)


class TrainingModel(BaseModel, extra=Extra.allow):
    autolog: bool
    optimizer: constr(strict=True)
    optimizer_params: Optional[Dict[str, Any]] = None
    
    epochs: PositiveInt
    batch_size: PositiveInt  # if format is numpy
    loss: constr(strict=True)
    callbacks: Optional[conlist(item_type=Dict[str, Any], min_items=0)] 
    metrics: Optional[conlist(item_type=str, min_items=1, unique_items=True)]
    verbose: conint(ge=1, le=2)
    shuffle: bool
    class_weights: Optional[conlist(item_type=Any, min_items=1)]


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
