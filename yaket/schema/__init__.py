from pydantic_yaml import YamlModel
from typing import Dict, Optional, Any
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


#pydantic-yaml-0.8.0



class Training(BaseModel, extra=Extra.allow):
    autolog: bool
    optimizer: constr(strict=True)
    optimizer_params: Optional[Dict[str, Any]] = None
    metrics: conlist(item_type=str, min_items=1, unique_items=True)
    epochs: PositiveInt
    batch_size: PositiveInt  # if format is numpy
    loss: constr(strict=True)
    callbacks: conlist(item_type=Dict[str, Any], min_items=0)
    verbose: conint(ge=1, le=2)
    shuffle: bool
    class_weights: conlist(item_type=Any, min_items=1)

class TrainerModel(YamlModel, extra=Extra.allow):
    training: Training = Field(...)