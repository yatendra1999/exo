from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.llama import LlamaConfig

from .models import FlaxLlmModel, ShardedLlamaModel

def resolve_config(model_path: Path) -> tuple[PretrainedConfig, FlaxLlmModel]:
    
    config_path = model_path.joinpath("config.json")
    if not (config_path.exists() and config_path.is_file()):
        raise Exception("Unable to load the config from specified model.")
    
    config = PretrainedConfig.from_json_file(config_path)
    if config.model_type == "llama":
        return LlamaConfig.from_json_file(config_path), ShardedLlamaModel
    else:
        raise Exception("Provided model architecture is not supported.")