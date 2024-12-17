from abc import abstractmethod
from typing import Optional

import torch
import numpy as np
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig

from exo.inference.shard import Shard

class FlaxBaseModule(nnx.Module):
    
    @classmethod
    @abstractmethod
    def from_safetensor(cls, config: PretrainedConfig, key: str, path: str, framework: str, dense: bool = True):
        pass

    @classmethod
    def convert_from_pt(cls, tensor: torch.Tensor, dense: bool = True):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        
        if tensor.dim() < 2:
            return jnp.array(tensor.detach().numpy())

        if dense: ## Linear(Dense) layers in JAX require weights to be transposed if they are being converted from pytorch.
            return jnp.array(tensor.detach().numpy().transpose())

        return jnp.array(tensor.detach().numpy())

class FlaxLlmModel(nnx.Module):
    
    @abstractmethod
    def load_shard(self, config: PretrainedConfig, shard: Shard):
        pass

    @abstractmethod
    def sample_logits(self, request_id: str, hidden_state: np.ndarray) -> np.ndarray:
        pass