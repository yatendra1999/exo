from abc import abstractmethod
from typing import Optional

import torch
import numpy as np
from flax import nnx
from jax import numpy as jnp
import jax
from transformers import PretrainedConfig

from exo.inference.shard import Shard

class FlaxBaseModule(nnx.Module):
    
    @classmethod
    @abstractmethod
    def from_safetensor(cls, config: PretrainedConfig, key: str, path: str, framework: str, dense: bool = True):
        pass

    @classmethod
    def convert_from_pt(cls, tensor: torch.Tensor, dense: bool = True):

        dtype_dict = {
            torch.bool : jnp.bool,
            torch.uint8 : jnp.uint8,
            torch.int8 : jnp.int8,
            torch.int16 : jnp.int16,
            torch.int32 : jnp.int32,
            torch.int64 : jnp.int64,
            torch.float16 : jnp.float16,
            torch.float32 : jnp.float32,
            torch.float64 : jnp.float64,
            torch.complex64 : jnp.complex64,
            torch.complex128 : jnp.complex128,
            torch.bfloat16 : jax.dtypes.bfloat16
        }
        orig_dtype = tensor.dtype
        if orig_dtype == torch.bfloat16:
            tensor = tensor.float()

        jax_dtype = dtype_dict[orig_dtype]
        
        if tensor.dim() < 2:
            return jnp.array(tensor.detach().numpy(), dtype=jax_dtype)

        if dense: ## Linear(Dense) layers in JAX require weights to be transposed if they are being converted from pytorch.
            return jnp.array(tensor.detach().numpy().transpose(), dtype=jax_dtype)

        return jnp.array(tensor.detach().numpy(), dtype=jax_dtype)

class FlaxLlmModel(nnx.Module):
    
    @abstractmethod
    def load_shard(self, config: PretrainedConfig, shard: Shard):
        pass

    @abstractmethod
    def sample_logits(self, request_id: str, hidden_state: np.ndarray) -> np.ndarray:
        pass