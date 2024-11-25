from transformers.models.llama.modeling_flax_llama import *
from jax import numpy as jnp
import jax
from flax import nnx
# from flax import linen as nn

ACT_MAP: dict[str, callable] = {
    'silu': nnx.swish
}


class LLamaRMSNorm(nnx.Module):

    def __init__(self, config: LlamaConfig, weights: jax.Array):
        self.weights = nnx.Param(weights)
        self.epsilon = nnx.Param(config.rms_norm_eps)
        self.dtype = jax.dtypes.bfloat16

    def __call__(self, hidden_states: jax.Array):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weights * jnp.asarray(hidden_states, dtype=self.dtype)
    
class LLamaMLP(nnx.Module):
    
    def __init__(self, config: LlamaConfig, weights_map: dict[str, jax.Array], rng: nnx.rnglib.Rngs):
        self.up_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias, kernel_init=lambda x,y,z : weights_map['up'], rngs=rng)
        self.gate_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias, kernel_init=lambda x,y,z : weights_map['gate'], rngs=rng)
        self.down_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias, kernel_init=lambda x,y,z : weights_map['down'], rngs=rng)
        self.activation_fn = ACT_MAP[config.hidden_act]
         

    def __call__(self, hidden_states: jax.Array):
        ### Ignoring the values of pretraining_tp > 1. Find more details here: https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig.pretraining_tp
    
        return self.down_proj( self.activation_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

