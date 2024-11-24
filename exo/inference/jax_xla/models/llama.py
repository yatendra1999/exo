from transformers.models.llama.modeling_flax_llama import *
from jax import numpy as jnp
import jax


pretrained = FlaxLlamaForCausalLM.from_pretrained("neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic")
print(pretrained.config)