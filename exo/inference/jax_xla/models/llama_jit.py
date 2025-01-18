from concurrent.futures import ThreadPoolExecutor

import time
from transformers.models.llama.modeling_flax_llama import *
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as TorchLlamaDecoderLayer,
)
from transformers.utils import SAFE_WEIGHTS_NAME, cached_file
from jax import numpy as jnp
import jax
from flax import nnx
from flax.nnx import make_causal_mask, combine_masks
from exo.inference.shard import Shard
from safetensors import safe_open
from .base import FlaxBaseModule, FlaxLlmModel
from .utils.rope import compute_llama3_parameters
from .utils.logits import (
    TopKLogitProcessor,
    TopPLogitProcessor,
    TemperatureLogitProcessor,
    LogitProcessorList
)
from jax.nn import dot_product_attention
from jax import lax
import torch
jit_attention = jax.jit(dot_product_attention, static_argnames=['bias', 'mask', 'scale', 'is_causal', 'query_seq_lengths', 'key_value_seq_lengths', 'local_window_size', 'implementation'])


ACT_MAP: dict[str, callable] = {"silu": nnx.jit(nnx.swish)}
act_fn_jit = jax.jit(jax.nn.swish)


def convert_from_pt(tensor, dense: bool = False):
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

@jax.jit
def rotate_half(tensor: jax.Array):
    """Rotates half the hidden dims of the input."""
    mid = tensor.shape[-1] // 2
    upper_half = jax.lax.neg(jax.lax.slice_in_dim(tensor, start_index=mid, limit_index=tensor.shape[-1], axis=-1))
    lower_half = jax.lax.slice_in_dim(tensor, start_index=0, limit_index=mid, axis=-1)
    return lax.concatenate((upper_half, lower_half), dimension=tensor.ndim-1)
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]),
        axis=-1,
    )
    return rotate_half_tensor

@jax.jit
def prep_rotary_embed(tensor: jax.Array, inv_freq: jax.Array, attn_scaling: jax.Array):
    position_ids = jnp.expand_dims(jnp.arange(start=0, stop=tensor.shape[-1], dtype=jax.dtypes.bfloat16), axis=0)
    inv_freq_expanded = jnp.expand_dims(inv_freq, (0, 2))
    position_ids_expanded = jnp.expand_dims(position_ids, (1))

    freq = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
    freq = jnp.append(freq, freq, axis=-1)

    cos = jnp.cos(freq) * attn_scaling
    sin = jnp.sin(freq) * attn_scaling

    return sin, cos

@jax.jit
def apply_rotary_embed(query: jax.Array, key: jax.Array, sin: jax.Array, cos: jax.Array):
    orig_dtype = query.dtype
    query_rotated = rotate_half(query)
    key_rotated = rotate_half(key)
    sin = lax.expand_dims(sin, [2])
    cos = lax.expand_dims(cos, [2])
    query = lax.convert_element_type(lax.add(query * cos, query_rotated * sin), orig_dtype)
    key = lax.convert_element_type(lax.add(key * cos, key_rotated * sin), orig_dtype)
    return query, key

# @partial(nnx.jit, static_argnames=['expand_axis'])
# class LlamaRotaryEmbedding(nnx.Module):

#     sin = nnx.Param(jnp.zeros((1,1)))
#     cos = nnx.Param(jnp.zeros((1,1)))
#     inv_freq = nnx.Param(jnp.zeros((1,1)))
#     attention_scaling = nnx.Param(jnp.ones((1)))

#     # @partial(nnx.jit, static_argnames=['config'])
#     def __init__(self, config: LlamaConfig):
#         self.inv_freq.value, self.attention_scaling.value = compute_llama3_parameters(config)

#     def create_embed(self, position_ids: jax.Array):
#         inv_freq_expanded = jnp.expand_dims(self.inv_freq, (0, 2))
#         position_ids_expanded = jnp.expand_dims(position_ids, (1))

#         freq = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
#         freq = jnp.append(freq, freq, axis=-1)

#         self.cos.value = jnp.cos(freq) * self.attention_scaling
#         self.sin.value = jnp.sin(freq) * self.attention_scaling

#     # @partial(nnx.jit, static_argnames=['expand_axis'])
#     def __call__(self, query: jax.Array, key: jax.Array, expand_axis = 2):
#         cos = jnp.expand_dims(self.cos.value, axis=expand_axis)
#         sin = jnp.expand_dims(self.sin.value, axis=expand_axis)

#         query_embed = ((query * cos) + (rotate_half(query) * sin)).astype(query.dtype)
#         key_embed = ((key * cos) + (rotate_half(key) * sin)).astype(key.dtype)

#         return query_embed, key_embed

# rotary_embedding = None


# # Define nnx-based Llama Attention
# class LlamaAttention(FlaxBaseModule):
#     num_heads = nnx.Param(jnp.array(1))
#     num_kv_heads = nnx.Param(jnp.array(1))
#     cached_key = nnx.Param(jnp.array([]))
#     cached_value = nnx.Param(jnp.array([]))
#     cache_index = nnx.Param(jnp.array(0))

#     def __init__(
#         self,
#         config: LlamaConfig,
#         weights: dict[str, jax.Array],
#         rngs: nnx.rnglib.Rngs,
#         dtype=jax.dtypes.bfloat16,
#     ):
#         self.config = config
#         self.dtype = dtype
#         self.attention_dropout = config.attention_dropout
#         self.hidden_size = config.hidden_size
#         self.num_heads.value = config.num_attention_heads
#         self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads.value)
#         self.num_kv_heads.value = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads.value // self.num_kv_heads.value
#         self.rope_theta = config.rope_theta
#         self.is_causal = True,
#         self.cached_key.value = jnp.zeros((1, 1, self.num_kv_heads.value, self.head_dim))
#         self.cached_value.value = jnp.zeros((1, 1, self.num_kv_heads.value, self.head_dim))

#         self.attention_softmax_in_fp32 = dtype is not jnp.float32

#         self.q_proj = nnx.Linear(
#             config.hidden_size,
#             self.num_heads.value * self.head_dim,
#             use_bias=False,
#             rngs=rngs,
#         )
#         self.q_proj.kernel.value = weights['q']
#         self.q_proj = nnx.jit(self.q_proj)
        
#         self.k_proj = nnx.Linear(
#             config.hidden_size,
#             self.num_kv_heads.value * self.head_dim,
#             use_bias=False,
#             rngs=rngs,
#         )
#         self.k_proj.kernel.value = weights['k']
#         self.k_proj = nnx.jit(self.k_proj)

#         self.v_proj = nnx.Linear(
#             config.hidden_size,
#             self.num_kv_heads.value * self.head_dim,
#             use_bias=False,
#             rngs=rngs,
#         )
#         self.v_proj.kernel.value = weights["v"]
#         self.v_proj = nnx.jit(self.v_proj)

#         self.o_proj = nnx.Linear(
#             self.num_heads.value * self.head_dim,
#             config.hidden_size,
#             use_bias=False,
#             rngs=rngs,
#         )
#         self.o_proj.kernel.value = weights["o"]
#         self.o_proj = nnx.jit(self.o_proj)
#         self.attn_impl = nnx.jit(dot_product_attention, static_argnames=['is_causal', 'bias', 'mask'])

#     @classmethod
#     def from_safetensor(cls, config, key: str, path: str, framework: str):
#         rngs = nnx.Rngs(0)
#         with safe_open(path, framework=framework) as st:
#             weights = {
#                 "q": cls.convert_from_pt(st.get_tensor(f"{key}.q_proj.weight")),
#                 "k": cls.convert_from_pt(st.get_tensor(f"{key}.k_proj.weight")),
#                 "v": cls.convert_from_pt(st.get_tensor(f"{key}.v_proj.weight")),
#                 "o": cls.convert_from_pt(st.get_tensor(f"{key}.o_proj.weight")),
#             }
#         return cls(config=config, rngs=rngs, weights=weights)

#     def _split_heads(self, hidden_states: jax.Array, num_heads:int):
#         return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, -1))

#     def _merge_heads(self, hidden_states: jax.Array):
#         return hidden_states.reshape(hidden_states.shape[:2] + (-1,))
    
#     @nnx.jit
#     def pre_attn(self, hidden_states: jax.Array):
#         query = self.q_proj(hidden_states)
#         key = self.k_proj(hidden_states)
#         value = self.v_proj(hidden_states)

#         # Split heads
#         query = query.reshape(*query.shape[:2], self.num_heads, -1)
#         key = key.reshape(*key.shape[:2], self.num_kv_heads, -1)
#         value = value.reshape(*value.shape[:2], self.num_kv_heads, -1)

#         # Apply rotary embeddings
#         global rotary_embedding
#         query, key = rotary_embedding(query, key)

#         # Apply caching for autoregressive decoding
#         self._concatenate_to_cache(
#             key, value
#         )
#         key = jnp.repeat(self.cached_key, self.num_key_value_groups, axis=2)
#         value = jnp.repeat(self.cached_value, self.num_key_value_groups, axis=2)
#         return query, key, value
    
#     @nnx.jit
#     def post_attn(self, attn_weights: jax.Array):
#         attn_output = self._merge_heads(attn_weights)
#         attn_output = self.o_proj(attn_output)
#         return attn_output

#     def __call__(
#         self,
#         hidden_states: jax.Array,
#         attention_mask: jax.Array,
#         position_ids: jax.Array,
#     ):
#         query, key, value = self.pre_attn(hidden_states)
#         is_causal = True if position_ids.shape[-1] > 1 else False
#         attn_weights = jit_attention(query, key, value, bias=None, is_causal=is_causal, mask=None)
#         return self.post_attn(attn_weights)

#     # @nnx.jit
#     def _concatenate_to_cache(self, key: jax.Array, value):
#         """
#         This function takes projected key, value states from a single input token and concatenates the states to cached
#         states from previous steps.
#         """
#         orig_key = self.cached_key[..., :self.cache_index.value, :, :].astype(key.dtype)
#         key = jnp.append(orig_key, key, axis=-3)
#         self.cached_key.value = key

#         orig_value = self.cached_value[..., :self.cache_index.value, :, :].astype(value.dtype)
#         value = jnp.append(orig_value, value, axis=-3)
#         self.cached_value.value = value

#         num_updated_cache_vectors = key.shape[-3]
#         self.cache_index.value = self.cache_index.value + num_updated_cache_vectors


# class LlamaRMSNorm(FlaxBaseModule):

#     def __init__(self, config: LlamaConfig, weights: jax.Array):
#         self.weights = nnx.Param(weights)
#         self.epsilon = nnx.Param(config.rms_norm_eps)

#     def __call__(self, hidden_states: jax.Array):
#         input_dtype = hidden_states.dtype
#         variance = jnp.asarray(hidden_states, dtype=jnp.float32)
#         variance = jnp.power(variance, 2)
#         variance = variance.mean(-1, keepdims=True)
#         # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
#         hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

#         return self.weights * jnp.asarray(hidden_states, dtype=input_dtype)

#     @classmethod
#     def from_safetensor(cls, config, key, path, framework):
#         with safe_open(path, framework=framework) as st:
#             weights = st.get_tensor(f"{key}.weight")
#         weights = cls.convert_from_pt(weights)
#         return cls(config, weights)


# class LlamaMLP(FlaxBaseModule):

#     def __init__(
#         self,
#         config: LlamaConfig,
#         weights_map: dict[str, jax.Array],
#         rng: nnx.rnglib.Rngs,
#     ):
#         self.up_proj = nnx.Linear(
#             config.intermediate_size,
#             config.hidden_size,
#             use_bias=config.mlp_bias,
#             rngs=rng,
#         )
#         self.up_proj.kernel.value = weights_map["up"]

#         self.gate_proj = nnx.Linear(
#             config.intermediate_size,
#             config.hidden_size,
#             use_bias=config.mlp_bias,
#             rngs=rng,
#         )
#         self.gate_proj.kernel.value = weights_map["gate"]

#         self.down_proj = nnx.Linear(
#             config.hidden_size,
#             config.intermediate_size,
#             use_bias=config.mlp_bias,
#             rngs=rng,
#         )
#         self.down_proj.kernel.value = weights_map["down"]
#         self.activation_fn = ACT_MAP[config.hidden_act]

#     @nnx.jit
#     def __call__(self, hidden_states: jax.Array):
#         ### Ignoring the values of pretraining_tp > 1. Find more details here: https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig.pretraining_tp

#         return self.down_proj(
#             self.activation_fn(self.gate_proj(hidden_states))
#             * self.up_proj(hidden_states)
#         )

#     @classmethod
#     def from_safetensor(cls, config, key, path, framework):
#         with safe_open(path, framework=framework) as st:
#             weights = {
#                 "up": cls.convert_from_pt(st.get_tensor(f"{key}.up_proj.weight")),
#                 "gate": cls.convert_from_pt(st.get_tensor(f"{key}.gate_proj.weight")),
#                 "down": cls.convert_from_pt(st.get_tensor(f"{key}.down_proj.weight")),
#             }
#         return cls(config, weights, rng=nnx.Rngs(0))


# class LlamaDecoderLayer(FlaxBaseModule):

#     def __init__(
#         self,
#         config: LlamaConfig,
#         safetensor_path=None,
#         safetensor_key: str = None,
#         weights_map: dict = None,
#         rngs: nnx.rnglib.Rngs = nnx.Rngs(0),
#     ):

#         if weights_map is not None:
#             if all(
#                 [
#                     x in weights_map
#                     for x in [
#                         "input_layernorm",
#                         "self_attn",
#                         "post_attention_layernorm",
#                         "mlp",
#                     ]
#                 ]
#             ):
#                 self.input_layernorm = LlamaRMSNorm(
#                     config, weights_map["input_layernorm"]
#                 )
#                 self.self_attn = LlamaAttention(config, weights_map["self_attn"], rngs)
#                 self.post_attention_layernorm = LlamaRMSNorm(
#                     config, weights_map["post_attention_layernorm"]
#                 )
#                 self.mlp = LlamaMLP(config, weights_map["mlp"], rngs)
#                 return
#             else:
#                 raise Exception("Weights provided do not contain all required layers.")

#         if safetensor_path is None or safetensor_key is None:
#             raise Exception(
#                 "Both safetensor_path and safetensor_key are required to init layer from safetensors file."
#             )

#         self.input_layernorm = LlamaRMSNorm.from_safetensor(
#             config, f"{safetensor_key}.input_layernorm", safetensor_path, framework="pt"
#         )
#         self.self_attn = LlamaAttention.from_safetensor(
#             config, f"{safetensor_key}.self_attn", safetensor_path, framework="pt"
#         )
#         self.post_attention_layernorm = LlamaRMSNorm.from_safetensor(
#             config,
#             f"{safetensor_key}.post_attention_layernorm",
#             safetensor_path,
#             framework="pt",
#         )
#         self.mlp = LlamaMLP.from_safetensor(
#             config, f"{safetensor_key}.mlp", safetensor_path, framework="pt"
#         )


#     @classmethod
#     def from_safetensor(cls, config, key, path, framework, dense=True):
#         return cls(config, safetensor_key=key, safetensor_path=path)

#     # @partial(nnx.jit, static_argnames=["is_causal"])
#     def __call__(
#         self,
#         hidden_states,
#         is_causal: bool,
#         position_ids=None,
#     ):
#         residual = hidden_states
#         hidden_states = self.input_layernorm(hidden_states)
#         attn_output = self.self_attn(
#             hidden_states,
#             position_ids=position_ids,
#             is_causal=is_causal
#         )
#         hidden_states = residual + attn_output

#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         # residual connection
#         hidden_states = residual + hidden_states

#         return hidden_states


# class LlamaEmbedding(nnx.Embed, FlaxBaseModule):

#     @classmethod
#     def from_safetensor(cls, config, key, path, framework):
#         with safe_open(path, framework=framework) as st:
#             weights = st.get_tensor(f"{key}.weight")
#         weights = cls.convert_from_pt(weights, dense=False)
#         return cls(
#             weights.shape[0],
#             weights.shape[1],
#             embedding_init=lambda x, y, z: weights,
#             rngs=nnx.Rngs(0),
#         )


# class ShardedLlamaModel(FlaxLlmModel):
#     embed = None
#     norm = None
#     layers: list[FlaxBaseModule] = []
#     cache_positions: dict[str, int] = {}
#     executor: ThreadPoolExecutor
#     config: LlamaConfig | None
#     shard: Shard | None
#     lm_head: nnx.Module
#     dtype = jax.dtypes.bfloat16

#     def __init__(self):
#         self.shard = None
#         self.config = None
#         self.executor = ThreadPoolExecutor(max_workers=1)

#     def _load_model(self):
#         if self.shard == None:
#             raise Exception("Model attempted to load from an empty shard.")
#         if self.config == None:
#             raise Exception("Model attempted to load from an empty config.")

#         ## TODO
#         safetensor_path = cached_file(
#             "unsloth/Llama-3.2-1B-Instruct", SAFE_WEIGHTS_NAME
#         )

#         global rotary_embedding
#         rotary_embedding = LlamaRotaryEmbedding(self.config)

#         if self.shard.is_first_layer():
#             embeddings = LlamaEmbedding.from_safetensor(
#                 self.config, "model.embed_tokens", safetensor_path, framework="pt"
#             )
#             self.embed = embeddings
#             self.lm_head = embeddings.attend

#         for layer_idx in range(self.shard.start_layer, self.shard.end_layer + 1):
#             layer_module = (LlamaDecoderLayer.from_safetensor(
#                 self.config,
#                 f"model.layers.{layer_idx}",
#                 safetensor_path,
#                 framework="pt",
#             ))
#             self.layers.append(layer_module)

#         if self.shard.is_last_layer():
#             module = (LlamaRMSNorm.from_safetensor(
#                 self.config, f"model.norm", safetensor_path, framework="pt"
#             ))
#             self.norm = module

#     def load_shard(self, config: LlamaConfig, shard: Shard):
#         if self.shard == shard:
#             return
#         self.shard = shard
#         if hasattr(self, "layers"):
#             self.layers = []
#         self.config = config
#         self._load_model()

#     def generate_args(self, request_id: str, input_shape: tuple[int, ...]) -> dict[str,]:
#         if request_id in self.cache_positions:
#             start = self.cache_positions[request_id]
#         else:
#             start = 0
#         end = start + input_shape[-1]
#         self.cache_positions[request_id] = end
#         model_args = {
#             # "attention_mask" : jnp.ones(input_shape, dtype=jnp.uint4),
#             "is_causal": True if input_shape[1] > 1 else False,
#             "position_ids": jnp.expand_dims(jnp.arange(start=start, stop=end, dtype=self.dtype), axis=0)
#         }
#         return model_args

#     # @partial(nnx.jit, static_argnames=['request_id'])
#     def __call__(self, request_id: str, hidden_states: jax.Array) -> jax.Array:
#         last_time = time.time_ns()
#         start_time = time.time_ns()
#         model_args = self.generate_args(request_id, hidden_states.shape)
#         global rotary_embedding
#         rotary_embedding.create_embed(model_args['position_ids'])

#         if self.embed != None:
#             hidden_states = self.embed(hidden_states)
#             print(f"Embed time: {time.time_ns() - last_time} ns")
#             last_time = time.time_ns()

#         for layer in self.layers:
#             hidden_states = layer(hidden_states, **model_args)
#             print(f"Layer time: {time.time_ns() - last_time} ns")
#             last_time = time.time_ns()

#         if self.norm != None:
#             hidden_states = self.norm(hidden_states)
#             print(f"Norm time: {time.time_ns() - last_time} ns")
        
#         print(f"Total time: {time.time_ns() - start_time} ns")
#         return hidden_states

#     # @partial(nnx.jit, static_argnames=['temperature', 'top_k', 'top_p'])
#     def sample_logits(self, hidden_state: np.ndarray, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> np.ndarray:
#         logits_processor = LogitProcessorList([TemperatureLogitProcessor(temperature), TopKLogitProcessor(top_k), TopPLogitProcessor(top_p)])
#         hidden_state = jnp.array(hidden_state)
#         logits = self.lm_head(
#             hidden_state[:, -1:, :]
#         )  ## Keep only the logits from last token as only that is required for generation.
#         logits = np.squeeze(logits)
#         logits = logits_processor(logits, [])
#         probs = nnx.softmax(logits, axis=-1)
#         key = jax.random.PRNGKey(0)
#         next_tokens = jax.random.choice(
#             key, a=jnp.arange(probs.shape[-1]), p=probs, shape=(1,)
#         ).squeeze(0)
#         return np.array(next_tokens)

@jax.jit
def test_attn_jax_jit(q, k, v):
    return dot_product_attention(q, k, v, bias=None, mask=None, is_causal=True)

@nnx.jit
def test_attn_nnx_jit(q, k, v):
    return dot_product_attention(q, k, v, bias=None, mask=None, is_causal=True)

def test_attn_no_jit(q, k, v):
    return dot_product_attention(q, k, v, bias=None, mask=None, is_causal=True)

@jax.jit
def rms_norm(hidden_state: jax.Array, weight: jax.Array, eps: jax.Array):
    input_dtype = hidden_state.dtype
    variance = lax.convert_element_type(hidden_state, jnp.float32)
    variance = lax.square(variance)
    variance = variance.mean(-1, keepdims=True)
    # use `lax.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
    hidden_state = hidden_state / lax.sqrt(variance + eps)
    return lax.convert_element_type(weight * hidden_state, input_dtype)

@jax.jit
def _linear(
        input: jax.Array,
        weight: jax.Array,
        ):
    return lax.dot_general(input, weight, (((input.ndim - 1,), (0,)), ((), ())))


# _split_query: callable
# _split_kv: callable

def generate_splitter(num_heads: int) -> callable:
    func = eval(f"jax.jit(lambda arr : jax.lax.reshape(arr, (*arr.shape[:2], {num_heads}, arr.shape[-1] // {num_heads})))")
    return func


# @partial(nnx.jit, static_argnames=['num_heads'])
# def _split_query(arr: jax.Array, num_heads: int):
#     return jax.lax.reshape(arr, (*arr.shape[:2], num_heads, arr.shape[-1] // num_heads))

# @partial(nnx.jit, static_argnames=['num_heads'])
# def _split_kv(arr: jax.Array, num_heads: int):
#     return jax.lax.reshape(arr, (*arr.shape[:2], num_heads, arr.shape[-1] // num_heads))

@jax.jit
def _merge_heads(hidden_states: jax.Array):
    return hidden_states.reshape(hidden_states.shape[:2] + (-1,))

@jax.jit ## Reduces from 0.4 -> 0.3 but also increases the eps to e-5 from e-6. Revisit.
def mlp(hidden_states: jax.Array, up_proj: jax.Array, gate_proj: jax.Array, down_proj: jax.Array):
    up = _linear(hidden_states, up_proj)
    gate = _linear(hidden_states, gate_proj)
    return _linear(act_fn_jit(gate) * up, down_proj)

# def lax_norm(tensor: jax.Array, weight: jax.Array, eps: jax.Array):
#     orig_type = tensor.dtype
#     variance = lax.convert_element_type(tensor, jnp.float32)
#     variance = lax.square(tensor)
#     variance = variance.mean(-1, keepdims=True)
#     variance = lax.add(variance,eps)
#     variance = lax.sqrt(variance)
#     tensor = lax.div(tensor, variance)
#     out = tensor * weight ## TODO Check lax impl
#     return lax.convert_element_type(out, orig_type)

# def lax_linear(input: jax.Array, weight: jax.Array):
#     return lax.dot_general(input, weight, (((input.ndim - 1,), (0,)), ((), ())))

# def lax_rotate_half(tensor: jax.Array):
#     mid = tensor.shape[-1] // 2
#     upper_half = jax.lax.neg(jax.lax.slice_in_dim(tensor, start_index=mid, limit_index=tensor.shape[-1], axis=-1))
#     lower_half = jax.lax.slice_in_dim(tensor, start_index=0, limit_index=mid, axis=-1)
#     return lax.concatenate((upper_half, lower_half), dimension=tensor.ndim-1)

# def lax_rotary_embed(tensor: jax.Array, sin: jax.Array, cos: jax.Array):
#     orig_type = tensor.dtype
#     rotated = lax_rotate_half(tensor)
#     sin = lax.expand_dims(sin, [2])
#     cos = lax.expand_dims(cos, [2])
#     return jax.lax.convert_element_type(jax.lax.add(tensor * cos, rotated * sin), orig_type)

# def lax_attention(
#         hidden_state: jax.Array,
#         k_proj: jax.Array,
#         o_proj: jax.Array,
#         q_proj: jax.Array,
#         v_proj: jax.Array,
#         sin: jax.Array,
#         cos: jax.Array,
#         num_kv_heads: int,
#         num_attention_heads: int,
# ):
#     q = lax_linear(hidden_state, q_proj)
#     k = lax_linear(hidden_state, k_proj)
#     v = lax_linear(hidden_state, v_proj)

#     q = jax.lax.reshape(q, (*hidden_state.shape[:2], num_attention_heads, q.shape[-1] // num_attention_heads))
#     k = jax.lax.reshape(k, (*hidden_state.shape[:2], num_kv_heads, k.shape[-1] // num_kv_heads))
#     v = jax.lax.reshape(v, (*hidden_state.shape[:2], num_kv_heads, v.shape[-1] // num_kv_heads))

#     q = lax_rotary_embed(q, sin, cos)
#     k = lax_rotary_embed(k, sin, cos)
#     return jax.nn.dot_product_attention(q,k,v,is_causal=True)

# def lax_layer(
#         hidden_state: jax.Array,
#         input_layernorm: jax.Array,
#         down_proj: jax.Array,
#         gate_proj: jax.Array,
#         up_proj: jax.Array,
#         post_attention_layernorm: jax.Array,
#         k_proj: jax.Array,
#         o_proj: jax.Array,
#         q_proj: jax.Array,
#         v_proj: jax.Array,
#         sin: jax.Array,
#         cos: jax.Array,
#         epsilon: jax.Array,
#         num_kv_heads: int,
#         num_attention_heads: int
# ):
#     hidden_state = lax.convert_element_type(hidden_state, jax.dtypes.bfloat16)
#     epsilon = lax.convert_element_type(epsilon, jax.dtypes.bfloat16)
#     hidden_state = lax_norm(hidden_state, input_layernorm, epsilon)
#     residual = hidden_state
#     return lax_attention(hidden_state, k_proj,o_proj,q_proj,v_proj,sin,cos, num_kv_heads, num_attention_heads)

@jax.jit
def causal_attention(q,k,v):
    return jit_attention(q,k,v,is_causal=True,implementation="cudnn")

@jax.jit
def single_token_attention(q,k,v):
    return jit_attention(q,k,v,is_causal=False,implementation="cudnn")

@partial(jax.jit, static_argnames=['num_kv_heads', 'num_attention_heads'])
def pre_cache(
    hidden_state: jax.Array,
    input_layernorm: jax.Array,
    epsilon: jax.Array,
    q_proj: jax.Array,
    k_proj: jax.Array,
    v_proj: jax.Array,
    sin: jax.Array,
    cos: jax.Array,
    num_kv_heads: int,
    num_attention_heads: int,
):
    hidden_state = jax.lax.convert_element_type(hidden_state, jax.dtypes.bfloat16)
    residual = hidden_state
    hidden_state = rms_norm(hidden_state, input_layernorm, epsilon)
    q = _linear(hidden_state, q_proj)
    k = _linear(hidden_state, k_proj)
    v = _linear(hidden_state, v_proj)
    q = _split_query(q, num_attention_heads)
    k = _split_kv(k, num_kv_heads)
    v = _split_kv(v, num_kv_heads)
    q,k = apply_rotary_embed(q, k, sin, cos)
    return q,k,v,residual

@jax.jit
def post_cache(
    attn_out: jax.Array,
    residual: jax.Array,
    o_proj: jax.Array,
    post_attention_layernorm: jax.Array,
    epsilon: jax.Array,
    down_proj: jax.Array,
    gate_proj: jax.Array,
    up_proj: jax.Array,
):
    attn_out = _merge_heads(attn_out)
    attn_out = _linear(attn_out, o_proj)
    hidden_state = residual + attn_out
    residual = hidden_state
    hidden_state = rms_norm(hidden_state, post_attention_layernorm, epsilon)
    hidden_state = mlp(hidden_state, up_proj, gate_proj, down_proj)
    return hidden_state + residual

# @partial(jax.jit, static_argnames=['num_kv_heads', 'num_attention_heads'])
def test_layer(
        hidden_state: jax.Array,
        input_layernorm: jax.Array,
        epsilon: jax.Array,
        q_proj: jax.Array,
        k_proj: jax.Array,
        v_proj: jax.Array,
        sin: jax.Array,
        cos: jax.Array,
        kv_cache: jax.Array,
        o_proj: jax.Array,
        post_attention_layernorm: jax.Array,
        down_proj: jax.Array,
        gate_proj: jax.Array,
        up_proj: jax.Array,
        num_kv_heads: int,
        num_attention_heads: int,
):
    # epsilon = epsilon.astype(jax.dtypes.bfloat16)
    q,k,v,residual = pre_cache(hidden_state,
        input_layernorm,
        epsilon,
        q_proj,
        k_proj,
        v_proj,
        sin,
        cos,
        num_kv_heads,
        num_attention_heads
    )
    

    ## Use cache
    kv = lax.concatenate(
        (k, v),
        dimension=0
    )
    kv_cache = lax.concatenate((kv_cache, kv), dimension=1)
    k = kv_cache[0, ...]
    v = kv_cache[1, ...]

    # attn_out = jax.lax.cond(hidden_state.shape[-3] > 1, causal_attention, single_token_attention, q, k, v)
    attn_out = jit_attention(q,k,v, is_causal=True, implementation="cudnn")
    hidden_state = post_cache(attn_out, residual, o_proj, post_attention_layernorm, epsilon, down_proj, gate_proj, up_proj)
    return (hidden_state, kv_cache)

@nnx.jit
class ModelLayer(nnx.Module):

    kv_cache = nnx.Param(jnp.zeros((2, 0, 8, 64), dtype=jax.dtypes.bfloat16))

    # def __init__(
    #     self,
    #     num_kv_heads: int,
    #     num_attention_heads: int
    # ):
    #     self.num_attention_heads.value = num_attention_heads
    #     self.num_kv_heads.value = num_kv_heads
        
    
    def clear_cache(self):
        self.kv_cache.value = jnp.zeros((2, 0, 8, 64), dtype=jax.dtypes.bfloat16)
    
    def __call__(self,
        hidden_state: jax.Array,
        input_layernorm: jax.Array,
        epsilon: jax.Array,
        q_proj: jax.Array,
        k_proj: jax.Array,
        v_proj: jax.Array,
        sin: jax.Array,
        cos: jax.Array,
        o_proj: jax.Array,
        post_attention_layernorm: jax.Array,
        down_proj: jax.Array,
        gate_proj: jax.Array,
        up_proj: jax.Array
    ):
        hidden_state = jax.lax.convert_element_type(hidden_state, jax.dtypes.bfloat16)
        residual = hidden_state
        hidden_state = rms_norm(hidden_state, input_layernorm, epsilon)
        q = _linear(hidden_state, q_proj)
        k = _linear(hidden_state, k_proj)
        v = _linear(hidden_state, v_proj)
        q = _split_query(q)
        k = _split_kv(k)
        v = _split_kv(v)
        q,k = apply_rotary_embed(q, k, sin, cos)

        ## Use cache
        kv = lax.concatenate(
            (k, v),
            dimension=0
        )
        self.kv_cache.value = lax.concatenate((self.kv_cache.value, kv), dimension=1)
        k = self.kv_cache.value[0, ...]
        v = self.kv_cache.value[1, ...]
        attn_out = jit_attention(q,k,v, is_causal=True, implementation="cudnn")
        attn_out = _merge_heads(attn_out)
        attn_out = _linear(attn_out, o_proj)
        hidden_state = residual + attn_out
        residual = hidden_state
        hidden_state = rms_norm(hidden_state, post_attention_layernorm, epsilon)
        hidden_state = mlp(hidden_state, up_proj, gate_proj, down_proj)
        return hidden_state + residual
    
@jax.jit
def model_test_pre_cache(
        hidden_state: jax.Array,
        input_layernorm: jax.Array,
        epsilon: jax.Array,
        q_proj: jax.Array,
        k_proj: jax.Array,
        v_proj: jax.Array,
        sin: jax.Array,
        cos: jax.Array
):
    hidden_state = jax.lax.convert_element_type(hidden_state, jax.dtypes.bfloat16)
    residual = hidden_state
    hidden_state = rms_norm(hidden_state, input_layernorm, epsilon)
    q = _linear(hidden_state, q_proj)
    k = _linear(hidden_state, k_proj)
    v = _linear(hidden_state, v_proj)
    q = _split_query(q)
    k = _split_kv(k)
    v = _split_kv(v)
    q,k = apply_rotary_embed(q, k, sin, cos)
    return residual, q, k, v

@jax.jit
def model_test_post_cache(
        residual: jax.Array,
        attn_out: jax.Array,
        epsilon: jax.Array,
        o_proj: jax.Array,
        post_attention_layernorm: jax.Array,
        down_proj: jax.Array,
        gate_proj: jax.Array,
        up_proj: jax.Array
):
    attn_out = _merge_heads(attn_out)
    attn_out = _linear(attn_out, o_proj)
    hidden_state = residual + attn_out
    residual = hidden_state
    hidden_state = rms_norm(hidden_state, post_attention_layernorm, epsilon)
    hidden_state = mlp(hidden_state, up_proj, gate_proj, down_proj)
    return hidden_state + residual
    
class JITLlamaModel():

    layers: list[ModelLayer] = []
    jit_layers: list[callable] = []
    kv_cache: nnx.Param = nnx.Param([])
    shard: Shard
    config: LlamaConfig
    eps: jax.Array = jnp.array(1e-5)
    model_norm: jax.Array
    hidden_state: jax.Array
    input_layernorm: jax.Array
    post_attention_layernorm: jax.Array
    q_proj: jax.Array
    k_proj: jax.Array
    v_proj: jax.Array
    o_proj: jax.Array
    down_proj: jax.Array
    gate_proj: jax.Array
    up_proj: jax.Array

    def __init__(self):
        pass

    def load_partial(self, shard: Shard, st_path: str, config: LlamaConfig):
        config.num_key_value_heads
        config.num_attention_heads
        self.config = config
        self.hidden_size = config.hidden_size
        self.shard = shard
        self.eps = jnp.array(config.rms_norm_eps)
        self.num_layers = shard.get_layer_count()
        
        if shard.is_first_layer() or shard.is_last_layer():
            self.embeddings = nnx.Embed(num_embeddings=config.vocab_size, features=config.hidden_size, rngs=nnx.Rngs(0))
            self.lm_head = nnx.jit((self.embeddings.attend))
            self.embeddings = nnx.jit(self.embeddings)

        # for _ in range(shard.start_layer, shard.end_layer + 1):
        #     layer_module = ModelLayer()
        #     self.layers.append(layer_module)
        
        # for layer in self.layers:
        #     self.jit_layers.append(nnx.jit(layer))
        
        self.kv_cache.value = [jnp.zeros((2, 0, 8, 64), dtype=jax.dtypes.bfloat16) for _ in range(shard.start_layer, shard.end_layer + 1)]

        def concat_weights(st, key, start, end, dense: bool = True) -> jax.Array:
            return jax.lax.concatenate([lax.expand_dims(convert_from_pt(st.get_tensor(f"model.layers.{i}.{key}.weight"), dense), [0]) for i in range(start, end)], 0)

            weights = convert_from_pt(st.get_tensor(f"model.layers.{start}.{key}.weight"), dense)
            weights = lax.expand_dims(weights, [0])
            for i in range(start +1 , end):
                layer_weights = convert_from_pt(st.get_tensor(f"model.layers.{i}.{key}.weight"), dense)
                layer_weights = lax.expand_dims(layer_weights, [0])
                weights = lax.concatenate((weights, layer_weights), 0)
            assert weights.shape[0] == end - start
            return weights

        ## Load weights
        print("Loading model weight")
        with safe_open(st_path, framework="pt") as st:
            if self.embeddings is not None:
                self.embeddings.embedding.value = convert_from_pt(st.get_tensor("model.embed_tokens.weight"))
            self.input_layernorm = concat_weights(st, "input_layernorm", shard.start_layer, shard.end_layer + 1, dense=False)
            self.down_proj = concat_weights(st, "mlp.down_proj", shard.start_layer, shard.end_layer + 1)
            self.gate_proj = concat_weights(st, "mlp.gate_proj", shard.start_layer, shard.end_layer + 1)
            self.up_proj = concat_weights(st, "mlp.up_proj", shard.start_layer, shard.end_layer + 1)
            self.post_attention_layernorm = concat_weights(st, "post_attention_layernorm", shard.start_layer, shard.end_layer + 1, dense=False)
            self.q_proj = concat_weights(st, "self_attn.q_proj", shard.start_layer, shard.end_layer + 1)
            self.k_proj = concat_weights(st, "self_attn.k_proj", shard.start_layer, shard.end_layer + 1)
            self.v_proj = concat_weights(st, "self_attn.v_proj", shard.start_layer, shard.end_layer + 1)
            self.o_proj = concat_weights(st, "self_attn.o_proj", shard.start_layer, shard.end_layer + 1)
            if shard.is_last_layer():
                self.model_norm = convert_from_pt(st.get_tensor("model.norm.weight"))
        print("Model weights loaded")

    def reset_cache(self):
        self.kv_cache.value = [jnp.zeros((2, 0, 8, 64), dtype=jax.dtypes.bfloat16) for _ in range(self.num_layers)]

    def jit_call(
            self,
            hidden_state: jax.Array,
            input_layernorm: jax.Array,
            epsilon: jax.Array,
            q_proj: jax.Array,
            k_proj: jax.Array,
            v_proj: jax.Array,
            sin: jax.Array,
            cos: jax.Array,
            o_proj: jax.Array,
            post_attention_layernorm: jax.Array,
            down_proj: jax.Array,
            gate_proj: jax.Array,
            up_proj: jax.Array
    ):
        pass

    def use_kv_cache(self, k, v, layer_idx):
        updating_cache = self.kv_cache.value
        kv = lax.concatenate(
            (k, v),
            dimension=0
        )
        layer_cache = updating_cache[layer_idx]
        layer_cache = lax.concatenate((layer_cache, kv), dimension=1)
        k = layer_cache[0, ...]
        v = layer_cache[1, ...]
        updating_cache[layer_idx] = layer_cache
        self.kv_cache.value = updating_cache
        return k,v

    def __call__(self,
        hidden_state: jax.Array,
        sin: jax.Array,
        cos: jax.Array
    ):
        # if hidden_state.shape[-1] == self.hidden_size and len(hidden_state.shape) == 3:
        ## TODO: Maintain the cache state and also generate the rotary embeddings based on the req id
        
        # print(self.kv_cache.value)
        hidden_state = self.embeddings(hidden_state)
        updating_cache = self.kv_cache.value
        for layer_idx in range(self.num_layers):
            residual, q, k, v = model_test_pre_cache(
                hidden_state,
                input_layernorm = self.input_layernorm[layer_idx, ...],
                epsilon = self.eps,
                q_proj = self.q_proj[layer_idx, ...],
                k_proj = self.k_proj[layer_idx, ...],
                v_proj = self.v_proj[layer_idx, ...],
                sin = sin,
                cos = cos
            )

            kv = lax.concatenate(
                (k, v),
                dimension=0
            )
            layer_cache = updating_cache[layer_idx]
            layer_cache = lax.concatenate((layer_cache, kv), dimension=1)
            k = layer_cache[0, ...]
            v = layer_cache[1, ...]
            updating_cache[layer_idx] = layer_cache
            # k,v = self.use_kv_cache(k, v, layer_idx)
            
            attn_out = dot_product_attention(q,k,v, is_causal=True, implementation="cudnn")
            hidden_state = model_test_post_cache(
                residual=residual,
                attn_out=attn_out,
                epsilon=self.eps,
                o_proj = self.o_proj[layer_idx, ...],
                post_attention_layernorm = self.post_attention_layernorm[layer_idx, ...],
                down_proj = self.down_proj[layer_idx, ...],
                gate_proj = self.gate_proj[layer_idx, ...],
                up_proj = self.up_proj[layer_idx, ...]
            )

            # hidden_state = self.jit_layers[layer_idx](
            #     hidden_state,
            #     input_layernorm = self.input_layernorm[layer_idx, ...],
            #     epsilon = self.eps,
            #     q_proj = self.q_proj[layer_idx, ...],
            #     k_proj = self.k_proj[layer_idx, ...],
            #     v_proj = self.v_proj[layer_idx, ...],
            #     sin = sin,
            #     cos = cos,
            #     o_proj = self.o_proj[layer_idx, ...],
            #     post_attention_layernorm = self.post_attention_layernorm[layer_idx, ...],
            #     down_proj = self.down_proj[layer_idx, ...],
            #     gate_proj = self.gate_proj[layer_idx, ...],
            #     up_proj = self.up_proj[layer_idx, ...]
            # )
        if self.shard.is_last_layer():
            hidden_state = rms_norm(hidden_state, self.model_norm, self.eps)
        print(self.kv_cache.value)
        self.kv_cache.value = updating_cache
        return hidden_state