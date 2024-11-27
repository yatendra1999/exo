from transformers.models.llama.modeling_flax_llama import *
from transformers.models.llama.modeling_llama import LlamaDecoderLayer as TorchLlamaDecoderLayer
from jax import numpy as jnp
import jax
from flax import nnx
from flax.nnx import make_causal_mask, combine_masks


ACT_MAP: dict[str, callable] = {
    'silu': nnx.swish
}

# Helper functions for attention

def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)

class LlamaRotaryEmbedding(nnx.Module):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype = jnp.float32):
        self.config = config
        self.dtype = dtype

        # Compute the sinusoidal positional embeddings during initialization
        head_dim = config.hidden_size // config.num_attention_heads
        self.sincos = create_sinusoidal_positions(config.max_position_embeddings, head_dim)

    def __call__(self, key, query, position_ids):
        # Retrieve sin and cos positional embeddings based on position_ids
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        # Apply rotary positional embedding to key and query
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        # Ensure outputs are of the correct dtype
        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query

class VariableCache(nnx.Variable):
    pass

# Define nnx-based Llama Attention
class LlamaAttention(nnx.Module):
    def __init__(self, config: LlamaConfig, weights: dict[str,jax.Array], rngs: nnx.rnglib.Rngs, dtype=jnp.float32, causal=True, is_cross_attention=False):
        self.config = config
        self.dtype = dtype
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.attention_softmax_in_fp32 = dtype is not jnp.float32
        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool"
        )

        self.q_proj = nnx.Linear(config.hidden_size, self.num_heads * self.head_dim, kernel_init=lambda x, y, z : weights['q'], rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, kernel_init=lambda x, y, z : weights['k'], rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, kernel_init=lambda x, y, z : weights['v'], rngs=rngs)
        self.o_proj = nnx.Linear(self.num_heads * self.head_dim, config.hidden_size, kernel_init=lambda x, y, z : weights['o'], rngs=rngs)

        self.rotary_emb = LlamaRotaryEmbedding(config, dtype=dtype)

        # Caching for autoregressive decoding
        self.cache = nnx.State({
            "cached_key": jnp.zeros,
            "cached_value": jnp.zeros,
            "cache_index": lambda: jnp.array(0, dtype=jnp.int32)
        })

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, -1))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (-1,))

    def __call__(self, hidden_states, attention_mask, position_ids, deterministic=True, init_cache=False, output_attentions=False):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Split heads
        query = self._split_heads(query, self.config.num_attention_heads)
        key = self._split_heads(key, self.config.num_key_value_heads)
        value = self._split_heads(value, self.config.num_key_value_heads)

        # Apply rotary embeddings
        key, query = self.rotary_emb(key, query, position_ids)

        # Handle causal masking
        query_length, key_length = query.shape[1], key.shape[1]
        causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = combine_masks(attention_mask, causal_mask)

        # Apply caching for autoregressive decoding
        if init_cache or self.cache.cached_key is not None:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)

        # Attention bias
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # Compute attention weights
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=None if deterministic else self.make_rng("dropout"),
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        )

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        # Compute attention outputs
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = hasattr(self, "cached_key")
        if not is_initialized:
            self.cached_key = VariableCache(jnp.zeros(key.shape, key.dtype))
            self.cached_value = VariableCache(jnp.zeros(value.shape, value.dtype))
            self.cache_index = VariableCache(jnp.array(0, dtype=jnp.int32))
        else:
            *batch_dims, max_length, num_heads, depth_per_head = self.cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = self.cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
            value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
            self.cached_key.value = key
            self.cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            self.cache_index.value = self.cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask
    
class LlamaRMSNorm(nnx.Module):

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

class LlamaMLP(nnx.Module):
    
    def __init__(self, config: LlamaConfig, weights_map: dict[str, jax.Array], rng: nnx.rnglib.Rngs):
        self.up_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias, kernel_init=lambda x,y,z : weights_map['up'], rngs=rng)
        self.gate_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias, kernel_init=lambda x,y,z : weights_map['gate'], rngs=rng)
        self.down_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias, kernel_init=lambda x,y,z : weights_map['down'], rngs=rng)
        self.activation_fn = ACT_MAP[config.hidden_act]
         

    def __call__(self, hidden_states: jax.Array):
        ### Ignoring the values of pretraining_tp > 1. Find more details here: https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig.pretraining_tp
    
        return self.down_proj( self.activation_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

class LlamaDecoderLayer(nnx.Module):

    def __init__(self, config: LlamaConfig, weights_map: dict, rngs: nnx.rnglib.Rngs):

        config.max_position_embeddings = 1024 ## Hardcoded right now for testing and avoiding OOM kills.

        ## Add option to load the tensor weights from the safetensor file and then use those to init the layer component weights
        self.input_layernorm = LlamaRMSNorm(config, weights_map['input_layernorm'])
        self.self_attn = LlamaAttention(config, weights_map['self_attn'], rngs)
        self.post_attention_layernorm = LlamaRMSNorm(config, weights_map['post_attention_layernorm'])
        self.mlp = LlamaMLP(config, weights_map['mlp'], rngs)

    @classmethod
    def from_torch(cls, config: LlamaConfig, layer: TorchLlamaDecoderLayer):

        rngs = nnx.rnglib.Rngs(0)
        weights_map = {
            "input_layernorm": layer.input_layernorm.weight.detach().numpy(),
            "self_attn": {
                'q': layer.self_attn.q_proj.weight.detach().numpy().transpose(),
                'k': layer.self_attn.k_proj.weight.detach().numpy().transpose(),
                'v': layer.self_attn.v_proj.weight.detach().numpy().transpose(),
                'o': layer.self_attn.o_proj.weight.detach().numpy().transpose()
            },
            "post_attention_layernorm" : layer.post_attention_layernorm.weight.detach().numpy(),
            "mlp": {
                'gate': layer.mlp.gate_proj.weight.detach().numpy().transpose(),
                'up' : layer.mlp.up_proj.weight.detach().numpy().transpose(),
                'down': layer.mlp.down_proj.weight.detach().numpy().transpose()
            }
        }

        return cls(config, weights_map, rngs)
    
    @classmethod
    def from_safetensors(cls, file_path: str, layer_idx: int):
        """
        TODO : Init the model weights directly from the safetensors file.
        """
        pass

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        return (hidden_states,) + outputs[1:]

# if __name__ == "__main__":
#     from jax import numpy as jnp
#     def flaxtensor(x):
#         return jnp.array(x.detach().numpy())
#     flax_layers = LlamaDecoderLayer.from_torch(self.config, decoder_layer)
#     flax_out = flax_layers(flaxtensor(hidden_states), attention_mask=causal_mask,
#                         position_ids=flaxtensor(position_ids))