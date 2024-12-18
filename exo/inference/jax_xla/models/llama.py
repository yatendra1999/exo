from concurrent.futures import ThreadPoolExecutor

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


ACT_MAP: dict[str, callable] = {"silu": nnx.swish}


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]),
        axis=-1,
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class LlamaRotaryEmbedding(nnx.Module):

    def __init__(self, config: LlamaConfig):
        
        self.inv_freq, self.attention_scaling = compute_llama3_parameters(config)

    def create_embed(self, position_ids):
        inv_freq_expanded = jnp.expand_dims(self.inv_freq, (0, 2))
        position_ids_expanded = jnp.expand_dims(position_ids, (1))

        freq = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        freq = jnp.append(freq, freq, axis=-1)

        self.cos = jnp.cos(freq) * self.attention_scaling
        self.sin = jnp.sin(freq) * self.attention_scaling

        return self.cos, self.sin

    def __call__(self, query, key, expand_axis = 2):
        cos = jnp.expand_dims(self.cos, axis=expand_axis)
        sin = jnp.expand_dims(self.sin, axis=expand_axis)
        query_embed = (query * cos) + (rotate_half(query) * sin)
        key_embed = (key * cos) + (rotate_half(key) * sin)

        return query_embed, key_embed

rotary_embedding = None

class VariableCache(nnx.Variable):
    pass


# Define nnx-based Llama Attention
class LlamaAttention(FlaxBaseModule):
    def __init__(
        self,
        config: LlamaConfig,
        weights: dict[str, jax.Array],
        rngs: nnx.rnglib.Rngs,
        dtype=jnp.float32,
    ):
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

        self.q_proj = nnx.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.q_proj.kernel.value = weights['q']
        
        self.k_proj = nnx.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.k_proj.kernel.value = weights['k']

        self.v_proj = nnx.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.v_proj.kernel.value = weights["v"]

        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            rngs=rngs,
        )
        self.o_proj.kernel.value = weights["o"]

        self.cache = nnx.State(
            {
                "cached_key": jnp.zeros,
                "cached_value": jnp.zeros,
                "cache_index": lambda: jnp.array(0, dtype=jnp.int32),
            }
        )

    @classmethod
    def from_safetensor(cls, config, key: str, path: str, framework: str):
        rngs = nnx.Rngs(0)
        with safe_open(path, framework=framework) as st:
            weights = {
                "q": cls.convert_from_pt(st.get_tensor(f"{key}.q_proj.weight")),
                "k": cls.convert_from_pt(st.get_tensor(f"{key}.k_proj.weight")),
                "v": cls.convert_from_pt(st.get_tensor(f"{key}.v_proj.weight")),
                "o": cls.convert_from_pt(st.get_tensor(f"{key}.o_proj.weight")),
            }
        return cls(config=config, rngs=rngs, weights=weights)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, -1))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (-1,))

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Split heads
        query = self._split_heads(query, self.config.num_attention_heads)
        key = self._split_heads(key, self.config.num_key_value_heads)
        value = self._split_heads(value, self.config.num_key_value_heads)

        # Apply rotary embeddings
        global rotary_embedding
        query, key = rotary_embedding(query, key)
        
        # Apply caching for autoregressive decoding
        key, value, attention_mask = self._concatenate_to_cache(
            key, value, query, attention_mask
        )

        # Broadcast Key and Value to match Query shape
        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)

        is_causal = True if position_ids.shape[-1] > 1 else False

        # Calculate attention values
        attn_weights = dot_product_attention(query, key, value, bias=None, is_causal=is_causal, mask=None)
        attn_output = self._merge_heads(attn_weights)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = hasattr(self, "cached_key")
        if not is_initialized:
            self.cached_key = VariableCache(key)
            self.cached_value = VariableCache(value)
            self.cache_index = VariableCache(key.shape[-3])
        else:
            *batch_dims, max_length, num_heads, depth_per_head = (
                self.cached_key.value.shape
            )
            cur_index = self.cache_index.value
            key = jnp.append(self.cached_key.value, key, axis=-3)
            value = jnp.append(self.cached_value.value, value, axis=-3)
            self.cached_key.value = key
            self.cached_value.value = value
            num_updated_cache_vectors = query.shape[-3]
            self.cache_index.value = self.cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask


class LlamaRMSNorm(FlaxBaseModule):

    def __init__(self, config: LlamaConfig, weights: jax.Array):
        self.weights = nnx.Param(weights)
        self.epsilon = nnx.Param(config.rms_norm_eps)

    def __call__(self, hidden_states: jax.Array):
        input_dtype = hidden_states.dtype
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weights * jnp.asarray(hidden_states, dtype=input_dtype)

    @classmethod
    def from_safetensor(cls, config, key, path, framework):
        with safe_open(path, framework=framework) as st:
            weights = st.get_tensor(f"{key}.weight")
        weights = cls.convert_from_pt(weights)
        return cls(config, weights)


class LlamaMLP(FlaxBaseModule):

    def __init__(
        self,
        config: LlamaConfig,
        weights_map: dict[str, jax.Array],
        rng: nnx.rnglib.Rngs,
    ):
        self.up_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=config.mlp_bias,
            rngs=rng,
        )
        self.up_proj.kernel.value = weights_map["up"]

        self.gate_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=config.mlp_bias,
            rngs=rng,
        )
        self.gate_proj.kernel.value = weights_map["gate"]

        self.down_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=config.mlp_bias,
            rngs=rng,
        )
        self.down_proj.kernel.value = weights_map["down"]
        self.activation_fn = ACT_MAP[config.hidden_act]

    def __call__(self, hidden_states: jax.Array):
        ### Ignoring the values of pretraining_tp > 1. Find more details here: https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig.pretraining_tp

        return self.down_proj(
            self.activation_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )

    @classmethod
    def from_safetensor(cls, config, key, path, framework):
        with safe_open(path, framework=framework) as st:
            weights = {
                "up": cls.convert_from_pt(st.get_tensor(f"{key}.up_proj.weight")),
                "gate": cls.convert_from_pt(st.get_tensor(f"{key}.gate_proj.weight")),
                "down": cls.convert_from_pt(st.get_tensor(f"{key}.down_proj.weight")),
            }
        return cls(config, weights, rng=nnx.Rngs(0))


class LlamaDecoderLayer(FlaxBaseModule):

    def __init__(
        self,
        config: LlamaConfig,
        safetensor_path=None,
        safetensor_key: str = None,
        weights_map: dict = None,
        rngs: nnx.rnglib.Rngs = nnx.Rngs(0),
    ):

        config.max_position_embeddings = (
            1024  ## Hardcoded right now for testing and avoiding OOM kills.
        )

        if weights_map is not None:
            if all(
                [
                    x in weights_map
                    for x in [
                        "input_layernorm",
                        "self_attn",
                        "post_attention_layernorm",
                        "mlp",
                    ]
                ]
            ):
                self.input_layernorm = LlamaRMSNorm(
                    config, weights_map["input_layernorm"]
                )
                self.self_attn = LlamaAttention(config, weights_map["self_attn"], rngs)
                self.post_attention_layernorm = LlamaRMSNorm(
                    config, weights_map["post_attention_layernorm"]
                )
                self.mlp = LlamaMLP(config, weights_map["mlp"], rngs)
                return
            else:
                raise Exception("Weights provided do not contain all required layers.")

        if safetensor_path is None or safetensor_key is None:
            raise Exception(
                "Both safetensor_path and safetensor_key are required to init layer from safetensors file."
            )

        self.input_layernorm = LlamaRMSNorm.from_safetensor(
            config, f"{safetensor_key}.input_layernorm", safetensor_path, framework="pt"
        )
        self.self_attn = LlamaAttention.from_safetensor(
            config, f"{safetensor_key}.self_attn", safetensor_path, framework="pt"
        )
        self.post_attention_layernorm = LlamaRMSNorm.from_safetensor(
            config,
            f"{safetensor_key}.post_attention_layernorm",
            safetensor_path,
            framework="pt",
        )
        self.mlp = LlamaMLP.from_safetensor(
            config, f"{safetensor_key}.mlp", safetensor_path, framework="pt"
        )

    @classmethod
    def from_torch(cls, config: LlamaConfig, layer: TorchLlamaDecoderLayer):

        rngs = nnx.rnglib.Rngs(0)
        weights_map = {
            "input_layernorm": layer.input_layernorm.weight.detach().numpy(),
            "self_attn": {
                "q": layer.self_attn.q_proj.weight.detach().numpy().transpose(),
                "k": layer.self_attn.k_proj.weight.detach().numpy().transpose(),
                "v": layer.self_attn.v_proj.weight.detach().numpy().transpose(),
                "o": layer.self_attn.o_proj.weight.detach().numpy().transpose(),
            },
            "post_attention_layernorm": layer.post_attention_layernorm.weight.detach().numpy(),
            "mlp": {
                "gate": layer.mlp.gate_proj.weight.detach().numpy().transpose(),
                "up": layer.mlp.up_proj.weight.detach().numpy().transpose(),
                "down": layer.mlp.down_proj.weight.detach().numpy().transpose(),
            },
        }

        return cls(config, weights_map, rngs)

    @classmethod
    def from_safetensor(cls, config, key, path, framework, dense=True):
        return cls(config, safetensor_key=key, safetensor_path=path)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaEmbedding(nnx.Embed, FlaxBaseModule):

    @classmethod
    def from_safetensor(cls, config, key, path, framework):
        with safe_open(path, framework=framework) as st:
            weights = st.get_tensor(f"{key}.weight")
        weights = cls.convert_from_pt(weights, dense=False)
        return cls(
            weights.shape[0],
            weights.shape[1],
            embedding_init=lambda x, y, z: weights,
            rngs=nnx.Rngs(0),
        )


class ShardedLlamaModel(FlaxLlmModel):
    embed = None
    norm = None
    layers: list[FlaxBaseModule] = []
    cache_positions: dict[str, int] = {}
    executor: ThreadPoolExecutor
    config: LlamaConfig | None
    shard: Shard | None
    lm_head: nnx.Module

    def __init__(self):
        self.shard = None
        self.config = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _load_model(self):
        if self.shard == None:
            raise Exception("Model attempted to load from an empty shard.")
        if self.config == None:
            raise Exception("Model attempted to load from an empty config.")

        ## TODO
        safetensor_path = cached_file(
            "unsloth/Llama-3.2-1B-Instruct", SAFE_WEIGHTS_NAME
        )

        global rotary_embedding
        rotary_embedding = LlamaRotaryEmbedding(self.config)

        if self.shard.is_first_layer():
            embeddings = LlamaEmbedding.from_safetensor(
                self.config, "model.embed_tokens", safetensor_path, framework="pt"
            )
            self.embed = (embeddings)
            self.lm_head = (embeddings.attend)

        for layer_idx in range(self.shard.start_layer, self.shard.end_layer + 1):
            layer_module = (LlamaDecoderLayer.from_safetensor(
                self.config,
                f"model.layers.{layer_idx}",
                safetensor_path,
                framework="pt",
            ))
            self.layers.append(layer_module)

        if self.shard.is_last_layer():
            module = (LlamaRMSNorm.from_safetensor(
                self.config, f"model.norm", safetensor_path, framework="pt"
            ))
            self.norm = module

    def load_shard(self, config: LlamaConfig, shard: Shard):
        if self.shard == shard:
            return
        self.shard = shard
        if hasattr(self, "layers"):
            self.layers = []
        self.config = config
        self._load_model()

    def generate_args(self, request_id: str, input_shape: tuple[int, ...]) -> dict[str,]:
        if request_id in self.cache_positions:
            start = self.cache_positions[request_id]
        else:
            start = 0
        end = start + input_shape[-1]
        self.cache_positions[request_id] = end
        model_args = {
            # "attention_mask" : jnp.ones(input_shape, dtype=jnp.uint4),
            "attention_mask": None,
            "position_ids": jnp.expand_dims(jnp.arange(start=start, stop=end), axis=0)
        }
        return model_args

    # @partial(nnx.jit, static_argnames=['request_id'])
    def __call__(self, request_id: str, hidden_states: jax.Array) -> jax.Array:

        model_args = self.generate_args(request_id, hidden_states.shape)
        global rotary_embedding
        rotary_embedding.create_embed(model_args['position_ids'])

        if self.embed != None:
            hidden_states = self.embed(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, **model_args)

        if self.norm != None:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    # @partial(nnx.jit, static_argnames=['temperature', 'top_k', 'top_p'])
    def sample_logits(self, hidden_state: np.ndarray, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> np.ndarray:
        logits_processor = LogitProcessorList([TemperatureLogitProcessor(temperature), TopKLogitProcessor(top_k), TopPLogitProcessor(top_p)])
        hidden_state = jnp.array(hidden_state)
        logits = self.lm_head(
            hidden_state[:, -1:, :]
        )  ## Keep only the logits from last token as only that is required for generation.
        logits = np.squeeze(logits)
        logits = logits_processor(logits, [])
        probs = nnx.softmax(logits, axis=-1)
        key = jax.random.PRNGKey(0)
        next_tokens = jax.random.choice(
            key, a=jnp.arange(probs.shape[-1]), p=probs, shape=(1,)
        ).squeeze(0)
        return np.array(next_tokens)


if __name__ == "__main__":
    from transformers.utils import SAFE_WEIGHTS_NAME, cached_file
    from transformers import PretrainedConfig

    model_id = "unsloth/Llama-3.2-1B-Instruct"
    # resolved_archive_file = cached_file(model_id, SAFE_WEIGHTS_NAME)
    # test_shard = Shard(start_layer=0, n_layers=16, end_layer=15, model_id=model_id)
    # engine = ShardedLlamaModel(None)
    config = PretrainedConfig.from_pretrained(model_id, rngs=nnx.Rngs(0))

    position_ids = [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45]]
    
    embedding = LlamaRotaryEmbedding(config)
    embedding.create_embed(position_ids)