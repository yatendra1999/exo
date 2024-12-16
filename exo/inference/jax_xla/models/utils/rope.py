import jax.numpy as jnp
import math
from typing import Tuple, Optional

from transformers import PretrainedConfig
from transformers.models.llama import LlamaConfig

def compute_llama3_parameters(
    config: LlamaConfig
) -> Tuple[jnp.ndarray, float]:
    """
    Computes the inverse frequencies for llama 3.1 using JAX.

    Args:
        config ([`~transformers.models.llama.LlamaConfig`]):
            The llama 3 model configuration.

    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq

    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

    is_medium_freq = jnp.logical_and(wavelen >= high_freq_wavelen, wavelen <= low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
) -> Tuple[jnp.ndarray, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation using JAX.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """

    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq, attention_factor