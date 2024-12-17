from flax import nnx
from jax import numpy as jnp, lax, nn
from abc import abstractmethod

class LogitProcessor(nnx.Module):

    @abstractmethod
    def __call__(self, logits: jnp.ndarray, position_ids: jnp.ndarray) -> jnp.ndarray:
        pass

class LogitProcessorList(LogitProcessor):
    def __init__(self, processors: list[LogitProcessor]):
        self.processors = processors

    def __call__(self, logits, position_ids):
        for processor in self.processors:
            logits = processor(logits, position_ids)
        return logits
    
class TemperatureLogitProcessor(LogitProcessor):
    
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits, position_ids):
        return logits / self.temperature

class TopKLogitProcessor(LogitProcessor):
    
    def __init__(self, top_k: int):
        self.top_k = top_k
        self.fill_value = -1 * jnp.inf

    def __call__(self, logits, position_ids):
        top_k_values, top_k_indices = lax.top_k(logits, self.top_k)
        top_k_values = top_k_values.squeeze()
        top_k_indices = top_k_indices.squeeze()
        mask = jnp.full(logits.shape, self.fill_value)
        mask = mask.at[..., top_k_indices].set(top_k_values)
        return mask
    
class TopPLogitProcessor(LogitProcessor):
    
    def __init__(self, top_p: int):
        self.top_p = top_p
        self.fill_value = -1 * jnp.inf

    def __call__(self, logits, position_ids):
        sorted_logits = jnp.sort(logits)
        sorted_indices = jnp.argsort(logits)

        cumulative_probabilities = nn.softmax(sorted_logits, axis=-1).cumsum(axis=-1)

        indices_to_keep = jnp.where(cumulative_probabilities > (1 - 0.9))[-1]
        orig_indices_to_keep = sorted_indices[..., indices_to_keep]
        orig_values_to_keep = sorted_logits[... , indices_to_keep]
        mask = jnp.full(logits.shape, self.fill_value)
        mask = mask.at[..., orig_indices_to_keep].set(orig_values_to_keep)
        return mask
