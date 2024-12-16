from flax import nnx
from jax import numpy as jnp
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