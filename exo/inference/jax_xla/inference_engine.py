from ..inference_engine import InferenceEngine
from ..shard import Shard
import numpy as np

class JAXShardedInferenceEngine(InferenceEngine):

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        pass
    
    async def sample(self, x: np.ndarray) -> np.ndarray:
        pass

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        pass

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray) -> np.ndarray:
        pass

    async def ensure_shard():
        pass