import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from jax import numpy as jnp
import jax
from flax import nnx

from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from exo.models import model_cards

from exo.download.shard_download import ShardDownloader
from exo.download.hf.hf_helpers import get_local_snapshot_dir

from ..inference_engine import InferenceEngine

from .config import resolve_config
from .models import FlaxLlmModel

class JAXShardedInferenceEngine(InferenceEngine):

    shard_downloader: ShardDownloader
    current_shard: Shard | None = None
    module: FlaxLlmModel
    model_root_local: Path
    executor: ThreadPoolExecutor
    
    def __init__(self, downloader: ShardDownloader):
        self.shard_downloader = downloader
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        '''Convert the input string into tokenized array.'''
        await self.ensure_shard(shard)
        return np.array(self.tokenizer.encode(prompt))
        tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
        return await asyncio.get_running_loop().run_in_executor(self.executor, np.array, tokens)
    
    async def sample(self, x: np.ndarray) -> np.ndarray:
        '''Sample the logits for the next token.'''
        return self.module.sample_logits(x)
        return await asyncio.get_running_loop().run_in_executor(self.executor, self.module.sample_logits, x)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        '''Convert the model output into actual response string.'''
        await self.ensure_shard(shard)
        return self.tokenizer.decode(tokens)
        return await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray) -> np.ndarray:
        '''Run the model layers on the input array'''
        await self.ensure_shard(shard)
        return self.module(request_id, jnp.array(input_data))
        return await asyncio.get_running_loop().run_in_executor(self.executor, lambda request_id, x : self.module(request_id, jnp.array(x)), request_id, input_data)

    async def ensure_shard(self, shard: Shard) -> None:
        if shard == self.current_shard:
            return
        self.current_shard = shard
        await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
        print(shard.model_id)
        model_repo = model_cards[shard.model_id]['repo'][self.__class__.__name__]
        self.tokenizer = await resolve_tokenizer(model_repo)
        model_root_local = await get_local_snapshot_dir(model_repo)
        config, module_cls = resolve_config(model_root_local)
        self.module = module_cls()
        self.module.load_shard(config, shard)
