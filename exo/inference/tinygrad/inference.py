from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, convert_from_huggingface, fix_bf16, sample_logits
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import load_state_dict
from tinygrad import Tensor, nn, Context
from exo.inference.inference_engine import InferenceEngine
from typing import Optional, Tuple
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
from .stateful_model import StatefulModel
import asyncio
import aiofiles

Tensor.no_grad = True
# default settings
TEMPERATURE = int(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0

async def get_model_config(model_path: Path) -> dict:
  config_path = model_path / "config.json"
  if not config_path.exists():
    raise ValueError(f"Config file not found at {config_path}")

  async with aiofiles.open(config_path) as f:
    config = json.loads(await f.read())

  return {
    "args": {
      "dim": config["hidden_size"],
      "n_heads": config["num_attention_heads"],
      "n_kv_heads": config.get("num_key_value_heads", config["num_attention_heads"]),
      "n_layers": config["num_hidden_layers"],
      "norm_eps": config["rms_norm_eps"],
      "rope_theta": config.get("rope_theta", 500000),
      "vocab_size": config["vocab_size"],
      "hidden_dim": config["intermediate_size"],
      "rope_scaling": config.get("rope_scaling", None),
      "tie_word_embeddings": config.get("tie_word_embeddings", False)
    },
    "files": config.get("num_shards", 1)
  }

async def build_transformer(model_path: Path, shard: Shard, device=None):
  # Get model config from HF config file
  model_config = await get_model_config(model_path)

  # build model
  linear = nn.Linear
  with Context(THREEFRY=0):
    model = Transformer(**model_config["args"], linear=linear, max_context=8192, jit=True, shard=shard)

  # load weights
  if model_path.is_dir():
    if (model_path/"model.safetensors.index.json").exists():
      weights = load(str(model_path/"model.safetensors.index.json"), shard)
    elif (model_path/"model.safetensors").exists():
      weights = load(str(model_path/"model.safetensors"), shard)
    else:
      weights = concat_weights(
        [load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(model_config["files"])],
        device[0] if isinstance(device, tuple) else device
      )
  else:
    weights = load(str(model_path), shard)

  weights = convert_from_huggingface(
    weights,
    model,
    model_config["args"]["n_heads"],
    model_config["args"]["n_kv_heads"]
  )
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=False)  # consume=True
  return model

class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = 0.0) -> np.ndarray:
    logits = x[:, -1, :]
    def sample_wrapper():
      return sample_logits(Tensor(logits).flatten(), temp, 0, 0.8, top_p, 0.0).realize()
    out = await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)
    return out.numpy().astype(int)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return np.array(tokens)

  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)
    return tokens

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray) -> np.ndarray:
    await self.ensure_shard(shard)
    output_data = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.model(Tensor(input_data), request_id).realize())
    return output_data.numpy()

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)

    if self.shard != shard:
      model_shard = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: asyncio.run(build_transformer(model_path, shard)))

      tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
      self.tokenizer = await resolve_tokenizer(tokenizer_path)
      self.shard = shard
      self.model = await asyncio.get_running_loop().run_in_executor(self.executor, StatefulModel, model_shard)
