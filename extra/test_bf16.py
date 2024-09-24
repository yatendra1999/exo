from typing import Tuple, Union, Optional, Dict, Any
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv


def test_bf16():
  weights = {
    "weight1": Tensor.randn(10, 10, dtype=dtypes.bfloat16),
    "weight2": Tensor.randn(20, 20, dtype=dtypes.float32)
  }

  result = Tensor.randn(10, 10, dtype=dtypes.bfloat16) + Tensor.randn(10, 10, dtype=dtypes.bfloat16)
  result.realize()
  print(result)
  print(result.numpy())


if __name__ == "__main__":
  test_bf16()