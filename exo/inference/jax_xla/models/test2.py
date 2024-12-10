import asyncio

def test_llama_torch():
    from transformers import LlamaForCausalLM,  AutoTokenizer

    # Load model and tokenizer
    model_name = "unsloth/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    # Define prompt
    prompt = "Explain the importance of data engineering in machine learning."

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate response
    from exo.tracer import model_tracer
    model_tracer(model)
    outputs = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.7)

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

def print_decoded(arr):
    from transformers import AutoTokenizer

    # Load model and tokenizer
    model_name = "unsloth/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Next Tokens: {tokenizer.decode(arr)}")

def test_llama():
    from exo.inference.jax_xla.models.llama import LlamaDecoderLayer
    from transformers.models.llama import LlamaConfig
    # from jax import numpy as jnp
    from jax import live_arrays, profiler
    from flax import nnx
    import pickle

    config: LlamaConfig = LlamaConfig.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

    with open('pickle.dump', 'rb') as f:
        weights_map = pickle.load(f)
    rngs = nnx.Rngs(0)


    config.max_position_embeddings = 512

    
    layer = LlamaDecoderLayer(config,weights_map, rngs)
    print("Layer loaded")
    live_array_list = live_arrays()
    print(f"Live Arrays : {len(live_arrays())}")
    used_bytes = 0
    for arr in live_array_list:
        print(arr.shape)
    profiler.save_device_memory_profile("memory.prof")

def get_tokenizer():
    from transformers import AutoTokenizer

    # Load model and tokenizer
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def get_flax_model():
    from transformers.utils import SAFE_WEIGHTS_NAME, cached_file
    from exo.inference.shard import Shard
    from exo.inference.jax_xla.models.llama import ShardedLlamaModel

    model_id = "unsloth/Llama-3.2-1B-Instruct"
    resolved_archive_file = cached_file(model_id, SAFE_WEIGHTS_NAME)
    test_shard = Shard(start_layer = 0, n_layers = 16, end_layer = 15, model_id = model_id)
    engine = ShardedLlamaModel(None)

    engine.load_shard(test_shard)
    return engine

def get_inference_engine():
    from exo.inference.shard import Shard
    from exo.download.shard_download import NoopShardDownloader
    from exo.inference.jax_xla.inference_engine import JAXShardedInferenceEngine
    model_id = "llama-3.2-1b"
    shard = Shard(start_layer = 0, n_layers = 16, end_layer = 15, model_id = model_id)

    engine = JAXShardedInferenceEngine(NoopShardDownloader())
    return shard, engine

def get_tensor_eps(flax_arr, pt_tensor):
    import numpy as np
    f_arr = np.array(flax_arr)
    p_arr = pt_tensor.detach().numpy()
    if f_arr.shape != p_arr.shape:
        raise Exception("Cannot calculate eps when dims do no match.")
    return p_arr - f_arr

def test_flax_model():
    model = get_flax_model()
    tokenizer = get_tokenizer()

    prompt = "Explain the importance of data engineering in machine learning."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    from jax import numpy as jnp
    from flax import nnx
    import jax.random as random

    output = model(jnp.array(input_ids))

    print(output)

    logits = model.lm_head(output[:, -1:, :]) ## Keep only the logits from last token as only that is required.
    logits = jnp.squeeze(logits)
    probs = nnx.softmax(logits, axis=-1)

    key = random.PRNGKey(0)
    next_tokens = random.choice(key, a=jnp.arange(probs.shape[-1]), p=probs, shape=(1,)).squeeze(0)
    print(next_tokens)

async def test_sharded_flax():
    shard, engine = get_inference_engine()
    input_ids = [[128000,    849,  21435,    279,  12939,    315,    828,  15009,    304,
           5780,   6975,     13]]
    from exo.tracer import model_tracer
    import numpy as np
    await engine.ensure_shard(shard)
    model_tracer(engine)
    output = await engine.infer_tensor('test', shard, np.array(input_ids))
    print(output)



def torch_tracer(fn: callable, file_path: str):
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump([],f)
    def wrapped(*args, **kwargs):
        with open(file_path, 'rb') as f:
            data_points = pickle.load(f)
        data = {"args":args, 'kwargs': kwargs}
        ret_val = fn(*args, **kwargs)
        data['ret'] = ret_val
        data_points.append(data)
        with open(file_path, 'wb') as f:
            pickle.dump(data_points, f)
        return ret_val
    return wrapped

def get_model_calls():
    from exo.tracer import TracerInstance, Session
    session = Session()
    MODEL_PATH = "LlamaModel.LlamaModel.LlamaForCausalLM.LlamaForCausalLM.LlamaForCausalLM.LlamaForCausalLM.LlamaForCausalLM"
    module_calls = session.query(TracerInstance).where(TracerInstance.exec_path == MODEL_PATH).all()
    LAYER_PATH = f"LlamaDecoderLayer.LlamaDecoderLayer.LlamaModel.{MODEL_PATH}"
    layer_calls = session.query(TracerInstance).where(TracerInstance.exec_path == LAYER_PATH).all()
    session.expunge_all()
    session.close()
    return layer_calls, module_calls


if __name__ == "__main__":
    get_model_calls()
    # test_llama_torch()
    # asyncio.run(test_sharded_flax())
    # test_flax_model()