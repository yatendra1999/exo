import exo.inference.jax_xla.models.llama as base_llama 
import exo.inference.jax_xla.models.llama_jit as jit_llama
from exo.inference.jax_xla.models.llama import LlamaEmbedding
from exo.inference.shard import Shard
from transformers import LlamaConfig
from transformers.utils import SAFE_WEIGHTS_NAME, cached_file
import jax
from jax import numpy as jnp
from jax.nn import dot_product_attention
from flax import nnx
from timeit import timeit
from threading import Thread
import time


jit_attention = jax.jit(dot_product_attention, static_argnames=['bias', 'mask', 'scale', 'is_causal', 'query_seq_lengths', 'key_value_seq_lengths', 'local_window_size', 'implementation'])

model_id = 'unsloth/Llama-3.2-1B-Instruct'
config = LlamaConfig.from_pretrained(model_id)
st_path = cached_file(model_id, SAFE_WEIGHTS_NAME)
shard = Shard(start_layer=0, n_layers=16, end_layer=15, model_id=model_id)
embeddings = LlamaEmbedding.from_safetensor(config, "model.embed_tokens", st_path, 'pt')
token_ids = [[
        128000,
        128006,
        9125,
        128007,
        271,
        38766,
        1303,
        33025,
        2696,
        25,
        6790,
        220,
        2366,
        18,
        198,
        15724,
        2696,
        25,
        220,
        717,
        3799,
        220,
        2366,
        19,
        271,
        128009,
        128006,
        882,
        128007,
        271,
        849,
        21435,
        279,
        12939,
        315,
        828,
        15009,
        304,
        5780,
        6975,
        13,
        128009,
        128006,
        78191,
        128007,
        271,
    ]]
# token_ids = [[9125]]
batch_size = 1
query_len = len(token_ids[0])
pos_ids = jnp.expand_dims(jnp.arange(query_len), (0))
hidden_state = embeddings(jnp.array(token_ids))

module_name = "LlamaAttention"
module_init_args = [config, "model.layers.0.self_attn", st_path, "pt"]


setattr(base_llama, "rotary_embedding", getattr(base_llama, "LlamaRotaryEmbedding")(config))
# setattr(jit_llama, "rotary_embedding", getattr(jit_llama, "LlamaRotaryEmbedding")(config))
getattr(getattr(base_llama, "rotary_embedding"), "create_embed")(pos_ids)
# getattr(getattr(jit_llama, "rotary_embedding"), "create_embed")(pos_ids)

## JIT position embedding __call__ function
# setattr(jit_llama, 'rotary_embedding', nnx.jit(getattr(jit_llama, "rotary_embedding"),))

# base_module = getattr(getattr(base_llama, module_name), "from_safetensor")(*module_init_args)
# jit_module = getattr(getattr(jit_llama, module_name), "from_safetensor")(*module_init_args)

## Intermediate init
# getattr(base_module, 'create_embed')(pos_ids)
# getattr(jit_module, 'create_embed')(pos_ids)


# func_name = "_split_heads"
# base_func = getattr(base_module, func_name)
# jit_func = getattr(jit_module, func_name)

## To call module
# base_func = base_module
# jit_func = jit_module

# timeit_globals = {
#     "config": config,
#     "base_func": base_func,
#     "jit_func": jit_func,
#     "pos_ids": pos_ids,
#     "hidden_state": hidden_state,
#     # "sin": sin,
#     # "cos": cos
# }


# def get_random_array(dims) -> jax.Array:
#     return jax.random.normal(jax.random.key(0), dims)

# def test_func_tuple(args: str):
#     base_time = timeit(f"base_func({args})", globals=timeit_globals, number=1)
#     jit_first_time = timeit(f"jit_func({args})[0].block_until_ready()", globals=timeit_globals, number=1)
#     jit_next_time = timeit(f"jit_func({args})[0].block_until_ready()", globals=timeit_globals, number=1)
#     print(f"Normal: {base_time}\t JIT first: {jit_first_time}\t JIT subsequent: {jit_next_time}")
#     eps = jnp.std(eval(f"base_func({args})[0]") - eval(f"jit_func({args})[0]") )
#     eps_2 = jnp.std(eval(f"base_func({args})[1]") - eval(f"jit_func({args})[1]") )
#     print(f"Array diff: {eps} {eps_2}")

# def test_func(args: str, jit_args = None):
#     if jit_args == None:
#         jit_args = args
#     ## Timing
#     # jit_first_time = timeit(f"jit_func({jit_args}).block_until_ready()", globals=timeit_globals, number=1)
#     # jit_next_time = timeit(f"jit_func({jit_args}).block_until_ready()", globals=timeit_globals, number=10)
#     # base_time = timeit(f"base_func({args})", globals=timeit_globals, number=1)
#     # base_time_next = timeit(f"base_func({args})", globals=timeit_globals, number=10)
#     # print(f"Normal: {base_time}\t Normal subsequent: {base_time_next}\t JIT first: {jit_first_time}\t JIT subsequent: {jit_next_time}")
    
#     ## EPS Eval
#     base_ret = eval(f"base_func({args})")
#     jit_ret = eval(f"jit_func({jit_args})")
#     jit_ret.block_until_ready()
#     # print(f"Array diff:{jnp.std(base_ret[0] - jit_ret[0])}, {jnp.std(base_ret[1] - jit_ret[1])}, {jnp.std(base_ret[2] - jit_ret[2])}")
#     # print(jnp.std(dot_product_attention(*base_ret, bias=None, is_causal=True, mask=None) - jit_attention(*base_ret, bias=None, is_causal=True, mask=None)))
#     eps = jnp.std(base_ret - jit_ret)
#     print(f"Array diff: {eps}")

# test_func("hidden_state, None, pos_ids", "hidden_state, None, pos_ids")

# def test_attention():
#     q,k,v = eval(f"base_func(hidden_state, None, pos_ids)")
#     from exo.inference.jax_xla.models.llama_jit import test_attn_jax_jit, test_attn_nnx_jit, test_attn_no_jit
#     attn_1 = test_attn_no_jit(q,k,v)
#     attn_2 = test_attn_jax_jit(q,k,v)
#     attn_3 = test_attn_nnx_jit(q,k,v)
#     print(f"EPS: {jnp.std(attn_1 - attn_2)}, {jnp.std(attn_2 - attn_3)}, {jnp.std(attn_1 - attn_3)}")

# test_attention()

def calc_eps(a: jax.Array, b: jax.Array):
    print(f"STD : {jnp.std(a - b)}")
    # eps = (jax.lax.sqrt(jax.lax.square(b - a))).mean(axis=-1)
    # orig = jax.lax.abs(a).mean(axis=-1)
    # eps = eps/orig
    # print(f"EPS: Mean: {eps.mean()*100} Min: {eps.min()*100} Max: {eps.max()*100}")

def test_layer_performance():
    from exo.inference.jax_xla.models.llama_jit import test_layer, ModelLayer
    from exo.inference.jax_xla.models.llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    llama = LlamaDecoderLayer.from_safetensor(config,"model.layers.0", st_path, "pt")

    eps = jnp.array(config.rms_norm_eps, dtype=jax.dtypes.bfloat16)
    input_layernorm = llama.input_layernorm.weights
    post_attention_layernorm = llama.post_attention_layernorm.weights
    q_proj = llama.self_attn.q_proj.kernel.value
    k_proj = llama.self_attn.k_proj.kernel.value
    v_proj = llama.self_attn.v_proj.kernel.value
    o_proj = llama.self_attn.o_proj.kernel.value
    down_proj = llama.mlp.down_proj.kernel.value
    gate_proj = llama.mlp.gate_proj.kernel.value
    up_proj = llama.mlp.up_proj.kernel.value
    rot_embed:LlamaRotaryEmbedding = getattr(base_llama, "rotary_embedding")
    sin = rot_embed.sin
    cos = rot_embed.cos
    kv_cache= jnp.zeros((2, 0, 8, 64), dtype=jax.dtypes.bfloat16)
    cache_index =0
    llama_jit = ModelLayer(
        num_kv_heads=config.num_key_value_heads,
        num_attention_heads=config.num_attention_heads
    )
    jit_kwargs = {
        "hidden_state": hidden_state,
        "input_layernorm": input_layernorm.value,
        "post_attention_layernorm": post_attention_layernorm,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": o_proj,
        "down_proj": down_proj,
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "sin": sin,
        "cos": cos,
        "epsilon": eps,
        # "num_kv_heads": config.num_key_value_heads,
        # "num_attention_heads": config.num_attention_heads,
        # "is_causal": True,
        # "cache_index": cache_index,
        # "kv_cache": kv_cache
    }

    kwargs = {
        "hidden_states": hidden_state,
        "attention_mask": None,
        "position_ids": pos_ids
    }

    func_globals = {
        "llama": llama,
        "llama_jit": llama_jit,
        "jit_kwargs": jit_kwargs,
        "kwargs": kwargs
    }
    
    # for i in range(5):
    # print(f"TEST CACHE {i}")
    

    ## Test Flax Timings for comparison

    start = time.time_ns()
    out = llama(**kwargs)
    out.block_until_ready()
    base_time = time.time_ns() - start
    start = time.time_ns()
    out_next = llama(**kwargs)
    out_next.block_until_ready()
    sub_time = time.time_ns() - start
    print(f"Timings Flax: Base: {base_time}, Subsequent: {sub_time}")

    
    ## Call once for JIT compilation
    start = time.time_ns()
    out_jit_first = llama_jit(**jit_kwargs)
    out_jit_first.block_until_ready()
    print(f"JIT Base Time: {time.time_ns() - start}")
    # jit_kwargs['kv_cache'] = kv_cache
    
    ## Profile Subsequent calls with JIT cache
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False):
        start = time.time_ns()
        out_jit = llama_jit(**jit_kwargs)
        out_jit.block_until_ready()
        print(f"JIT Time: {time.time_ns() - start}")

        # jit_kwargs['kv_cache'] = kv_cache

        ## Test 2
        start = time.time_ns()
        out_jit = llama_jit(**jit_kwargs)
        out_jit.block_until_ready()
        print(f"JIT Time: {time.time_ns() - start}")

    
    ## Calculate EPS
    calc_eps(out_next, out_jit)


    # print(f"TIMINGS: BASE {time_base}ns JIT {time_jit}ns")
    # jit_kwargs['kv_cache'] = kv_cache
    
    # base_time = timeit(f"llama(**kwargs)[0].block_until_ready()", globals=func_globals, number=10)
    # jit_first_time = timeit(f"llama_jit(**jit_kwargs)[0].block_until_ready()", globals=func_globals, number=1)
    # jit_next_time = timeit(f"llama_jit(**jit_kwargs)[0].block_until_ready()", globals=func_globals, number=10)
    # print(f"TIME WITH CACHE: BASE:{base_time}\tJIT{jit_first_time}\tJIT NEXT:{jit_next_time}")

# print("## WITHOUT JIT")
# with jax.disable_jit():
#     test_layer()
print("## WITH JIT")
# test_layer_performance()



def test_full_model():
    from exo.inference.jax_xla.models.llama_jit import JITLlamaModel, prep_rotary_embed, generate_splitter
    from exo.inference.jax_xla.models.utils.rope import compute_llama3_parameters

    setattr(jit_llama, "_split_query", generate_splitter(config.num_attention_heads))
    setattr(jit_llama, "_split_kv", generate_splitter(config.num_key_value_heads))

    model = JITLlamaModel()
    model.load_partial(shard, st_path, config)
    rot_embed = prep_rotary_embed
    jit_model = model
    freq, attn_scaling = compute_llama3_parameters(config)
    print("#"*10)
    start = time.time_ns()
    sin, cos = rot_embed(jnp.array(token_ids), freq, attn_scaling)
    out_1 = jit_model(jnp.array(token_ids), sin, cos)
    out_1.block_until_ready()
    print(f"{out_1.shape}: {time.time_ns() - start}")
    start = time.time_ns()
    out_2 = jit_model(jnp.array(token_ids), sin, cos)
    out_2.block_until_ready()
    print(f"{out_2.shape}: {time.time_ns() - start}")
    print("Wahta is ahskldjfp[pindfg]")

# test_full_model()



def aot_cache(config: LlamaConfig, max_tokens: int = 100):
    cache = jnp.zeros((2, 0, config.num_key_value_heads, config.hidden_size // config.num_key_value_heads))
    for i in range(max_tokens):
        k = jnp.ones((1, 1, config.num_key_value_heads, config.hidden_size // config.num_key_value_heads))
        v = jnp.ones((1, 1, config.num_key_value_heads, config.hidden_size // config.num_key_value_heads))
        kv = jax.lax.concatenate(
            (k, v),
            dimension=0
        )
        cache = jax.lax.concatenate((cache, kv), dimension=1)
        k = cache[0, ...]
        v = cache[1, ...]

def run_iters(model: callable, gen_embed: callable, freq: jax.Array, attn_scaling: jax.Array, start_tokens: jax.Array, next_token: jax.Array, times: int = 10):
    base_start = time.time_ns()
    sin, cos  = gen_embed(start_tokens, freq, attn_scaling)
    base_out = model(start_tokens, sin, cos)
    print(f"Base Time: {time.time_ns() - base_start} ns")
    base_start = time.time_ns()
    for i in range(times):
        sin, cos = gen_embed(next_token, freq, attn_scaling)
        next_run = model(next_token, sin, cos)
    total_ns = time.time_ns() - base_start
    per_token = total_ns / times
    tokens_per_second = 1000000000 / per_token
    print(f"Total time: {total_ns}, token_times: {per_token}, per second: {tokens_per_second}")


def test_aot_cache_compilation():
    from exo.inference.jax_xla.models.llama_jit import JITLlamaModel, prep_rotary_embed, generate_splitter
    from exo.inference.jax_xla.models.utils.rope import compute_llama3_parameters

    setattr(jit_llama, "_split_query", generate_splitter(config.num_attention_heads))
    setattr(jit_llama, "_split_kv", generate_splitter(config.num_key_value_heads))

    model = JITLlamaModel()
    jit_llama.load_partial(model, shard, st_path, config)
    # model = nnx.jit(model)

    initial_token_ids = jnp.array(token_ids)
    next_tokens = jnp.array([[1303]])

    freq, attn_scaling = compute_llama3_parameters(config)
    run_iters(model, prep_rotary_embed, freq, attn_scaling, initial_token_ids, next_tokens)
    # aot_cache(config)
    jit_llama.reset_cache(model)
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    run_iters(model, prep_rotary_embed, freq, attn_scaling, initial_token_ids, next_tokens)

test_aot_cache_compilation()


# from exo.inference.jax_xla.models.llama_jit import get_splitter
# test_func = get_splitter(8)
# print(test_func)
# print(hidden_state.shape)
# test_func(hidden_state).shape