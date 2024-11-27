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
    outputs = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.7)

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

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


if __name__ == "__main__":
    test_llama_torch()