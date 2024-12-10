from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaMLP, LlamaDecoderLayer, LlamaSdpaAttention
import numpy as np 

full_model = LlamaForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")


print("pre trained model loaded in pytorch")
torch_config = full_model.config

torch_model_layer_1: LlamaDecoderLayer = full_model.base_model.layers[0]

torch_norm_layer: LlamaRMSNorm = torch_model_layer_1.input_layernorm
torch_norm_layer_weight: np.array  = full_model.base_model.layers[0].input_layernorm.weight.detach().numpy()

torch_mlp_layer: LlamaMLP = torch_model_layer_1.mlp
torch_mlp_weights = {
    'gate': torch_mlp_layer.gate_proj.weight.detach().numpy().transpose(),
    'up' : torch_mlp_layer.up_proj.weight.detach().numpy().transpose(),
    'down': torch_mlp_layer.down_proj.weight.detach().numpy().transpose(),
}

torch_attention_layer: LlamaSdpaAttention = torch_model_layer_1.self_attn
torch_attention_weights = {
    'q': torch_model_layer_1.self_attn.q_proj.weight.detach().numpy().transpose(),
    'k': torch_model_layer_1.self_attn.k_proj.weight.detach().numpy().transpose(),
    'v': torch_model_layer_1.self_attn.v_proj.weight.detach().numpy().transpose(),
    'o': torch_model_layer_1.self_attn.o_proj.weight.detach().numpy().transpose()
}

