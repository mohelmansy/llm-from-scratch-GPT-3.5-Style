
from dataclasses import dataclass
from transformers import LlamaConfig, LlamaForCausalLM

@dataclass
class ModelSpec:
    hidden_size: int = 768
    n_layer: int = 12
    n_head: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 1024
    vocab_size: int = 50257
    rope_theta: float = 100000.0

def build_llama_like(spec: ModelSpec) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.n_layer,
        num_attention_heads=spec.n_head,
        intermediate_size=spec.intermediate_size,
        max_position_embeddings=spec.max_position_embeddings,
        vocab_size=spec.vocab_size,
        rope_theta=spec.rope_theta,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(cfg)
