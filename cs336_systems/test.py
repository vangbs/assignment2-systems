import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.config import MODEL_CONFIGS
from torchinfo import summary

config = MODEL_CONFIGS["small"]
model = BasicsTransformerLM(
    vocab_size=config.vocab_size,
    context_length=config.context_length,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    rope_theta=config.rope_theta,
).to("meta")
summary(model, input_size=(1, config.context_length), dtypes=[torch.int64])