from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    vocab_size: int = 10000
    batch_size: int = 4
    context_length: int = 512
    rope_theta: float | None = 10000.0

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small":  ModelConfig(768,3072,12,12),
    "medium": ModelConfig(1024,4096,24,16),
    "large":  ModelConfig(1280,5120,36,20),
    "xl":     ModelConfig(2560,10240,32,32),
    "10B":    ModelConfig(4608,12288,50,36),
}

if __name__ == "__main__":
    import torch
    from cs336_basics.model import BasicsTransformerLM
    from torchinfo import summary
    config = MODEL_CONFIGS["xl"]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )
    summary(model, input_size=(1, config.context_length), dtypes=[torch.int64])