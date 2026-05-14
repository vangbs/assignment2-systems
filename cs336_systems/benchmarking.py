import torch
from collections.abc import Callable
import timeit
from cs336_basics.config import ModelConfig
from cs336_basics.model import BasicsTransformerLM
import pickle
from cs336_basics.config import MODEL_CONFIGS
from einops import rearrange

def benchmarking(func: Callable, warmup: int, steps: int,) -> list[float]:
    for _ in range(warmup):
        func()
    times = []
    torch.cuda.synchronize()
    for _ in range(steps):
        start = timeit.default_timer()
        func()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    return times

def forward_pass(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    outputs = model(inputs)
    return torch.nn.functional.cross_entropy(
        rearrange(outputs, "batchsize context_length vocab_size -> batchsize vocab_size context_length"),
        targets
    )

def forward_and_backward_pass(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
):
    loss = forward_pass(model, inputs, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

def single_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
):
    forward_and_backward_pass(model, inputs, targets, optimizer)
    optimizer.step()


def benchmarking_with_model(config: ModelConfig, warmup: int, steps: int):
    device = torch.device("cuda")
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(device)

    #from torchinfo import summary
    #summary(model, input_size=(config.batch_size, config.context_length), dtypes=[torch.int64], device=device)
    
    inputs = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
    targets = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=inputs.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    f = benchmarking(lambda: forward_pass(model, inputs, targets), warmup, steps)
    fb = benchmarking(lambda: forward_and_backward_pass(model, inputs, targets, optimizer), warmup, steps)
    fbs = benchmarking(lambda: single_step(model, inputs, targets, optimizer), warmup, steps)
    with open("data/benchmarking_results.pkl", "wb") as file:
        pickle.dump({
            "f": f,
            "fb": fb,
            "fbs": fbs,
        }, file)
    

if __name__ == "__main__":
    benchmarking_with_model(MODEL_CONFIGS['small'], warmup=5, steps=10)