import torch
from collections.abc import Callable
import timeit
from cs336_basics.config import ModelConfig
from cs336_basics.model import BasicsTransformerLM
import pickle
from cs336_basics.config import MODEL_CONFIGS
from einops import rearrange
import torch.cuda.nvtx as nvtx
import contextlib

def benchmarking(func: Callable, warmup: int, steps: int, nvtx_label: str) -> list[float]:
    for _ in range(warmup):
        func()
    times = []
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(steps):
        start = timeit.default_timer()
        with nvtx.range(nvtx_label):
            func()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    torch.cuda.memory._dump_snapshot(f"data/memory_snapshot_{nvtx_label}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    return times

def forward_pass(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ctx: contextlib.AbstractContextManager,
):
    with ctx:
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            rearrange(outputs, "batchsize context_length vocab_size -> batchsize vocab_size context_length"),
            targets
        )
    return loss

def forward_and_backward_pass(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ctx: contextlib.AbstractContextManager,
):
    loss = forward_pass(model, inputs, targets, ctx)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

def single_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ctx: contextlib.AbstractContextManager,
):
    forward_and_backward_pass(model, inputs, targets, optimizer, ctx)
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
    
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    #ctx = contextlib.nullcontext()
    ex_setups = {
        "forward_pass": lambda: forward_pass(model, inputs, targets, ctx),
        "forward_and_backward_pass": lambda: forward_and_backward_pass(model, inputs, targets, optimizer, ctx),
        "single_step": lambda: single_step(model, inputs, targets, optimizer, ctx),
    }

    results = {}
    for setup_name, setup_func in ex_setups.items():
        results[setup_name] = benchmarking(setup_func, warmup, steps, setup_name)
    
    with open("data/benchmarking_results.pkl", "wb") as file:
        pickle.dump(results, file)
    

if __name__ == "__main__":
    benchmarking_with_model(MODEL_CONFIGS['small'], warmup=5, steps=1)