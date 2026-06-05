import torch
from collections.abc import Callable
import timeit
from cs336_basics.model import scaled_dot_product_attention
from einops import rearrange
from itertools import product
import numpy as np

def crit(y: torch.Tensor) -> torch.Tensor:
    return y.pow(2).sum()

def benchmarking(
    warmup: int,
    steps: int,
    func: Callable,
    *args
) -> tuple[list[float], list[float], list[float]]:
    for _ in range(warmup):
        loss = crit(func(*args))
        loss.backward()
        for arg in args:
            if arg.requires_grad:
                arg.grad.zero_()
    
    fwd_times = []
    bwd_times = []
    mems = []
    torch.cuda.synchronize()
    for _ in range(steps):
        start = timeit.default_timer()
        
        loss = crit(func(*args))
        torch.cuda.synchronize()
        
        after_forward = timeit.default_timer()
        mems.append(torch.cuda.memory_allocated())
        
        loss.backward()
        torch.cuda.synchronize()
        
        end = timeit.default_timer()

        for arg in args:
            if arg.requires_grad:
                arg.grad.zero_()
        torch.cuda.synchronize()
        fwd_times.append(after_forward - start)
        bwd_times.append(end - after_forward)
        
    return fwd_times, bwd_times, mems


def attention_benchmark(
    attn: Callable,
    d_model: int,
    context_length: int,
    batch_size: int = 8,
    num_heads: int = 1,
    warmup: int = 10,
    steps: int = 100,
):
    d_k = d_v = d_model // num_heads
    device = torch.device("cuda")
    Q = torch.randn(batch_size, context_length, num_heads * d_k, device=device)
    K = torch.randn(batch_size, context_length, num_heads * d_k, device=device)
    V = torch.randn(batch_size, context_length, num_heads * d_v, device=device)
    Q, K, V = (
        rearrange(X, "... context_length (heads d) -> ... heads context_length d", heads=num_heads)
        for X in (Q, K, V)
    )
    mask = torch.tril(torch.ones(context_length, context_length, device=device)).bool()
    batch_dim = (1,) * (len(Q.shape) - 2)
    mask = mask.expand(batch_dim + mask.shape)
    Q.requires_grad_()
    K.requires_grad_()
    V.requires_grad_()

    base_mem = torch.cuda.memory_allocated()
    forward_time, backward_time, mems = benchmarking(
        warmup,
        steps,
        attn,
        Q, K, V, mask
    )
    mems = [(a - base_mem) / 1024**2 for a in mems]
    return forward_time, backward_time, mems

if __name__ == "__main__":
    torch.cuda.init()
    torch.cuda.empty_cache()
    d_models = [16, 32, 64, 128]
    context_lengths = [256, 1024, 4096]#, 8192, 16384]
    attn = torch.compile(scaled_dot_product_attention, fullgraph=True)
    for d_model, context_length in product(d_models, context_lengths):
        print(f"Benchmarking {d_model} {context_length}")
        forward_time, backward_time, mems = attention_benchmark(attn, d_model, context_length)
        print(f'{np.mean(forward_time)}s, {np.mean(backward_time)}s, {np.mean(mems)}MB')