import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit

base = 250000
world_size = 6
warmup = 5
steps = 10

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, length):
    setup(rank, world_size)
    data = torch.randint(0, 10, (length,))
    dist.all_reduce(data, async_op=False)
    dist.destroy_process_group()

if __name__ == "__main__":
    for sz in [1, 10, 100, 1000]:
        for _ in range(warmup):
            mp.spawn(fn=distributed_demo, args=(world_size, sz * base), nprocs=world_size, join=True)
        start = timeit.default_timer()
        for _ in range(steps):
            mp.spawn(fn=distributed_demo, args=(world_size, sz * base), nprocs=world_size, join=True)
        end = timeit.default_timer()
        print(f"Time taken: {end - start}")