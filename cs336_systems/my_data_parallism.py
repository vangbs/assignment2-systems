import torch
import torch.distributed as dist
from collections.abc import Callable
from typing import Any
import heapq


class My_data_bucket:
    def __init__(self, dist_fn: Callable, size: int = 25 * 1024**2, **kwargs):
        self.size = size
        self.current_size = 0
        self.buffer = []
        self.ongoing = []
        self.dist_fn = dist_fn
        self.kwargs = kwargs
    
    @torch.no_grad()
    def flush(self):
        flat_tensors = torch._utils._flatten_dense_tensors(self.buffer)
        handle = self.dist_fn(flat_tensors, async_op=True, **self.kwargs)
        self.ongoing.append((handle, flat_tensors, self.buffer))
        self.current_size = 0
        self.buffer = []
    
    def send(
        self,
        param: torch.Tensor,
    ):
        self.buffer.append(param)
        self.current_size += param.numel() * param.element_size()
        if self.current_size >= self.size:
            self.flush()
    
    @torch.no_grad()
    def finish(self):
        self.flush()
        for handle, flat_tensors, buffer in self.ongoing:
            handle.wait()
            unflattened_tensors = torch._utils._unflatten_dense_tensors(flat_tensors, buffer)
            for param, unflattened_tensor in zip(buffer, unflattened_tensors):
                param.copy_(unflattened_tensor)
        self.ongoing = []

# Assume deterministic execution order in backward pass
class My_ddp_model(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.data_bucket_broadcast = My_data_bucket(dist.broadcast, src=0)
        self.data_bucket_all_reduce = My_data_bucket(dist.all_reduce)
        for param in module.parameters():
            self.data_bucket_broadcast.send(param)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(lambda p: self.data_bucket_all_reduce.send(p.grad))
        self.data_bucket_broadcast.finish()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        self.data_bucket_all_reduce.finish()
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.div_(world_size)

def get_params_list(params, rank, current_loading):
    sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)
    
    result = []
    param_to_rank = {}

    for param in sorted_params:
        load, rank_id = heapq.heappop(current_loading)
        if rank_id == rank:
            result.append(param)
        heapq.heappush(current_loading, (load + param.numel(), rank_id))
        param_to_rank[param] = rank_id
    
    return result, param_to_rank

class My_ddp_optimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params_groups,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs
    ):
        params_groups = list(params_groups)
        if not isinstance(params_groups[0], dict):
            params_groups = [{'params': params_groups}]

        world_size = dist.get_world_size()
        
        # heap_element: (current_load, rank_id)
        # No need to heapify, already ordered
        self.current_loading = [(0, i) for i in range(world_size)]
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.optimizer = None
        self.data_bucket = [My_data_bucket(dist.broadcast, src=i) for i in range(world_size)]
        self.param_to_rank = {}

        # Maintain all parameters, will be broadcasted later
        super().__init__(params_groups, defaults=kwargs)
        
        
    def add_param_group(self, params_group: dict[str, Any]):
        local_params, param_to_rank = get_params_list(params_group['params'], dist.get_rank(), self.current_loading)
        local_group = dict(params_group)
        local_group['params'] = local_params
        self.param_to_rank.update(param_to_rank)

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
        else:
            self.optimizer.add_param_group(local_group)
        
        super().add_param_group(params_group)
    
    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)
        for group in self.param_groups:
            for param in group['params']:
                self.data_bucket[self.param_to_rank[param]].send(param)
        for bucket in self.data_bucket:
            bucket.finish()
        return loss

