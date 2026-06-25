import torch
import triton
import triton.language as tl
from cs336_systems.config import TilingConfig
import math
from einops import einsum, rearrange

tilingconfig = TilingConfig()
block_q, block_k = tilingconfig.block_q, tilingconfig.block_k


class MyFlashAttention_torch(torch.autograd.Function):
    # Do not track gradients by default
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False
    ):
        t_q = math.ceil(Q.shape[-2] / block_q)
        t_k = math.ceil(K.shape[-2] / block_k)
        sqrt_d = math.sqrt(Q.shape[-1])
        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype = torch.float32)
        for i in range(t_q):
            q_i = Q[..., i * block_q : (i + 1) * block_q, :]
            o_i = torch.zeros_like(q_i, dtype = torch.float32)
            l_i = torch.zeros(q_i.shape[:-1], device=Q.device, dtype = torch.float32)
            last_m_i = torch.full(q_i.shape[:-1], float('-inf'), device=Q.device, dtype = torch.float32)
            
            for j in range(t_k):
                k_j = K[..., j * block_k : (j + 1) * block_k, :]
                v_j = V[..., j * block_k : (j + 1) * block_k, :]
                s_ij = einsum(q_i, k_j, "... block_q d, ... block_k d -> ... block_q block_k") / sqrt_d
                m_i = torch.maximum(last_m_i, torch.max(s_ij, dim=-1).values)
                p_i = torch.exp(s_ij - rearrange(m_i, "... -> ... 1"))
                l_i = torch.exp(last_m_i - m_i) * l_i + torch.sum(p_i, dim=-1)
                o_i = (
                    rearrange(torch.exp(last_m_i - m_i), "... -> ... 1") * o_i
                    + einsum(p_i.to(v_j.dtype), v_j, "... block_q block_k, ... block_k d -> ... block_q d")
                )
                last_m_i = m_i
            
            o_i = rearrange(1.0 / l_i, "... -> ... 1") * o_i
            L_i = m_i + torch.log(l_i)
            O[..., i * block_q : (i + 1) * block_q, :] = o_i.to(Q.dtype)
            L[..., i * block_q : (i + 1) * block_q] = L_i
        
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
        

    # Do not track gradients by default
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
    
# O.dtype == Q.dtype, L.dtype == tl.float32
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    batch_index = tl.program_id(0)
    q_tile_idx = tl.program_id(1)
    t_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (q_tile_idx * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0)
    )
    K_T_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (D, N_KEYS),
        strides = (stride_kd, stride_kk),
        offsets = (0, 0),
        block_shape = (D, K_TILE_SIZE),
        order = (0, 1)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0)
    )
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    last_m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    for j in range(t_k):
        k_T = tl.load(K_T_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        # s_ij: fp32
        s_ij = tl.dot(q, k_T) * scale
        
        # boundary check
        q_idx = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        q_mask = q_idx < N_QUERIES
        k_mask = k_idx < N_KEYS
        mask = q_mask[:, None] & k_mask[None, :]
        
        if IS_CAUSAL:
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            mask = mask & causal_mask
        
        s_ij = tl.where(mask, s_ij, float('-inf'))

        
        # float('-inf') protection to avoid nan
        local_max = tl.maximum(tl.max(s_ij, axis=-1), -1000000.0)
        m = tl.maximum(last_m, local_max)
        
        p = tl.exp(s_ij - m[:, None])
        coe = tl.exp(last_m - m)
        l = coe * l + tl.sum(p, axis=-1)
        o = coe[:, None] * o + tl.dot(p.to(v.dtype), v)
        last_m = m

        K_T_block_ptr = K_T_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    eps = 1e-8
    l += eps
    o = (1.0 / l)[:, None] * o
    l = last_m + tl.log(l)
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets = (q_tile_idx * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (q_tile_idx * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,)
    )
    tl.store(O_block_ptr, o.to(q.dtype), boundary_check=(0,))
    tl.store(L_block_ptr, l, boundary_check=(0,))


class MyFlashAttention_triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False
    ):
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Q, K, V must be contiguous"
        # Return views of Q, K, V
        Q_flat, K_flat, V_flat = (
            rearrange(X, "... context_length d -> (...) context_length d")
            for X in (Q, K, V)
        )
        O_flat = torch.zeros_like(Q_flat)
        L_flat = torch.zeros(Q_flat.shape[:-1], device=Q.device, dtype=torch.float32)
        
        flash_fwd_kernel[(Q_flat.shape[0], triton.cdiv(Q_flat.shape[1], block_q))](
            Q_flat, K_flat, V_flat,
            O_flat, L_flat,
            Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
            K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
            V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
            O_flat.stride(0), O_flat.stride(1), O_flat.stride(2),
            L_flat.stride(0), L_flat.stride(1),
            Q_flat.shape[1], K_flat.shape[1],
            1 / math.sqrt(Q_flat.shape[-1]),
            Q_flat.shape[-1],
            block_q,
            block_k,
            is_causal
        )
        
        ctx.save_for_backward(L_flat, Q_flat, K_flat, V_flat, O_flat)
        ctx.is_causal = is_causal
        
        O = O_flat.reshape(Q.shape)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError

if __name__ == "__main__":
    d_model: int = 64
    context_length: int = 256
    batch_size: int = 8
    num_heads: int = 1
    d_k = d_v = d_model // num_heads
    device = torch.device("cuda")
    Q = torch.randn(batch_size, context_length, num_heads * d_k, device=device, dtype=torch.bfloat16)
    K = torch.randn(batch_size, context_length, num_heads * d_k, device=device, dtype=torch.bfloat16)
    V = torch.randn(batch_size, context_length, num_heads * d_v, device=device, dtype=torch.bfloat16)
    Q, K, V = (
        rearrange(X, "... context_length (heads d) -> ... heads context_length d", heads=num_heads).contiguous()
        for X in (Q, K, V)
    )
    Q.requires_grad_()
    K.requires_grad_()
    V.requires_grad_()
    O = MyFlashAttention_triton.apply(Q, K, V)
    torch.cuda.synchronize()
    print(O.shape)
