# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import warnings

import torch

from fla.ops.nsa.naive import naive_nsa_compression, naive_nsa_selection, naive_nsa_topk
from fla.ops.utils.pooling import mean_pooling

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = flash_attn_varlen_func = None


def naive_ahnsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    aha_gate_soft: torch.Tensor,
    aha_gate_hard: torch.Tensor,
    block_indices: torch.LongTensor | None = None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 64,
    window_size: int = 512,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
) -> torch.Tensor:
    r"""
    Pure-PyTorch reference implementation of AHNSA, for correctness debugging against
    `fla.ops.ahnsa.parallel.ahnsa_attn` only. Always computes every branch densely
    (no compute-skipping): the distant path (compression + selection) is computed in
    full and then multiplied by the STE hard AHA gate, while the local sliding-window
    path is always on -- i.e. this always follows the "masked training" semantics of
    `method.md` Sec 3.1, regardless of a `training` flag.

    Args / Returns: see `fla.ops.ahnsa.parallel.ahnsa_attn`.
    """
    assert block_counts is not None, "block counts must be provided for selection"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"

    if cu_seqlens is not None:
        if isinstance(cu_seqlens, tuple):
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
        else:
            cu_seqlens_q = cu_seqlens_k = cu_seqlens
    else:
        cu_seqlens_q = cu_seqlens_k = None

    k_cmp, v_cmp = mean_pooling(k, block_size, cu_seqlens), mean_pooling(v, block_size, cu_seqlens)
    o_cmp, _ = naive_nsa_compression(
        q=q, k_cmp=k_cmp, v_cmp=v_cmp, block_size=block_size, scale=scale, cu_seqlens=cu_seqlens,
    )
    if block_indices is None:
        block_indices = naive_nsa_topk(
            q=q, k_cmp=k_cmp, block_counts=block_counts, block_size=block_size, scale=scale, cu_seqlens=cu_seqlens,
        )
    o_sel = naive_nsa_selection(q, k, v, block_indices, block_size, scale, cu_seqlens)

    gate_ste = aha_gate_hard + (aha_gate_soft - aha_gate_soft.detach())
    o_distant = g_cmp.unsqueeze(-1) * o_cmp + g_slc.unsqueeze(-1) * o_sel
    o = gate_ste.unsqueeze(-1) * o_distant

    if window_size > 0:
        if cu_seqlens is not None:
            o_swa = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                causal=True,
                window_size=(window_size - 1, 0),
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(q, k, v, causal=True, window_size=(window_size - 1, 0))
        o = o + g_swa.unsqueeze(-1) * o_swa
    return o
