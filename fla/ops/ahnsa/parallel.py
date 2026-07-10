# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# AHNSA: AHA-gated Native Sparse Attention.
#
# Design reference: `method.md` ("NSA + AHA 结合方案 v1") in the project rebuttal
# workspace. Summary of the design:
#
#   NSA is reorganized into two paths instead of three parallel branches:
#     - local path:   sliding-window attention (NSA's `g_swa` branch), always on.
#     - distant path: compression attention -> top-k block selection -> gathered
#                      selection attention (NSA's `g_cmp` / `g_slc` branches),
#                      executed serially and gated as *one unit* by a hard,
#                      per-(batch, token, head) AHA gate trained with a
#                      Straight-Through Estimator (STE).
#
#   o = g_swa * o_swa + aha_gate * (g_cmp * o_cmp + g_slc * o_sel)
#
# Note on KV-cache correctness (method.md Sec 8): this module never persists a
# separate "compressed KV cache" across calls. `fla.ops.nsa.parallel.parallel_nsa`
# recomputes the compressed K/V (`mean_pooling`) from the *full* raw K/V every
# single forward call. Consequently, skipping the distant path for the current
# query (AHA gate = 0) never causes any historical information to be lost: as
# long as the raw KV cache is maintained unconditionally (which happens one
# level up, in `fla.layers.ahnsa`, before this function is even called), any
# *future* query can always recompute a fully correct, complete compressed
# representation from that raw cache, regardless of what earlier queries' AHA
# gates were.

from __future__ import annotations

import warnings

import torch

from fla.ops.nsa.parallel import parallel_nsa

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = flash_attn_varlen_func = None


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
) -> torch.Tensor:
    r"""
    The always-on local ("Here") branch of AHNSA: standard causal sliding-window
    attention, delegated to FlashAttention-2 exactly as NSA's own `g_swa` branch
    does inside `fla.ops.nsa.parallel.parallel_nsa`. Exposed as a standalone
    function (rather than only reachable through `parallel_nsa`) so that the
    inference-time compute-skip path in `ahnsa_attn` can invoke *only* this
    branch, without ever materializing the compression/selection kernels.

    Args:
        q (torch.Tensor): queries of shape `[B, TQ, HQ, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]` (GQA enforced, see `parallel_nsa`).
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        window_size (int): sliding window size (number of past tokens visible, causal).
        scale (float, optional): softmax scale. Defaults to `1 / sqrt(K)`.
        cu_seqlens: see `parallel_nsa`.

    Returns:
        o (torch.Tensor): outputs of shape `[B, TQ, HQ, V]`.
    """
    if cu_seqlens is not None:
        if isinstance(cu_seqlens, tuple):
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
        else:
            cu_seqlens_q = cu_seqlens_k = cu_seqlens
        o = flash_attn_varlen_func(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            causal=True,
            window_size=(window_size - 1, 0),
            softmax_scale=scale,
        ).unsqueeze(0)
    else:
        o = flash_attn_func(
            q, k, v,
            causal=True,
            window_size=(window_size - 1, 0),
            softmax_scale=scale,
        )
    return o


def ahnsa_attn(
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
    training: bool = True,
) -> torch.Tensor:
    r"""
    AHA-gated Native Sparse Attention (AHNSA).

    The local sliding-window ("Here") branch is unconditional. The distant path --
    NSA's compression (`g_cmp`) and selection (`g_slc`) branches -- is gated as a
    single unit by a per-(batch, token, head) hard AHA gate:

        o = g_swa * o_swa + aha_gate * (g_cmp * o_cmp + g_slc * o_sel)

    Args:
        q (torch.Tensor): queries of shape `[B, TQ, HQ, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]` (GQA enforced, see `parallel_nsa`).
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        g_cmp, g_slc, g_swa (torch.Tensor):
            NSA's own *soft* fusion gates for the compression / selection / sliding-window
            branches, of shape `[B, TQ, HQ]` (i.e. `sigmoid(g_proj(hidden_states))`, unbound
            along the last dim). These are unrelated to the AHA gate below and are always
            applied exactly as in vanilla NSA.
        aha_gate_soft (torch.Tensor):
            The AHA router's soft score `sigmoid(router_logits)`, of shape `[B, TQ, HQ]`,
            differentiable w.r.t. the router weights.
        aha_gate_hard (torch.Tensor):
            The AHA router's hard decision `(aha_gate_soft > tau).float()`, of shape
            `[B, TQ, HQ]`. Expected to carry no gradient (e.g. produced by a comparison);
            the Straight-Through Estimator is applied internally in this function.
        block_indices, block_counts, block_size, window_size, scale, cu_seqlens:
            Forwarded to `fla.ops.nsa.parallel.parallel_nsa`; see its docstring.
        training (bool):
            When `False` (inference) and no (batch, token, head) triple in this call has
            `aha_gate_hard == 1`, the distant-path kernels (compression attention, top-k
            block selection, gathered selection attention) are skipped *entirely* and only
            the local branch is computed. This is a real compute saving, not merely a
            masked one (see module docstring re: KV-cache correctness). It is a
            call-granularity ("coarse") skip: it triggers whenever the *whole* forward
            call needs no distant access -- the common regime under AHA-style sparsity --
            but it does not skip compute for a *subset* of heads/tokens within a call that
            still needs the distant path for other heads/tokens. Fine-grained per-head
            skipping would additionally require either restricting to whole GQA groups or
            an active-query-compaction kernel (à la L2A); this is documented as future
            work in `method.md` Sec 9 and intentionally not implemented here.
            During training this fast path is disabled unconditionally, so that the
            (data-dependent) control flow never interacts with autograd/`torch.compile`
            graph capture for the training graph.

    Returns:
        o (torch.Tensor): outputs of shape `[B, TQ, HQ, V]`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if (not training) and bool((aha_gate_hard == 0).all()):
        # Coarse-grained inference fast path: nothing in this call needs the
        # distant path, so the compression + top-k + selection Triton kernels
        # are never invoked. `.all()` triggers one device sync per call, which
        # is negligible next to the kernels it lets us skip.
        o = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=v.dtype, device=v.device)
        if window_size > 0:
            o = sliding_window_attention(q, k, v, window_size=window_size, scale=scale, cu_seqlens=cu_seqlens)
            o = o * g_swa.unsqueeze(-1)
        return o

    # Straight-Through Estimator: forward uses the hard 0/1 decision, backward
    # treats the threshold as identity so gradients flow to `aha_gate_soft`
    # (and hence to the router weights) -- see method.md Sec 2.
    gate_ste = aha_gate_hard + (aha_gate_soft - aha_gate_soft.detach())

    # Gate NSA's distant-path soft weights with the AHA hard gate; the local
    # (sliding-window) weight `g_swa` is left untouched, i.e. always on. This
    # still invokes the (autograd-enabled) compression/selection kernels for
    # every token/head -- exactly the "masked" training-correct composition
    # described in method.md Sec 3.1/5, reusing fla's existing, tested NSA
    # kernels unmodified.
    g_cmp_eff = gate_ste * g_cmp
    g_slc_eff = gate_ste * g_slc

    return parallel_nsa(
        q=q,
        k=k,
        v=v,
        g_cmp=g_cmp_eff,
        g_slc=g_slc_eff,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
