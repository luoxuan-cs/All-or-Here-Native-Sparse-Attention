> **Note (EN):** This is the original design note (in Chinese, written during development) for **AHNSA**
> (AHA-gated Native Sparse Attention), the addition this fork makes on top of upstream
> `fla-org/flash-linear-attention`. It covers the motivation, algorithm, shapes/formulas, training vs.
> inference pseudocode, and KV-cache correctness rules. For where the corresponding code lives and how to
> use it, see the "AHNSA" section in [`README.md`](README.md). Section 10 (near the end of this file) maps
> the design directly onto the implementation files.

# NSA + AHA 结合方案（v1：Local 永远算，Distant 整体门控）

本方案把 NSA 的计算路径重新组织为 **local path**（永远执行）和 **distant path**（按 per-token-per-head 硬门控条件执行），以复现 AHA 的 "All-or-Here" 语义。本文档面向实现，给出 shape、公式、训练/推理伪代码和 KV cache 维护规则。

---

## 0. 设计结论（一句话）

```text
o[t, h] = o_swa[t, h] + gate[t, h] * o_distant[t, h]

o_distant = NSA 原有的 compression -> selection 串行路径
gate 由 fused q_proj 产生，per-token per-head，训练用 STE，推理可跳过计算
```

- **local path = NSA 的 sliding window attention（SWA）**，任何 token、任何 head 都必须执行，不受 gate 影响。
- **distant path = NSA 的 compression attention + selection attention 这条串行流程**，作为一个整体单元，由 AHA 风格的硬门控决定是否执行。
- 门控粒度是 **per-token, per-head**，不是 per-token（这是 AHA 相对 L2A/DuoAttention 的差异化设计，需要保留）。
- **不要**只 gate selection 而保留 compression attention 常开（那是另一个变体，本文档不采用，理由见第 5 节）。

---

## 1. NSA 原始结构回顾（依赖关系，不是三分支并列）

NSA 的计算依赖关系是 **两条路径**，其中一条内部是串行的：

```text
                     Q, K_raw, V_raw
                          │
            ┌─────────────┴─────────────┐
            │                           │
      Distant Path（串行）         Local Path
            │                           │
   1) 压缩历史 K,V -> K_cmp,V_cmp        │
            │                     sliding window
   2) compression attention:            attention
      o_cmp, p_cmp = Attn(Q,K_cmp,V_cmp)     │
            │                           │
   3) 用 p_cmp 给历史 block 打分            │
      并 top-n 选块（+强制 sink/local block） │
            │                           │
   4) selection attention:              │
      gather 原始 K,V(选中块)              │
      o_sel = Attn(Q, K_sel, V_sel)      │
            │                           │
            └─────────────┬─────────────┘
                          │
        o = g_cmp·o_cmp + g_sel·o_sel + g_swa·o_swa
        （g_cmp, g_sel, g_swa 是 NSA 原生的 soft fusion gate）
```

要点：

- Compression 和 Selection **不能并行**：selection 依赖 compression 产生的 `p_cmp` 来选块。
- Local（SWA）和 Compression 可以并行，二者互不依赖。
- `p_cmp` 就是 compression attention 的 softmax 概率，shape `[B, H, T, M]`（`M` = compressed block 数），不需要额外 sigmoid。

---

## 2. AHA 门控回顾（复用 fused q_proj 的写法）

参考 `xuan-luo/AHA-OLMO2` 的 `modeling_faolmo.py` 实现方式：把 router logit 融合进 `q_proj` 的输出通道，而不是新开一个独立的 router 权重矩阵。

```python
# q_proj 输出通道数从 H*D 扩到 H*D + H（每个 head 多一个 gate logit）
self.q_proj = nn.Linear(
    hidden_size,
    num_heads * head_dim + num_heads,
    bias=attention_bias,
)

query_states, gate_logits = torch.split(
    self.q_proj(hidden_states),
    [num_heads * head_dim, num_heads],
    dim=-1,
)
# query_states: [B, T, H*D] -> reshape -> [B, H, T, D]
# gate_logits:  [B, T, H]
```

STE 二值化：

```python
gate_soft = torch.sigmoid(gate_logits)              # [B, T, H]
gate_hard = (gate_soft > tau).to(gate_soft.dtype)     # tau = 0.5
gate_ste  = gate_hard + (gate_soft - gate_soft.detach())
```

- `gate_soft`：用于 L1 稀疏正则。
- `gate_hard`：真正决定 forward 是否执行 distant path。
- `gate_ste`：训练时参与前向数值计算，反向时梯度按恒等函数回传到 `gate_soft`。

---

## 3. 结合后的 forward 流程

### 3.1 训练模式（masked，全量计算，先保证正确性）

```text
输入: hidden_states X  [B, T, D_model]

1. Q, gate_logits = split(q_proj(X))
   Q: [B, H, T, D]
   gate_logits: [B, T, H]
   gate_soft = sigmoid(gate_logits)
   gate_hard = (gate_soft > tau)
   gate_ste  = gate_hard + gate_soft - gate_soft.detach()

2. K_raw, V_raw = k_proj(X), v_proj(X)
   写入 raw KV cache（无条件，全部 token 都写）

3. 维护 compressed KV cache（无条件，与 gate 无关）：
   每当某个 block 累计满 (block_size l, stride d):
       K_cmp[block], V_cmp[block] = Compress(K_raw[block], V_raw[block])
       写入 compressed KV cache

4. Local path（无条件计算）：
   o_swa = SlidingWindowAttention(Q, K_raw[-w:], V_raw[-w:])

5. Distant path（全量计算，用于训练期 mask 验证）：
   o_cmp, p_cmp = Attention(Q, K_cmp, V_cmp)
   selected_blocks = TopN(BlockScore(p_cmp)) ∪ sink_block ∪ local_blocks
   K_sel, V_sel = gather(K_raw, V_raw, selected_blocks)
   o_sel = Attention(Q, K_sel, V_sel)
   o_distant = g_cmp_nsa * o_cmp + g_sel_nsa * o_sel
   （g_cmp_nsa, g_sel_nsa 是 NSA 原生的 soft fusion 权重，来自 NSA 自己的小 MLP）

6. 融合：
   gate_b = gate_ste.unsqueeze(-1)     # [B, T, H, 1]
   o = o_swa + gate_b * o_distant      # [B, H, T, D] 或对应 reshape

7. reshape + o_proj:
   attn_output = reshape(o) -> [B, T, H*D]
   attn_output = o_proj(attn_output)

8. loss:
   loss = LM_loss + lambda * mean(gate_soft)
```

### 3.2 推理模式（compute-skipping，真正省算力）

```text
对每个 token t、每个 head h：

1. 计算 Q[t,h], gate_logits[t,h]（fused q_proj，必须算，成本很低）
2. 计算 K_raw[t], V_raw[t]，写入 raw KV cache（无条件）
3. 若 t 使某 block 凑满：计算并写入 compressed KV cache（无条件，摊销成本极低）
4. 计算 o_swa[t,h]（无条件，local path 恒定成本）
5. gate_hard[t,h] = (sigmoid(gate_logits[t,h]) > tau)
   if gate_hard[t,h] == 1:
       计算 o_cmp[t,h], p_cmp[t,h]           # compression attention
       用 p_cmp 选 top-n block
       gather + 计算 o_sel[t,h]              # selection attention
       o_distant[t,h] = g_cmp_nsa*o_cmp + g_sel_nsa*o_sel
   else:
       o_distant[t,h] = 0   # 完全跳过 ②③ 的 attention 计算，不产生也不需要
6. o[t,h] = o_swa[t,h] + gate_hard[t,h] * o_distant[t,h]
```

**正确性**：因为 compressed KV cache（步骤 3）的维护完全不依赖任何 token 自己的 gate 值，未来任何 gate=1 的 token 读取到的历史压缩记忆始终完整。gate 只影响"是否发起②③这两次读取型 attention 调用"，不影响"历史记忆是否被正确维护"。因此 compute-skipping 版本与 masked 训练版本在数学上等价，唯一区别是 gate=0 时不产生 ②③ 对应的中间张量。

---

## 4. 三类开销的性质（用于估算真实加速比）

```text
① Compress(K_raw, V_raw) -> K_cmp, V_cmp
   纯线性/MLP 变换，无 Q，无 softmax
   成本 O(block_size)，每 stride 步才触发一次
   与 gate 无关，恒定执行，几乎免费

② Attention(Q, K_cmp, V_cmp) -> o_cmp, p_cmp
   真实 attention，成本 O(M)，M = 压缩块数量（随历史增长）
   受 gate 控制，gate=0 时跳过

③ TopN select + gather + Attention(Q, K_sel, V_sel) -> o_sel
   真实 attention + 索引 gather，成本 O(n × l')，通常三者中最贵
   受 gate 控制，且依赖②的输出，②跳过则③自动跳过
```

gate=0 时节省的是②③（尤其是③，NSA 里最贵的部分），①始终保留但本身开销可忽略。

---

## 5. 为什么不采用"只 gate selection，compression 常开"的变体

曾讨论过的另一版本：

```text
o = g_swa*o_swa + g_cmp*o_cmp + gate * g_sel*o_sel   # (不采用)
```

这不是本方案（v1）。原因：

- 这样 compression attention（②）仍然对所有 token 无条件计算，不符合 AHA 严格的 "All（=local+全部distant）or Here（=local only）" 二元语义——AHA 的核心主张是整块跳过 global access，不是跳过 global access 的一部分。
- 保留它作为**备选变体（v2）**记录：如果 v1 训练不稳定或稀疏度上不去，可以退回 v2，只 gate 掉③（最贵的部分），②仍常开、且复用其结果做 selection 打分。v2 的好处是改动更小、更贴近原始 NSA 的信息流；代价是它不是纯粹的 AHA 化,论文表述需要弱化 "All-or-Here" 的类比。

---

## 6. Shape 速查表

```text
B: batch size
T: sequence length (query)
H: num attention heads (or num_kv_heads groups,视 GQA 设置)
D: head_dim
M: compressed block 数量 (≈ 历史长度 / stride d)
M_sel: selection block 数量 (block size l')
n: 每个 query 选中的 selection block 数（含强制 sink/local block）
w: sliding window 大小

Q:            [B, H, T, D]
gate_logits:  [B, T, H]
gate_soft:    [B, T, H]
gate_hard:    [B, T, H]
K_raw, V_raw: [B, H_kv, T, D]      (H_kv 可能 < H，GQA)
K_cmp, V_cmp: [B, H_kv, M, D]
p_cmp:        [B, H, T, M]
o_cmp:        [B, H, T, D]
K_sel, V_sel: [B, H, T, n*l', D]   (每个 query 各自 gather，不同 query 选块不同)
o_sel:        [B, H, T, D]
o_swa:        [B, H, T, D]
o_distant:    [B, H, T, D]
o (final):    [B, H, T, D] -> reshape -> [B, T, H*D] -> o_proj -> [B, T, D_model]
```

---

## 7. 训练细节 / 超参

```text
tau (门控阈值):          0.5（沿用 AHA 默认值）
lambda (L1 正则系数):     建议从 3e-4 起（沿用 AHA 默认值），需针对 NSA 重新扫描
sliding window size w:    与 NSA 预训练时的 w 保持一致（不建议改变，否则 local 语义漂移）
selection block size l':  沿用 NSA 预训练配置，不建议改
gate 初始化:              建议初始化 gate_logits 偏置使 gate_soft 初始接近 1
                          （即初始等价于原始 NSA，逐步在训练中学习何时可以跳过 distant path，
                           避免一开始就 collapse 到全 0）
```

Loss：

```text
loss = LM_loss + lambda * mean(gate_soft)
```

Router collapse 风险与应对（参考 rebuttal 中对 L2A 的讨论）：

- 若采用冻结 backbone、只训练 router 的参数高效设置，容易出现 gate 被正则压到全 0（因为 gate=0 的 token 在 next-token loss 下拿不到梯度）。
- 应对：训练时以小概率（如 5%）强制对所有 token 计算 distant path 并计入 loss，为 router 持续提供梯度信号；或采用端到端全参数微调（该设置下 backbone 表示本身在变化，gate collapse 风险显著降低，参考 AHA 论文的实验现象）。

---

## 8. KV Cache 维护规则（推理期，务必遵守）

```text
写入（无条件，任何 gate 值都要做）：
  - raw KV cache: 每个 token 的 K_raw, V_raw 必须写入
  - compressed KV cache: 每当一个 block 凑满，必须压缩并写入
    （压缩计算只依赖 block 内 raw KV，与该 block 内任何 token 自己的 gate 值无关）

读取（gate 决定，可跳过）：
  - compression attention（Q 查询 K_cmp/V_cmp）：gate=0 时跳过
  - selection attention（Q 查询 gather 出的 K_sel/V_sel）：gate=0 时跳过

不允许的错误实现：
  - 用当前 token 的 gate 值决定是否把它自己的 K/V 写入 raw cache 或参与未来 block 压缩
    （这会导致未来 gate=1 的 token 读到不完整历史，结果错误）
```

---

## 9. 实现落地建议

1. **基础代码**：以 `fla-org/flash-linear-attention` 的 `fla/layers/nsa.py::NativeSparseAttention` 为起点，在其 `q_proj` 处按第 2 节方式扩展输出通道，插入 gate 逻辑。
2. **第一阶段（验证正确性/稀疏度）**：只做 3.1 节的 masked 训练版本，不追求真实加速，先确认：
   - gate 是否 collapse；
   - 稀疏度 vs. 性能的 Pareto 曲线是否合理；
   - per-head 是否出现类似 AHA 论文里的 "always-on head" 长尾分布。
3. **第二阶段（真实加速）**：实现 3.2 节的 compute-skipping 推理路径，理想情况下参考 L2A 的 active-query compaction 思路，把 gate=1 的 token 压缩成连续 buffer 再统一喂给 NSA 的 selection Triton kernel，避免同 batch 内不同 token 走不同路径导致的 kernel 效率损失。
4. **消融实验建议**：
   - v1（本方案，gate 整个 distant path） vs. v2（只 gate selection，见第 5 节）；
   - 不同 tau、lambda 组合下的稀疏度-性能曲线；
   - per-token-per-head 粒度 vs. per-token 粒度（退化成类似 L2A）的对比，用于支撑"细粒度门控更优"的论点。

---

## 10. 代码落地情况（`ahnsa`，基于 `fla-org/flash-linear-attention`）

已在 `flash-linear-attention/` 这份本地 checkout 中新增了 `ahnsa` 模块，实现了本文档描述的 v1 方案（gate 整个 distant path），训练与推理路径都已实现：

```text
fla/ops/ahnsa/
  parallel.py   ahnsa_attn(...): 组合 NSA 的 distant path（内部仍调用未经修改的
                fla.ops.nsa.parallel.parallel_nsa，从而复用其已有的、经过测试的
                Triton kernel 及其 autograd 支持）与本地 sliding-window 分支
                （sliding_window_attention(...)，直接调 flash_attn，不经过
                compression/selection kernel）。
                - training=True，或 training=False 但本次 forward 里存在任何
                  gate_hard==1 的 (batch, token, head)：走"masked"组合，即
                  4.1 节所述——distant path 仍对所有 token 全量计算，只是结果
                  乘上 STE 硬 gate 后再与 local path 相加（对应第 3.1 节）。
                - training=False 且本次 forward 的 gate_hard 全为 0：直接跳过
                  compression + top-k + selection 三个 kernel，只算
                  sliding_window_attention，对应第 3.2 / 4 节的"真实跳过计算"。
                  这是 call 粒度（整批、整头）的粗粒度跳过，不是 per-head 的细粒度
                  跳过（第 9 节第 3 点提到的 active-query compaction 未实现）。
  naive.py      naive_ahnsa(...): 纯 PyTorch 参考实现（对齐 fla.ops.nsa.naive
                的写法），只用于正确性校验，不做任何 compute-skipping。

fla/layers/ahnsa.py
  AHNSAAttention: 对齐 fla.layers.nsa.NativeSparseAttention，区别是：
    - q_proj 输出多出 num_heads 个通道，作为 AHA router 的 logits（第 2 节的
      融合 q_proj 设计，模仿 xuan-luo/AHA-OLMO2 的 modeling_faolmo.py）；
    - 额外的 aha_gate_bias（每头一个可学习标量，初始值默认 4.0，即
      sigmoid(4)≈0.98）叠加在 router logits 上，保证训练初期 AHNSA 行为
      接近原始 NSA（第 7 节的 collapse 缓解措施之一）；
    - forward 额外返回 aha_gate_soft / aha_gate_hard（形状 [B, T, H]），供
      模型层算稀疏正则。
    - raw KV cache 的写入逻辑完全不变（无条件写入），对应第 8 节的约束；
      compressed KV 在这份实现里本来就是"每次 forward 都从完整 raw cache
      重新算"（fla 原生 parallel_nsa 内部行为），并非跨 step 增量维护的独立
      cache，所以第 8 节里"compressed cache 必须无条件维护"的要求在这份代码
      里是自动满足的——只要 raw cache 无条件写入，跳过某个 step 的
      compression kernel 调用绝不会丢历史信息。

fla/models/ahnsa/
  configuration_ahnsa.py   AHNSAConfig：在 NSAConfig 基础上加 aha_tau /
                            aha_lambda / aha_gate_bias_init 三个超参。
  modeling_ahnsa.py         AHNSABlock / AHNSAModel / AHNSAForCausalLM：结构
                            对齐 modeling_nsa.py，但去掉了 NSA 里的 attnres
                            旁支功能（与本研究问题无关，去掉以降低实现风险）；
                            在 ForCausalLM.forward 里按第 7 节实现了稀疏正则：
                            loss = LM_loss + aha_lambda * mean(aha_gate_soft)
                            （只在 training 时加，且用去掉最后一个 token 的
                            "shift" 版本，因为最后一个 token 在 teacher-forcing
                            下没有下一个 token 的标签，不该被正则化）。

已在以下位置完成注册（与 NSA 的注册方式一致）：
  fla/ops/__init__.py     导出 ahnsa_attn
  fla/layers/__init__.py  导出 AHNSAAttention
  fla/models/__init__.py  导出 AHNSAConfig / AHNSAForCausalLM / AHNSAModel，
                          并通过 fla/models/ahnsa/__init__.py 里的
                          AutoConfig.register / AutoModel.register /
                          AutoModelForCausalLM.register 接入 transformers 的
                          Auto 类，可直接 AutoModelForCausalLM.from_config(...)
                          或配合 AHNSAConfig(model_type='ahnsa') 使用。
```

尚未做（按第 9 节列为后续工作，本次未实现）：
- per-head/per-token 细粒度的 compute-skipping（需要 active-query compaction
  或按 GQA group 边界重组 kernel，风险较高，未在无法跑测试的情况下引入）；
- 消融实验脚本、与 naive_ahnsa 的数值对齐测试（用户要求本轮只改代码、不跑测试）。
