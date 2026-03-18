# Hetero TP for SSM-FA: Background & Details

Detailed reference for [discussion.md](discussion.md). Covers how hetero TP
works for attention, why it breaks for Mamba conv state, reference dimensions,
and the Design B staging buffer implementation.

---

## tp_ratio

`tp_ratio` is an **integer** (not a fraction). The sign encodes direction:

| Scenario | tp_ratio | Meaning |
|---|---|---|
| P TP=1, D TP=1 | +1 | Homogeneous |
| P TP=1, D TP=2 | +2 | 2 D workers share 1 P worker |
| P TP=2, D TP=1 | -2 | 1 D worker reads from 2 P workers |

- Positive: D reads a **slice** of P's KV block
- Negative: D reads from **multiple** P workers, stitches together

Source: `utils.py` `TpKVTopology.tp_ratio()`

---

## Regular Attention: How Hetero TP Works

Attention KV cache layout (HND): `[num_blocks, num_kv_heads, block_size, head_dim]`

KV heads are **contiguous in memory** → can slice with a byte offset.

### Case 1: D TP > P TP (tp_ratio > 0)

Example: P TP=1 (4 KV heads), D TP=2 (2 heads each)

```
P Worker0's block (4 heads, contiguous):
┌──────┬──────┬──────┬──────┐
│  H0  │  H1  │  H2  │  H3  │
└──────┴──────┴──────┴──────┘
  ^                    ^
  │                    │
  D Worker0            D Worker1
  rank_offset=0        rank_offset=2*head_bytes
  reads [H0,H1]       reads [H2,H3]
```

**Registration** (`add_remote_agent`, line ~1911):
```python
rank_offset = self.tp_rank % tp_ratio * remote_kv_block_len
addr = base_addr + block_offset + rank_offset
blocks_data.append((addr, local_block_len, device_id))
```

**Which remote rank?** (`get_target_remote_ranks`):
```python
return [self.tp_rank // tp_ratio]
# D Worker0: 0//2 = 0 → P rank 0
# D Worker1: 1//2 = 0 → P rank 0  (same P worker)
```

**Transfer:** 1 RDMA READ per D worker, each reading its contiguous head slice.

### Case 2: P TP > D TP (tp_ratio < 0)

Example: P TP=2 (2 heads each), D TP=1 (needs all 4 heads)

```
P Worker0: [H0 | H1]    P Worker1: [H2 | H3]
              │                        │
              └──── D Worker0 ─────────┘
              reads from both, writes into split local buffer
```

**Local buffer split** (`add_remote_agent`, line ~1862):
```python
for i in range(-tp_ratio):  # i = 0, 1
    remote_block_len = local_block_len // (-tp_ratio)
    addr = addr + i * remote_block_len
    # Chunk 0 → receives from P Worker0
    # Chunk 1 → receives from P Worker1
```

**Which remote ranks?** (`get_target_remote_ranks`):
```python
return [self.tp_rank * (-tp_ratio) + i for i in range(-tp_ratio)]
# D Worker0: [0, 1] → reads from P rank 0 and P rank 1
```

**Transfer** (`_read_blocks_for_req`, line ~2427): loop over remote ranks:
```python
for i, remote_rank in enumerate(remote_ranks):
    local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
    remote_xfer_side_handle = self.dst_xfer_side_handles[engine_id][remote_rank]
    self._read_blocks(...)
```
2 RDMA READs: one from each P worker into each local chunk.

### Summary

| | D TP > P TP (tp_ratio > 0) | P TP > D TP (tp_ratio < 0) |
|---|---|---|
| D reads from | 1 P worker (sliced) | Multiple P workers |
| Mechanism | `rank_offset` into remote block | Split local block into chunks |
| # RDMA READs | 1 per D worker | `abs(tp_ratio)` per D worker |
| Key requirement | Heads are contiguous in memory | Same |

---

## Why This Breaks for Mamba Conv State

### Mamba2 state shapes (Nemotron-3-Nano)

| TP | Conv state | SSM state |
|---|---|---|
| 1 | (3, 6144) | (64, 64, 128) |
| 2 | (3, 3072) | (32, 64, 128) |

### Conv state: mixed dimension

`conv_dim = intermediate_size + 2 * n_groups * state_size`

The conv state stores the last `conv_kernel - 1` time steps of `hidden_states_B_C`,
which is a concatenation of three sub-components:

```python
# mamba_mixer2.py line 470
self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
    hidden_states_B_C,
    [
        intermediate_size // tp_size,          # x (hidden states)
        groups_ssm_state_size // tp_size,      # B (input matrix)
        groups_ssm_state_size // tp_size,      # C (output matrix)
    ],
    dim=-1,
)
```

At TP=1: `conv_dim = [x:4096 | B:1024 | C:1024] = 6144`

When going to TP=2, each worker needs:
```
[x:2048 | B:512 | C:512] = 3072
```

But the TP=1 memory layout is `[x_full(4096) | B_full(1024) | C_full(1024)]`.
You **cannot** get `[x_half | B_half | C_half]` with a single contiguous slice:

```
TP=1 layout:     [xxxx....xxxx | BB..BB | CC..CC]
                  ├── 4096 ────┤├ 1024 ─┤├ 1024 ─┤

What TP=2 worker0 needs:
                  [xx....xx | B..B | C..C]
                  ├─ 2048 ──┤├ 512┤├ 512┤

These are NOT contiguous in the TP=1 layout!
You'd need 3 separate reads: x[0:2048], B[0:512], C[0:512]
```

### SSM state: head-like dimension (tractable)

SSM state shape: `(num_heads / tp_size, head_dim, state_size)`

The first dimension (`num_heads`) is cleanly splittable, just like attention heads:

```
TP=1: [(H0,64,128) | (H1,64,128) | ... | (H63,64,128)]

TP=2 worker0 needs: [(H0,64,128) | ... | (H31,64,128)]  ← contiguous slice!
TP=2 worker1 needs: [(H32,64,128) | ... | (H63,64,128)] ← contiguous slice!
```

This can use the same `rank_offset` approach as attention.

---

## Reference Dimensions (Nemotron-Nano-30B)

```
Model config:
  intermediate_size = 4096      (the "x" expansion size)
  n_groups          = 8         (SSM groups)
  state_size        = 128       (per-group state)
  d_conv            = 4         (conv kernel width → conv_rows = d_conv - 1 = 3)
  groups_ss         = n_groups × state_size = 1024

Conv dim per TP:
  Full (TP=1):  x=4096  B=1024  C=1024  → total conv_dim = 6144
  Half (TP=2):  x=2048  B=512   C=512   → total conv_dim = 3072

Memory per block (conv, bf16):
  TP=1: 3 rows × 6144 × 2B = 36,864 bytes
  TP=2: 3 rows × 3072 × 2B = 18,432 bytes
```

### conv_rows = d_conv − 1 (model-dependent)

`d_conv` is a model config parameter (the 1D conv kernel width). Mamba2 models
typically use `d_conv=4`, giving `conv_rows=3`. But this is NOT universal:

| Architecture | d_conv | conv_rows |
|---|---|---|
| Mamba2 (Nemotron-H, Nano-30B) | 4 | 3 |
| Mamba1 (original) | 4 | 3 |
| Future models | configurable | d_conv − 1 |

### Concrete numbers

Per Mamba layer per block:
- Conv state: 3 × 6144 × 4 bytes(float32) ≈ **73,728 bytes** (~72KB)
- SSM state:  64 × 64 × 128 × 4 bytes ≈ **2,097,152 bytes** (~2MB)
- Total Mamba page: ~2.1MB

Conv is ~3.5% of the total Mamba state. Replicating or recomputing it has
minimal bandwidth impact.

---

## Design B: Staging Buffer Implementation (First Try)

**Principle**: SSM = rank_offset (like attention), Conv = full read + local extract.

### RDMA constraint

NIXL uses a READ model: D initiates reads from P's registered memory.
Descriptors are registered at handshake time (once), not per-transfer.
Descriptor pairs must have matching sizes: `len(local_desc) == len(remote_desc)`.

P's memory layout is whatever the Mamba kernel wrote — D cannot control it.

### SSM: same as attention

SSM state shape: `(num_heads/TP, head_dim, state_size)` — heads are contiguous.

- D>P: `rank_offset` into P's SSM → reads contiguous head slice → direct to D's cache
- P>D: split D's local SSM buffer into chunks → read from multiple P workers

No new code needed — reuse existing attention hetero TP logic.

### Conv: staging buffer + post-processing

Conv state is `[x | B | C]` with non-contiguous TP shards.
D's conv slot is smaller than P's full conv → can't receive full read in-place.

**Registration time** (`add_remote_agent`):
- Conv remote descs: full P conv size, no `rank_offset`
- Conv local descs: point to **staging buffer** (sized to P's full conv)
- SSM remote descs: `rank_offset` applied (like attention)
- SSM local descs: point to D's actual SSM cache

**Transfer time** (`_read_blocks`):
- Conv: RDMA reads P's full conv → D's staging buffer
- SSM: RDMA reads P's head slice → D's SSM cache (direct)

**Post-transfer** (`get_finished`, after RDMA completes):
```python
x, B, C = torch.split(staging, [interm, groups_ss, groups_ss], dim=-1)
my_conv = torch.cat([
    x[:, rank*sx:(rank+1)*sx],
    B[:, rank*sb:(rank+1)*sb],
    C[:, rank*sb:(rank+1)*sb],
], dim=-1)
local_conv_cache[:] = my_conv
```

### Staging buffer sizing

Per Mamba layer per block: P's conv_bytes (e.g., 73KB at P TP=1).
Total: `conv_bytes × num_mamba_layers × max_concurrent_transfers`.
For Nemotron (P TP=1): ~73KB × 32 layers × ~16 concurrent ≈ 37MB. Manageable.

### Code changes needed

| File / Function | Change |
|---|---|
| `_validate_remote_agent_handshake` | Relax `assert tp_ratio == 1` for Mamba |
| `add_remote_agent` / `register_remote_blocks` | Branch conv vs SSM: different sizes + offsets |
| `register_local_xfer_handler` (or new variant) | Conv local descs → staging buffer |
| `get_finished` | Add conv extraction post-processing hook |
| `MambaSpec` / connector init | Plumb `intermediate_size`, `n_groups`, `state_size` |

### P>D direction

D reads from multiple P workers. Each P has its own conv shard (valid `[x_half|B_half|C_half]`).
D receives each into a separate staging slot, then stitches:
```python
x_full = cat([x_from_p0, x_from_p1], dim=-1)
B_full = cat([B_from_p0, B_from_p1], dim=-1)
C_full = cat([C_from_p0, C_from_p1], dim=-1)
local_conv = cat([x_full, B_full, C_full], dim=-1)
```

---

*Source: PR #36687, nixl_connector.py, mamba_mixer2.py, mamba_utils.py*
*Model reference: Nemotron-3-Nano (Mamba2 + FlashAttention)*
