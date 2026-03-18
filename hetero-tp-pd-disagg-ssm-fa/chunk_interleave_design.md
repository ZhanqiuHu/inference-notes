# Chunk-Interleaved Permutation for Hetero TP Mamba Conv Transfer

## 1. Problem

In vLLM P/D disaggregated serving, Prefiller (P) and Decoder (D) can run at
different tensor-parallel sizes (hetero TP). For attention KV, each D rank
reads its head slice from P via a contiguous byte range (`rank_offset`).

Mamba conv state layout is `[x | B | C]` — three components concatenated
along the last dimension. A TP shard requires non-contiguous slices from each
component. A single contiguous RDMA read cannot extract a valid shard.

```
P (TP=1):  [x_full (4096 cols) | B_full (1024 cols) | C_full (1024 cols)]

D (TP=2) rank 0 needs:  [x[0:2048] | B[0:512] | C[0:512]]
                          ^^^^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^
                          NOT contiguous in P's layout
```

## 2. Solution: Chunk-Interleaved Transpose

P rearranges its conv state so that each D TP shard is a contiguous byte
range. The rearrangement depends only on model hyperparameters — not on D's
TP size.

### 2.1 Chunk Definition

```
g = gcd(x_dim, B_dim)           — number of chunks
x_ratio = x_dim / g             — x-columns per chunk
B_ratio = C_ratio = B_dim / g   — B/C-columns per chunk
chunk_cols = x_ratio + B_ratio + C_ratio
```

Each chunk contains proportional slices of x, B, C. Since any valid TP
divides both `x_dim` and `B_dim`, it also divides `g = gcd(x_dim, B_dim)`.
Therefore every TP shard is an integer number of consecutive chunks.

### 2.2 Layout Transformation

Original: `(conv_rows, conv_dim)` — row-major, components concatenated.

Chunk-interleaved transposed: chunks laid out sequentially, each chunk's
columns transposed to make `conv_rows` values per column contiguous.

```
Original (conv_rows=3, x_ratio=4, B_ratio=1, C_ratio=1):

  Row 0: [x0  x1  x2  x3  | B0  | C0  | x4  x5  x6  x7  | B1  | C1  | ...]
  Row 1: [x0' x1' x2' x3' | B0' | C0' | x4' x5' x6' x7' | B1' | C1' | ...]
  Row 2: [x0" x1" x2" x3" | B0" | C0" | x4" x5" x6" x7" | B1" | C1" | ...]

Chunk-interleaved transposed (flat):

  Chunk 0: [x0 x0' x0" | x1 x1' x1" | x2 x2' x2" | x3 x3' x3" | B0 B0' B0" | C0 C0' C0"]
  Chunk 1: [x4 x4' x4" | x5 x5' x5" | x6 x6' x6" | x7 x7' x7" | B1 B1' B1" | C1 C1' C1"]
  ...

  TP=2: ◄── rank 0: chunks 0..511 ──►◄── rank 1: chunks 512..1023 ──►
        each is 1 contiguous RDMA read
```

### 2.3 Concrete Numbers (Nemotron-Nano-30B-A3B)

```
x_dim=4096  B_dim=C_dim=1024  conv_rows=3  dtype=bf16

g=1024  x_ratio=4  B_ratio=1  chunk_cols=6
chunk_bytes = 6 × 3 × 2 = 36 bytes

TP=1:  1024 chunks = 36.0 KiB/block
TP=2:   512 chunks = 18.0 KiB/block
TP=4:   256 chunks =  9.0 KiB/block
TP=8:   128 chunks =  7.5 KiB/block → always 1 contiguous read
```

## 3. Implementation

All changes in `nixl_connector.py` within `NixlConnectorWorker`.

### 3.1 Init (once, in `register_kv_caches`)

Build two permutation index tensors from model hyperparameters:

- **Forward** (`_mamba_chunk_perm_fwd`): original flat → chunk-interleaved transposed
- **Inverse** (`_mamba_chunk_perm_inv`): chunk-interleaved transposed → original shard

Both are `torch.long` tensors on GPU. Size: `conv_rows × conv_dim × 8` bytes
(~144 KiB for Nano TP=1, ~72 KiB for TP=2).

### 3.2 P-side (per request, in `start_load_kv`)

```python
selected = conv_cache.index_select(0, block_ids)   # copy blocks out
flat = selected.reshape(N, -1)                       # (N, conv_rows*conv_dim)
permuted = flat[:, perm_fwd]                         # GPU gather
conv_cache[block_ids] = permuted.reshape(...)        # write back in-place
```

### 3.3 RDMA Transfer

Standard descriptor registration — same as attention:

```python
addr = base_addr + block_id * page_size + rank_offset
```

`page_size` is the shared tensor's per-block stride (conv + SSM + padding).
`rank_offset` selects the shard within the conv portion.

### 3.4 D-side (after transfer completes, in `get_finished`)

```python
selected = conv_cache.index_select(0, block_ids)
flat = selected.reshape(N, -1)
restored = flat[:, perm_inv]                         # GPU gather
conv_cache[block_ids] = restored.reshape(...)
```

## 4. Why It Works for Any TP

**Lemma:** Any valid `D_TP` divides `g = gcd(x_dim, B_dim)`.

*Proof:* The model requires `D_TP | x_dim` and `D_TP | B_dim` for TP
sharding. Any common divisor of `x_dim` and `B_dim` divides their GCD.

**Consequence:** Each D rank gets `g / D_TP` consecutive, equal-sized chunks
— a single contiguous byte range. Each range contains the correct
proportional shard of x, B, and C.

## 5. Memory Overhead

| Component | Size | When |
|---|---|---|
| Perm index (P) | ~144 KiB/region | Allocated once at init |
| Perm index (D) | ~72 KiB/region | Allocated once at init |
| Shadow buffer | **None** | Removed (in-place permutation) |
| Staging buffer | **None** | Not needed |

Total extra memory: < 1 MiB across all regions.

## 6. Comparison with Previous Approaches

| | Option B (staging, on main) | This (chunk-interleave) |
|---|---|---|
| RDMA reads/rank | 1 (full conv) | 1 (shard only) |
| RDMA bytes/rank | full P conv | 1/tp_ratio of P conv |
| Extra GPU memory | staging buf on D | ~0 (perm index only) |
| D post-processing | split + cat + index_copy | 1 GPU gather |
| P pre-processing | none | 1 GPU gather (in-place) |
| TP-agnostic on P | N/A | Yes |
| Arbitrary P/D TP | Untested | Works for all valid combos |
