# Heterogeneous TP for Hybrid SSM-FA Models

Options for transferring Mamba conv state across different TP sizes in
disaggregated P/D serving. For background on how hetero TP works for attention
and why conv state breaks, see [background.md](background.md).

---

## The Problem

SSM state is head-sharded (like attention) → standard `rank_offset` works.
Conv state is `[x | B | C]` with three independently-sharded sub-components
→ **not** contiguous per TP shard:

```
P (TP=1) conv state per block (shape: 3 × 6144):
┌─────────────────────────────────────────────────────────────────┐
│ Row 0:  x₀(2048) │ x₁(2048) │ B₀(512) │ B₁(512) │ C₀│ C₁    │
│ Row 1:  ...                                                     │
│ Row 2:  ...                                                     │
└─────────────────────────────────────────────────────────────────┘

D rank 0 needs:  [x₀|B₀|C₀] per row  → non-contiguous in P's layout
NIXL can only read contiguous byte ranges → single (addr, len) won't work
```

---

## Options

All options share the same SSM approach: `rank_offset` on the head dimension
(identical to attention). They differ only in how **conv state** is handled.

### Option A: Multiple RDMA Reads

Issue separate RDMA reads for each sub-component (x, B, C) of each row.
Data lands directly in D's KV cache.

| Metric | Value |
|---|---|
| RDMA reads per block | `conv_rows × 3` = 9 |
| Bytes transferred | exact shard (zero waste) |
| Extra GPU memory | 0 |
| Post-processing | none |
| Descriptor count | 9× baseline |

**Pros:** zero extra memory, zero post-processing, minimum bandwidth
**Cons:** 9× RDMA descriptors → NIXL overhead; complex descriptor registration

---

### Option B: Staging Buffer + Post-Processing

D reads P's FULL conv into a GPU staging buffer, then splits/extracts its shard.

| Metric | Value |
|---|---|
| RDMA reads per block | 1 |
| Bytes transferred | full conv (2× shard) |
| Extra GPU memory | ~1.52 GiB staging on D |
| Post-processing | split + slice + cat + index_copy_ |
| Descriptor count | 1× baseline |

**Pros:** simple descriptors, fast GPU post-processing, minimal NIXL overhead
**Cons:** 2× bandwidth, ~1.5 GiB staging buffer on D

*Experimental implementation on branch `nixl-ssm-hetero-tp-rebased`.*

See [background.md § Design B](background.md#design-b-staging-buffer-implementation-first-try)
for full implementation details.

---

### Option C: Recompute Conv on Decoder

Only transfer SSM state; recompute conv from prompt tokens on D.

**Pros:** most bandwidth-efficient (SSM only)
**Cons:** requires model weights during KV transfer, added decode latency

---

### Option D: Chunk-Interleaved Transpose (Implemented)

Rearrange P's conv state using a TP-agnostic chunk-interleaved transpose so
that for **any** D TP, each rank's shard is contiguous. D reads with standard
`rank_offset`, then applies an inverse permutation.

**The chunk:** `g = gcd(x_dim, B_dim)` defines a repeating unit preserving
the x:B:C ratio. Any valid `target_tp` divides `g` → each D rank reads
`g/tp` contiguous chunks = 1 RDMA read.

```
Original:     [──── x (4096) ────│─ B (1024) ─│─ C (1024) ─]

Chunk-interleaved (1024 chunks, each 4x + 1B + 1C cols × 3 rows):
┌────────────┬────────────┬─────┬──────────────────┐
│  Chunk 0   │  Chunk 1   │ ... │  Chunk 1023      │
└────────────┴────────────┴─────┴──────────────────┘
TP=2:  │◄── rank 0: 512 chunks ──►│◄── rank 1: 512 chunks ──►│
TP=4:  │◄r0: 256►│◄r1: 256►│◄r2: 256►│◄r3: 256►│
```

| Metric | Value |
|---|---|
| RDMA reads per block | 1 |
| Bytes transferred | exact shard (zero waste) |
| Extra GPU memory | ~0.5 GiB shadow buffer on P |
| P pre-processing | 1 GPU gather (fwd permute) |
| D post-processing | 1 GPU gather (inv permute) |
| Descriptor count | 1× baseline |

**Pros:** TP-agnostic on P, minimum bandwidth, simple `rank_offset` descriptors,
net -79 lines vs Option B
**Cons:** requires P-side + D-side permutation, shadow buffer memory on P

---

## Comparison

| | A (Multi-RDMA) | B (Staging Buf) | C (Recompute) | D (Chunk-Interleave) |
|---|---|---|---|---|
| **RDMA reads/block** | conv_rows × 3 | 1 | 0 (conv) | 1 |
| **Bytes over RDMA** | exact shard | full conv (2×) | 0 (conv) | exact shard |
| **Extra GPU memory** | 0 | ~1.5 GiB (D) | 0 | ~0.5 GiB shadow (P) |
| **P pre-processing** | none | none | none | 1 GPU gather |
| **D post-processing** | none | split+extract+copy | recompute | 1 GPU gather |
| **Descriptor count** | 9× per block | 1× | 1× | 1× |
| **TP-agnostic on P** | yes | yes | yes | yes |
| **Implementation** | not started | ✅ experimental | not started | ✅ experimental |
| **Multi-D-TP** | free | free | free | free (TP-agnostic) |

---

## Design D Implementation

**Branch:** `nixl-ssm-hetero-tp-permute` (sha `0fefd00e6`)
**Net code impact:** -79 lines vs Option B (152 insertions, 231 deletions)

### Architecture

```
P: conv cache [x|B|C]  ──(fwd permute)──►  P: shadow buf [chunk-interleaved]
                                                   │
                                             RDMA (rank_offset → shard)
                                                   │
D: conv cache [x_r|B_r|C_r]  ◄──(inv permute)──  D: conv cache [chunk layout]
```

Both permutations are precomputed index tensors (built once at `register_kv_caches`).
At runtime, each is a single vectorized GPU gather — no Python loops.

### Flow

1. **Init (`register_kv_caches`):** P allocates shadow buffers, both sides build
   perm index tensors from model hyperparameters (`g = gcd(x_local, B_local)`)
2. **P-side (`start_load_kv`):** `_chunk_transpose_conv_inplace` — index_select
   finished blocks → `flat[:, perm_fwd]` → write to shadow buffer
3. **RDMA:** D reads from P's shadow buffer using standard `rank_offset`
4. **D-side (`get_finished`):** `_unchunk_transpose_conv_inplace` — `flat[:, perm_inv]`
   → restore original `(conv_rows, shard_dim)` layout

### Changes vs Option B

| Removed (Option B) | Added (Option D) |
|---|---|
| `_mamba_conv_staging_bufs` (D-side) | `_mamba_conv_shadow_bufs` (P-side) |
| `_allocate_mamba_conv_staging()` | `_mamba_chunk_perm_fwd/inv` index tensors |
| `_post_process_mamba_conv_staging()` | `_chunk_transpose_conv_inplace()` (P) |
| Staging handle swap in `_read_blocks_for_req` | `_unchunk_transpose_conv_inplace()` (D) |

### Results (1P1D, P_TP=1 D_TP=2, Nemotron-Nano-30B)

| Metric | Value |
|---|---|
| GSM8K accuracy (6 repeats) | mean=0.8384, std=0.0054 |
| KV cache hit rate | 100% (7.9M cumulative hits) |
| Transfer errors | 0 |
| Shadow buffer (P-side, 6 layers) | 525.66 MiB |
| Perm index size | 144 KiB/region (P), 72 KiB/region (D) |
| Transport | UCX cuda_ipc (NVLink zero-copy) |

### Known Limitations

- **Timing race (theoretical):** D could RDMA-read before P finishes permutation;
  mitigated by proxy round-trip latency; production fix: NIXL notification
- **Memory:** shadow buffer moved from D (Option B) to P — same total cost
- **2p1d (P_TP>D_TP):** blocked by pre-existing `main` limitation in
  `_validate_remote_agent_handshake`, not caused by Option D

---

*Source: PR #36687, nixl_connector.py, mamba_mixer2.py, mamba_utils.py*
*Model reference: Nemotron-3-Nano (Mamba2 + FlashAttention)*
