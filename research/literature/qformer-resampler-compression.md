# Query-Based Visual Resamplers: OmniDrive Q-Former & ORION QT-Former

**Lever 1 + 3** (visual resampling + temporal memory bank). This is the most direct answer to
the user's question: how SoTA controls the *number* of vision tokens fed to the LLM.

## The core idea

A pretrained ViT emits hundreds of patch tokens **per image**. A query-based resampler
(BLIP-2's Q-Former, Flamingo's Perceiver Resampler) inserts a small, fixed set of **learnable
query tokens** that cross-attend to the ViT features and absorb them into a controllable output
size. The LLM then sees `N_query` tokens instead of `N_patches × N_frames`. `N_query` is a
hyperparameter you set — *this is the "controllable number of vision tokens"* the user wants.

```
[ViT patch tokens: N_patch × N_view × N_frame]  →  cross-attn  →  [Q queries: fixed, small]  →  LLM
                                                        ↑
                                              learnable query tokens
```

## OmniDrive (CVPR 2025) — referenced across the library, not a standalone paper here

OmniDrive is cited by ≥15 papers in the library as the canonical "Q-Former / 3D-query"
multi-view→token approach (e.g. it appears in OneDrive, UniDriveVLA, ORION, Reasoning-VLA
comparison tables). Key points reconstructed from those references + the method:
- Uses a **Q-Former3D**: 2D multi-view image features are lifted and compressed into a fixed set
  of **3D-aware carrier queries** (sparse, position-encoded) before the LLM.
- This decouples LLM sequence length from camera count/resolution — adding cameras does not
  linearly grow LLM tokens.
- Open-loop nuScenes: 0.33 avg L2 (text-output VLM). **Latency 3,727 ms** (it is AR over text) —
  so OmniDrive shows the *compression* idea but is *not* a speed exemplar; the slow part is its
  autoregressive text decoding, not the visual resampler.
- ⚠️ Not in `raw/papers/`; numbers above are from library reference tables. Read the original
  (arXiv 2405.01533) before citing specifics.

**Takeaway for DeepSight**: the resampler is what makes "6 surround + 4 history frames" cheap.
The token count becomes a knob, not a function of camera count.

## ORION QT-Former (Bench2Drive 77.74 DS) — full note in library `wiki/sources/orion.md`

ORION is the most concrete, in-library implementation of the resampler pattern, and it targets
**the same benchmark as DeepSight (Bench2Drive)**. Its **QT-Former** compresses all multi-view
images into three query banks:

| Query type | Count | Role |
|---|---|---|
| Scene queries $Q_s$ | **512** | key info of current frame |
| Perception queries $Q_p$ | **600** | detection / traffic state / motion (supervised) |
| History queries $Q_h$ | **16** | summarize a FIFO memory bank of past frames |

Mechanism:
1. $Q_s, Q_p$ self-attend, then **cross-attend to multi-view image features** (with 3D positional
   encoding) → fixed-size scene/perception tokens.
2. **History queries first attend to a FIFO memory bank** $M$ (16 frames of past $Q_h$, with
   timestamp embeddings), *then* attend to the current scene queries:
   `Q_h = CA(Q_h, M+P_t, M+P_t)`; `Q̂_h = CA(Q_h, Q_s, Q_s)`.
   This is **Lever 3**: long temporal history is compressed into 16 recurrent tokens rather than
   re-encoding 4+ raw history frames as full image-token blocks.
3. Only scene tokens $x_s$ + history tokens $x_h$ (projected by a 2-layer MLP) go to the LLM
   (Vicuna v1.5 + LoRA). Perception queries are an auxiliary supervised head, not LLM input.

### Two ablations that directly inform DeepSight's design knobs
- **History-query count** (Table 5): $N_h{=}0$ → 65.1 DS; $N_h{=}8$ → 68.1; **$N_h{=}16$ → 74.1**;
  $N_h{=}32$ → 62.5. *There is a clear optimum*; too much history capacity drowns the current
  frame. Don't assume "more history tokens = better."
- **Traffic-state supervision** on $Q_p$ is the single biggest contributor (+18.3 DS). I.e. the
  compressed queries need an *auxiliary perception objective* to stay informative — pure
  planning loss under-trains them. DeepSight already has a DINOv3 world-feature objective, which
  plays an analogous "keep the compressed tokens grounded" role.

## How this maps onto DeepSight

Current DeepSight: 10 input frames → ~3,000 raw ViT tokens (no resampling; Qwen's native
`masked_scatter` path). The cheapest high-impact change is to **insert a resampler between the
ViT and the LLM**:

- Replace per-frame 299 patch tokens with, e.g., 64–128 scene queries per frame, or a single
  pooled bank across the 6 surround views (à la ORION's 512 scene queries total).
- Move the 4 history frames into a **16-query memory bank** instead of 4 full image-token blocks
  (~1,200 tokens → 16).
- Keep the DINOv3 future-BEV objective as the grounding signal for the queries (analogous to
  ORION's traffic-state supervision).

Estimated input-vision budget: ~3,000 → **~600 tokens** (≈5× shorter input prefix) with the
ORION-style configuration, before touching the 1,305 world-query block.

### Risks / things to verify
- Qwen2.5-VL's native dynamic-resolution path and `masked_scatter` fusion (see
  [SRC_CODE_MAP.md](../../SRC_CODE_MAP.md) §2.5) must be replaced/bypassed — the resampler
  changes the image-token contract.
- Resamplers usually need their own warm-up (ORION/OmniDrive both pretrain the Q-Former with a
  perception/VQA objective before action training). Expect a staged recipe, not a drop-in.
- ORION explicitly lists "memory bank cost grows with sequence length" and "no latency benchmark"
  as limitations — the resampler compresses *LLM* tokens but the ViT still runs on every frame.
