# Plan: Making DeepSight Lightweight for Fast Idea Iteration

Goal (from the request): cut the input-token budget and training cost so ideas can be tested
quickly on far fewer GPUs, **without** giving up much Bench2Drive performance. This plan is
derived from the SoTA survey in [literature/](literature/) — start with
[literature/00-taxonomy.md](literature/00-taxonomy.md).

## 1. Where DeepSight's cost actually is

Per-step LLM sequence today (verify exact numbers against the live config):

| Block | Tokens | Mechanism | Reference |
|---|---|---|---|
| Input vision: 4 history front + 6 surround @ 364×644 | **~3,000** (~299/frame) | Qwen ViT → `masked_scatter` | [infer_for_debug.py:37](../src/infer_for_debug.py#L37) |
| World queries `<\|bev_token_i\|>` (5 future BEV) | **1,305** | prefilled learnable embeddings | [SRC_CODE_MAP.md](../SRC_CODE_MAP.md) §3 |
| CoT + waypoints | variable | text via `lm_head` | SRC_CODE_MAP §2.5 |

Two compounding costs: **(a) long sequence** (~4,300+ tokens, O(L²) attention) and **(b) full
fine-tune of the 3B backbone** (64× H20). The biggest, cheapest wins attack both.

## 2. Two independent axes — do both

- **Axis A — shorten the sequence** (Levers 1–3, 6, 7): fewer input vision tokens, fewer world
  tokens, fewer CoT tokens.
- **Axis B — shrink the trainable surface** (Lever 5, OneDrive): freeze/LoRA the backbone, train a
  small expert. This is what actually drops the GPU count for fast iteration.

They compose: *frozen backbone + resampled input + compact world tokens* is the lightweight target.

## 3. Recommended experiment ladder (cheapest & highest-leverage first)

Each step is independently shippable and independently measurable on Bench2Drive (or open-loop L2
first for speed). Estimated savings are order-of-magnitude, to be confirmed.

### Step 0 — Instrument & baseline (prerequisite)
- Log the actual token counts per block and the wall-clock/GPU-mem breakdown for one training
  step. Confirm the ~3,000 / 1,305 estimates. Without this you can't attribute wins.
- Reduce the problem to a **small open-loop dev loop** (a few Bench2Drive routes, open-loop L2 +
  world-feature MSE) so each idea is testable in hours, not days.

### Step 1 — Freeze/LoRA the backbone, train a small head (Axis B, biggest cost cut)
*From AutoMoT + OneDrive ([frozen-backbone-kv-sharing.md](literature/frozen-backbone-kv-sharing.md),
[unified-decoder-transfer.md](literature/unified-decoder-transfer.md)).*
- Freeze Qwen2.5-VL transformer blocks; keep trainable only: `vis_head`, the waypoint head, and a
  LoRA/adapter (and optionally task-FFNs per OneDrive — reusing pretrained attention is the prior
  that matters; full FFN reuse isn't even optimal).
- Move the DINOv3 world-feature objective so it is driven by the trainable head, not by back-prop
  through the whole frozen stack.
- **Expected**: large drop in optimizer state + activation memory → fits a handful of GPUs.
  AutoMoT evidence: frozen backbone barely hurts planning (+1.24% L2) and *preserves* the general
  reasoning DeepSight's adaptive-CoT needs.
- **Risk/verify**: DeepSight's world-model head currently lives inside the Qwen forward; confirm it
  still trains well when blocks are frozen. If quality drops, widen to LoRA-on-attention only.

### Step 2 — Compress the 4 history frames into a memory bank (Axis A, ~1,200 → ~16 tokens)
*From ORION QT-Former ([qformer-resampler-compression.md](literature/qformer-resampler-compression.md)).*
- Replace the 4 full history-frame image-token blocks with **~16 recurrent history queries** that
  attend a FIFO bank of past states, then attend the current scene.
- **Use $N_h{=}16$** as the start — ORION shows 16 is optimal and **32 degrades**.
- **Expected**: ~1,200 input tokens → ~16. Cheapest large input-side cut; history is the most
  redundant part of the sequence.

### Step 3 — Resample the 6 surround views into a fixed query set (Axis A, ~1,800 → ~256–512)
*From OmniDrive / ORION (Lever 1) + UniDriveVLA sparse queries (Lever 2,
[sparse-perception-queries.md](literature/sparse-perception-queries.md)).*
- Insert a Q-Former/Perceiver resampler (or sparse 3D task queries) between the ViT and the LLM for
  the surround views; output a fixed `N_query` (try 256, sweep 128–512).
- Attach an **auxiliary perception objective** (detection/occupancy) to keep the compressed queries
  grounded — ORION/Percept-WAM show pure planning loss under-trains them.
- If keeping the single shared sequence, add **causal masking** so CoT/text tokens don't attend the
  compressed perception tokens — UniDriveVLA's anti-collapse fix without full MoT params.
- **Expected**: ~1,800 surround tokens → ~256. Combined with Step 2, input vision ~3,000 → ~300.
- **Risk/verify**: this breaks Qwen's native image-token contract (`masked_scatter`); needs a
  resampler warm-up stage. Camera-only — prefer "sparse queries from 2D features," not a metric
  BEV bottleneck (view-lifting is hard, Percept-WAM 25 vs 59 mAP).

### Step 4 — Shrink the 1,305-token world-query block (Axis A, potentially >100×)
*From OneVL latent tokens + PWM compact frames
([latent-reasoning-tokens.md](literature/latent-reasoning-tokens.md),
[compact-visual-encoders.md](literature/compact-visual-encoders.md)).*
- First **verify whether closed-loop eval consumes the dense predicted future-BEV at all**, or only
  the waypoints. If only waypoints, the dense 1,305-token future prediction is *training
  scaffolding* and can be (a) shrunk toward a PWM-style compact code (28 vs 261 tokens/frame) and/or
  (b) supervised via a **train-only decoder** (OneVL) and reduced to a few latent tokens at
  inference.
- **Adopt OneVL's staged curriculum** if you do this — direct joint training of latent tokens
  collapses (67 vs 89 PDMS).
- **Expected**: the single largest remaining block; high upside but highest uncertainty — gate it
  on the "does inference need dense BEV?" check.

### Step 5 — Replace AR text waypoints with a parallel head (output-side latency)
*From Reasoning-VLA / LinkVLA C2F / SpanVLA
([learnable-action-queries.md](literature/learnable-action-queries.md),
[efficient-action-decoding.md](literature/efficient-action-decoding.md)).*
- Swap numeric-text waypoints (`lm_head`, AR) for either learnable action queries (1 pass,
  Gaussian-init from Bench2Drive GT) or LinkVLA-style 2-pass C2F with a log-coordinate grid.
- **Expected**: closed-loop decode latency independent of waypoint count; ~60× faster decode in the
  reference papers. Small trainable module — composes with Step 1's frozen backbone.
- **Note**: LinkVLA shows *fewer* action tokens can improve accuracy (k=5 > k=10). Compress, don't
  just save.

### Step 6 (optional, deployment) — Fast–slow gating for the CoT/long-tail path
*From DualDriveVLA ([fast-slow-deployment.md](literature/fast-slow-deployment.md)).*
- Run the cheap compressed path by default; invoke full surround + long CoT only on low-confidence
  steps (gate on world-model error or a small trajectory scorer). Keeps long-tail capability while
  the *average* cost stays low. Lower priority than Steps 1–3.

## 4. Suggested "lightweight DeepSight" target config

Stack Steps 1–3 (+5) for the fast-iteration model:
- Frozen Qwen2.5-VL-3B + LoRA + small trainable heads (Step 1).
- Input vision: 16 history queries + ~256 surround queries ≈ **~300 tokens** (Steps 2–3) vs ~3,000.
- World queries: keep dense initially (1,305), revisit in Step 4 once verified.
- Parallel waypoint head (Step 5).
- Net per-step sequence ≈ **~1,700 tokens** (down from ~4,300), with a **fraction** of trainable
  params — the combination of "shorter sequence" + "small trainable surface" is what makes the
  GPU count drop enough to iterate quickly.

## 5. What to measure each iteration
- Open-loop L2 + world-feature MSE (fast proxy), then Bench2Drive DS/SR on a route subset.
- Tokens/step, training step time, peak GPU mem, min #GPUs to fit.
- **Guardrails from the survey**: watch for (a) reasoning/CoT degradation when compressing into the
  shared stream (UniDriveVLA collapse), (b) the history-query and action-vocab sweet spots
  (ORION $N_h$, LinkVLA k), (c) latent-token training instability without staging (OneVL).

## 6. Open questions to resolve before committing
1. Does Bench2Drive closed-loop inference consume the dense predicted future-BEV, or only the
   waypoints? (Gates Step 4's upside.)
2. Can the DINOv3 world-feature head train well with the backbone frozen, or is LoRA-on-attention
   the floor? (Gates Step 1's aggressiveness.)
3. Is the surround-view information mostly redundant per step (favoring heavy resampling) or
   safety-critical at full resolution for specific Bench2Drive abilities (merging, give-way)?
   ORION/LinkVLA multi-ability breakdowns are the reference for which scenarios need surround.
