# Efficient Action Decoding: LinkVLA (C2F) & SpanVLA (sparse-KV flow matching)

**Lever 4, output side — the two most relevant for DeepSight's benchmark.** LinkVLA is the
current **Bench2Drive SOTA** (91.01 DS), so it's the most direct comparison point. Full notes:
`wiki/sources/linkvla.md`, `wiki/sources/spanvla.md`.

## LinkVLA — coarse-to-fine, 361 ms → 48 ms (Bench2Drive 91.01 DS)

Backbone is tiny: **InternVL2-1B** (InternViT-300M + Qwen2-0.5B) + LoRA. Three ideas:

1. **Shared discrete codebook**: waypoints quantized into a BEV grid and merged with the text
   vocabulary — one codebook, one VLM, no modality gap.
2. **Log-coordinate transform** `z' = sign(z)·log(1+k|z|)` concentrates grid resolution near the
   ego and compresses far-field → fewer, better-allocated action tokens (56×101 = 5,656 grid).
   - **Table S1 — the key compression lesson**: k=5 (5,656 tokens) → **91.01 DS**; k=10 (7,245
     tokens) → 89.85. *More tokens / finer grid HURTS* — a larger discrete vocab confuses the LM.
     Compression is not just a cost saving; it can improve accuracy.
3. **Coarse-to-fine (C2F) decoding**: AR over T waypoints needs T forward passes. C2F does **two**:
   pass 1 predicts the endpoint, linearly interpolate a coarse path, pass 2 refines all waypoints
   **in parallel**. **361 ms → 48 ms (−86%)**, and C2F (91.01) even beats full AR (90.66).

Other notes: bidirectional language↔action objective (predict L from A and A from L) enriches
shared embeddings; spatial soft-labels (Gaussian over neighbor grid cells, σ=1.2) add +1.8% SR.

## SpanVLA — sparse-KV flow-matching expert (NAVSIM 90.3)

Backbone Qwen2.5-VL-3B (**DeepSight's backbone**), default 3 views × 4 history frames @ 2 Hz.
- VLM does reasoning + (training) discrete action tokens; at inference it stops at a special token
  and **hands off to a flow-matching action expert.**
- The expert reads **sparse VLM KV-cache** (every other layer, "interval 2") + a **historical-
  trajectory initialization** (flow from $a_{his}$, not from Gaussian noise) + time embedding.
  VLM is **stop-gradiented** while training the expert (cheap, preserves the backbone).
- **Table 5 ablation** (sparse caching): Full caching 88.1 PDMS @ 0.18 s; **Interval-2 90.3 @
  0.08 s**; last-layer-only 79.3. Sparse beats full *and* is faster. Historical init = +3.9 PDMS.
- **Table 4 runtime**: trajectory generation is **flat in waypoint count** (0.08 s for 10 *or* 50
  waypoints) vs AutoVLA AR which grows (0.40 s → 1.72 s). Total −46% (10 pts) to −74% (50 pts).

## How these map onto DeepSight

DeepSight emits waypoints as **numeric text via `lm_head`** (AR). Both papers replace that:
- **LinkVLA C2F** is the closest fit because DeepSight *also* targets Bench2Drive and *also* keeps
  CoT text — adopt: (a) log-grid action tokenization (allocate resolution near ego), (b) 2-pass
  C2F instead of full-AR waypoints, (c) cap the action vocabulary (their result: fewer is better).
- **SpanVLA's sparse-KV + flow-matching expert** is the lighter-to-train option that pairs with a
  **frozen/stop-grad backbone** (Lever 5): you train only the small expert over interval-2 KV.
  "Interval-2 sparse KV" is itself a compression knob — you don't need every layer's KV.
- Both give **waypoint-count-independent** decode latency — valuable for closed-loop control.

**Caveat**: LinkVLA's 48 ms **excludes** CoT generation (variable); SpanVLA still runs at ~1.5 Hz
unoptimized (33 ms/token). The action head is fast; the reasoning text is what you must also
compress (→ [latent-reasoning-tokens.md](latent-reasoning-tokens.md)).
