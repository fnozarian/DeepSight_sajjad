# Taxonomy: Vision-Token Compression & Efficiency for Driving VLAs

> Scope: techniques from recent SoTA end-to-end driving VLAs that control or compress the
> **input token budget** (multi-frame, multi-camera) or otherwise make training/inference
> lightweight, **without** sacrificing much planning performance. Source library:
> `/home/farzad/llm_wiki_knowledgebase` (52 papers). Each technique links to a detailed note.

## Why this matters for DeepSight

DeepSight feeds the LLM a long token sequence per step:

| Token stream | Count (current setup) | Source |
|---|---|---|
| Input vision (4 history front + 6 surround) | **~3,000** (≈299 tok/frame @ 364×644, Qwen2.5-VL patch14/merge2) | ViT → `masked_scatter` |
| World queries `<|bev_token_i|>` (5 future BEV frames) | **1,305** (5 × (256 + 1 CLS + 4 reg)) | prefilled learnable embeddings |
| CoT + waypoints (text) | variable | `lm_head` |

So **~4,300+ tokens before text**, dominated by (a) raw input vision tokens and (b) the
1,305 world-query block. Attention is O(L²); halving L is roughly a 4× attention FLOP cut and
a large KV-cache/memory win. This is exactly the axis the OmniDrive line of work attacks.
The techniques below are the levers, ordered by how directly they cut DeepSight's token count.

## The seven compression levers

| # | Lever | What it compresses | Representative work | Token effect | DeepSight fit |
|---|---|---|---|---|---|
| 1 | **Query-based visual resampler (Q-Former / Perceiver)** | Many ViT patch tokens → fixed small query set | OmniDrive, ORION QT-Former | 10 frames × 299 → e.g. 512 scene + 16 history | **High** — direct input-side cut; see [qformer-resampler-compression.md](qformer-resampler-compression.md) |
| 2 | **Sparse task queries instead of dense tokens** | Dense BEV/patch grid → sparse per-task queries | UniDriveVLA, Percept-WAM | Hundreds → tens of queries | **High** — replaces dense surround tokens; [sparse-perception-queries.md](sparse-perception-queries.md) |
| 3 | **Temporal memory bank** | Long history → a few recurrent history queries | ORION ($N_h{=}16$), HERMES state encoder | 4 history frames → 16 carried queries | **High** — kills the 4 history-frame cost; [qformer-resampler-compression.md](qformer-resampler-compression.md) |
| 4 | **Learnable action queries / parallel decode** | AR waypoint tokens → 1-pass query head | Reasoning-VLA, Percept-WAM, LinkVLA C2F, SpanVLA FM | N×T steps → 1–2 passes | **Med** — output side; [learnable-action-queries.md](learnable-action-queries.md), [efficient-action-decoding.md](efficient-action-decoding.md) |
| 5 | **Frozen backbone + KV-cache sharing / async** | Avoid fine-tuning the big VLM; reuse stale KV | AutoMoT (7.6× faster, frozen UE) | training & latency, not token count | **High for cost** — lets you train a small expert; [frozen-backbone-kv-sharing.md](frozen-backbone-kv-sharing.md) |
| 6 | **Latent reasoning tokens (train-time only)** | Long CoT → few latent tokens, decoders dropped at inference | OneVL, DynVLA | CoT length → ~6 latent tokens | **Med** — DeepSight already prefixes world tokens; [latent-reasoning-tokens.md](latent-reasoning-tokens.md) |
| 7 | **Compact / better-pretrained visual encoder** | Replace heavy ViT path; pretrain for planning | Drive-JEPA, Latent-WAM (DINOv2 distill), PWM 28-tok frames | tokens/frame + backbone size | **Med** — DeepSight already uses DINOv3 as target; [compact-visual-encoders.md](compact-visual-encoders.md) |

Cross-cutting enablers (not token cuts but cheap-experiment enablers):
- **Which params to train** — OneDrive: attention transfers, text-FFNs do not → train less. [unified-decoder-transfer.md](unified-decoder-transfer.md)
- **Fast–slow deployment** — run a cheap path by default, invoke the VLM only on hard cases. [fast-slow-deployment.md](fast-slow-deployment.md)

## Headline efficiency numbers worth remembering

| Method | Claim | Mechanism |
|---|---|---|
| AutoMoT | **7.6× lower latency**, +1.24% L2 only | frozen UE + async KV cache |
| Reasoning-VLA | **~61× faster** action decode (5.4s → 0.08s) | learnable queries, 1 parallel pass |
| LinkVLA C2F | **361ms → 48ms** (86% ↓) | endpoint + parallel refine (2 passes) |
| SpanVLA | trajectory time **flat in waypoint count**; −46–74% total | sparse-KV flow-matching expert |
| OneVL | latent CoT at **answer-only latency**, 4B | latent tokens + train-only decoders |
| HybridDriveVLA / DualDriveVLA | **3.2× throughput** at 91.0 PDMS | ViT default, VLM on 15% of cases |
| ORION QT-Former | multi-view → 512 scene + 16 history queries | Q-Former resampler |

## Important caveats (don't over-compress)

- **LinkVLA Table S1**: *more* action tokens (finer grid) **hurt** (91.0 → 89.9 DS). Bigger vocab confuses the LM. Compression can help, not just save cost.
- **ORION Table 5**: history queries have a sweet spot — $N_h{=}16$ best, $N_h{=}32$ *degrades* (too many history queries drown the current frame).
- **UniDriveVLA**: naively injecting dense perception/3D tokens into a *shared-weight* decoder causes **feature collapse** (cosine→1) and kills reasoning. Compression must respect the perception–reasoning separation (MoT or causal masking).
- **Camera-only BEV view-lifting is hard**: Percept-WAM camera-only BEV = 25.0 vs 58.9 mAP with LiDAR init. DeepSight is camera-only — relevant if you push compression into a BEV bottleneck.
