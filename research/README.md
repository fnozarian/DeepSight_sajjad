# DeepSight Research: Lightweight Token-Efficient Driving VLA

Motivation: DeepSight's current training (64× H20, full fine-tune of Qwen2.5-VL-3B, ~4,300+ tokens
per step) is too heavy to iterate on ideas quickly. This folder surveys how recent SoTA driving
VLAs **compress the input token budget** (multi-frame, multi-camera) and **cut training/inference
cost** without losing much performance, then turns that into a concrete plan for DeepSight.


## Start here
1. [literature/00-taxonomy.md](literature/00-taxonomy.md) — the 7 compression levers, a comparison
   table, headline efficiency numbers, and the "don't over-compress" caveats. **Read first.**
2. [plan.md](plan.md) — DeepSight's actual cost breakdown and a prioritized, shippable experiment
   ladder (Steps 0–6) with expected savings and risks.

## Literature notes (each = one lever / cluster, written through the compression lens)
| Note | Lever | One-line |
|---|---|---|
| [qformer-resampler-compression.md](literature/qformer-resampler-compression.md) | 1 + 3 | OmniDrive Q-Former / ORION QT-Former: multi-view + history → fixed small query set |
| [sparse-perception-queries.md](literature/sparse-perception-queries.md) | 2 | UniDriveVLA / Percept-WAM: sparse task queries; the feature-collapse diagnosis |
| [learnable-action-queries.md](literature/learnable-action-queries.md) | 4 | Reasoning-VLA: 1-pass query waypoint head (~61× faster decode) |
| [efficient-action-decoding.md](literature/efficient-action-decoding.md) | 4 | LinkVLA C2F (Bench2Drive SOTA) + SpanVLA sparse-KV flow matching |
| [frozen-backbone-kv-sharing.md](literature/frozen-backbone-kv-sharing.md) | 5 | AutoMoT: frozen VLM + async KV cache, 7.6× faster, +1.24% L2 only |
| [latent-reasoning-tokens.md](literature/latent-reasoning-tokens.md) | 6 | OneVL: latent CoT/world tokens, train-only decoders dropped at inference |
| [compact-visual-encoders.md](literature/compact-visual-encoders.md) | 7 | Drive-JEPA / Latent-WAM / PWM: encoder-side cost, compact future frames |
| [unified-decoder-transfer.md](literature/unified-decoder-transfer.md) | enabler | OneDrive: attention transfers, text-FFNs don't → train less |
| [fast-slow-deployment.md](literature/fast-slow-deployment.md) | enabler | DualDriveVLA: cheap path default, invoke VLM on 15% → 3.2× throughput |

## TL;DR recommendation
The two highest-leverage, composable moves for "iterate fast":
- **Freeze/LoRA the backbone, train small heads** (AutoMoT/OneDrive) — drops the GPU count.
- **Resample input vision** (history → 16-query memory bank; surround → ~256 queries) — cuts the
  ~3,000-token input prefix ~10×.

Together: ~4,300 → ~1,700 tokens/step at a fraction of the trainable params. Detailed in
[plan.md](plan.md) §3–4.

> Status: literature review + plan. No DeepSight code changed yet. Numbers for DeepSight's current
> token counts are estimates from the resize config and SRC_CODE_MAP — Step 0 of the plan is to
> confirm them by instrumenting a real training step.
