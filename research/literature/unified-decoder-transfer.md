# What Actually Transfers When You Adapt a VLM: OneDrive

**Cross-cutting enabler.** Not a token cut, but it tells you *which parameters are worth training*
— directly relevant to making the fine-tune cheaper. Full note: `wiki/sources/onedrive.md`.

## The diagnostic result (Table 1)

OneDrive ran a clean ablation: take a pretrained VLM, build a parallel driving decoder, and toggle
whether **attention** vs **FFN** weights are initialized from the pretrained model.

| Backbone | Attention init | FFN init | NDS |
|---|---|---|---|
| Qwen2.5-VL-3B | ✓ | ✓ | 27.14 |
| Qwen2.5-VL-3B | ✓ | ✗ | **31.37** |
| Qwen2.5-VL-3B | ✗ | ✓ | 27.95 |
| Qwen2.5-VL-3B | ✗ | ✗ | 30.15 |
| InternVL3-1B | ✓ | ✗ | **32.05** |

**The transferable prior in a VLM is causal attention, not the text-specialized FFN.** For
Qwen2.5-VL-3B (DeepSight's exact backbone), keeping attention but **randomizing the FFN** beats
keeping both (+4.2 NDS). Keeping the pretrained FFN can be *actively harmful*.

## OneDrive's architecture (single causal decoder, no Q-Former, no MoT)

One pretrained VLM decoder hosts everything in one sequence:
`Z = [X_img, Q_det, Q_lane, Q_plan, X_text]`. Heterogeneity is handled by **shallow-layer
adaptations only**: query-only self-attention among perception queries + **task-specific FFNs**
for det/lane/plan; deeper layers stay close to the pretrained LM for text. 3D positional
embeddings added to image/query tokens.

- NAVSIM latency **156 ms** vs ReCogDrive 263 ms (−40%); nuScenes 513 ms vs OmniDrive 3,727 ms.
- Token-order matters: det → lane → plan beats lane → det (Table 9).
- SFT-only, 86.8 PDMS — architectural-unification paper, not a leaderboard winner.

## How this maps onto DeepSight

DeepSight full-fine-tunes Qwen2.5-VL-3B on 64× H20. OneDrive implies two cheaper paths:
1. **Train task FFNs (or LoRA-FFN) but reuse pretrained attention** — much smaller trainable
   surface than full fine-tune, and OneDrive says full FFN reuse isn't even optimal. This is a
   concrete, low-risk way to cut training cost while keeping the attention prior that carries the
   visual/spatial reasoning.
2. **Put DeepSight's heterogeneous tokens (vision, world-query, waypoint, CoT) in the one shared
   decoder it already uses, but give the structured tokens their own shallow-layer FFNs** so they
   don't fight the language FFNs — a lighter alternative to full MoT (UniDriveVLA) when you want
   to keep one backbone.

**Caveat**: OneDrive is SFT-only and not tested under RL; and the InternVL3-ViT detection backbone
underperformed detection-specialized ViTs ("VLM token resolution is lower"). If you compress
DeepSight's vision tokens hard, remember the ViT resolution/token budget caps perception accuracy.
