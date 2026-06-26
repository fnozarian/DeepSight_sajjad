# Compact / Better-Pretrained Visual Encoders: Drive-JEPA, Latent-WAM, PWM, backbone diagnostics

**Lever 7.** Reduce cost at the encoder/backbone level rather than (or in addition to) the LLM
sequence. Most relevant because DeepSight already uses **DINOv3** as its world-feature target, so
this is adjacent territory. Sources: `wiki/concepts/foundation-backbones-for-ad.md`,
`wiki/sources/drive-jepa.md`, `wiki/sources/latent-wam.md`, `wiki/sources/policy-world-model.md`.

## The single most useful table (Drive-JEPA vision-pretraining ablation)

Same simple downstream planner, swap the frozen visual encoder:

| Encoder (ViT-L unless noted) | NAVSIM PDMS |
|---|---|
| ImageNet ResNet34 | 76.0 |
| DINOv2 ViT-L | 76.1 |
| SigLIP ViT-L | 83.4 |
| V-JEPA 2 ViT-L | 86.1 |
| **Drive-JEPA (driving-video-pretrained ViT-L)** | **89.0** |

**Temporal latent-prediction pretraining transfers to planning far better than static image-level
pretraining** when the decoder is intentionally simple. I.e. a *small* encoder with the right
pretraining can beat a bigger general one — directly relevant to "lightweight without losing
performance."

## Latent-WAM — compress a small encoder via geometric distillation

Deployed encoder is **DINOv2-Base**, distilled from a frozen **WorldMirror/VGGT geometry teacher**
at training time (teacher removed at inference). Scene features compressed into compact scene
tokens for latent world modeling.
- Distillation > concatenation: no geo-feature 88.3 EPDMS, frozen feature concat 88.0,
  **distillation 89.3**. Spatial features help only when *aligned into* the trainable planning
  representation, not appended as frozen KV.
- **Backbone-adaptation lesson**: full fine-tune DINOv2-Base = 89.3; **LoRA collapses to 68.5**.
  For geometric feature targets, low-rank adaptation is too restrictive — relevant if you try to
  LoRA-only DeepSight's DINOv3 path.

## Policy World Model (Show-o) — extreme frame tokenization

PWM encodes each **128×224 future frame as just 28 tokens** (8192-code VQ, trainable low-res
branch + frozen high-res context branch). This is the aggressive end of Lever 7: future-frame
prediction stays cheap enough to run *before* action prediction. Contrast with DeepSight's
**261 tokens per future BEV frame** (256 patches + CLS + 4 register) × 5 = 1,305. PWM suggests the
future-frame block could be ~10× smaller if you accept a coarser future representation.

## Backbone-role diagnostics (foundation-backbones concept)

- **Frozen-backbone designs can beat fine-tuned VLMs** when the action expert is well-coupled
  (AutoMoT) — reinforces Lever 5.
- **OneDrive**: attention transfers, text-FFNs don't (see
  [unified-decoder-transfer.md](unified-decoder-transfer.md)).
- **CLEAR**: a compact LLM (Qwen 0.8B) used only for *hidden-state routing/scoring* (not text
  generation) + frozen Drive-JEPA encoder → 93.7 PDMS. Hidden-state use is more deployment-
  friendly than text-format action generation.

## How this maps onto DeepSight

- DeepSight's **future-BEV target** is DINOv3 features; its **input** path is Qwen's own ViT. Two
  encoder-side levers: (1) shrink the **future-BEV token block** (1,305) toward a PWM-style compact
  code — biggest single block after input vision; (2) consider whether a **driving-video-pretrained
  encoder** (V-JEPA/Drive-JEPA style) on the *input* side would let you drop to fewer input frames
  at equal performance.
- If you try to cheapen DeepSight's DINOv3 alignment with LoRA, Latent-WAM's collapse result is a
  warning: geometric/feature-distillation targets may need full (small-)encoder training, not LoRA.
- DeepSight already follows the "predict future features to ground planning" thesis (its core
  contribution). The compression question is purely *how many tokens* that prediction needs — and
  PWM (28/frame) vs DeepSight (261/frame) shows there is a lot of headroom.
