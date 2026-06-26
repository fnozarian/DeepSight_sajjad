# Learnable Action Queries & Parallel Decoding: Reasoning-VLA, Percept-WAM

**Lever 4 (output side).** Not input-token compression, but it removes the *other* sequential
bottleneck — autoregressive waypoint decoding — and is cheap to add. Full notes:
`wiki/sources/reasoning-vla.md`, `wiki/concepts/perception-for-planning.md`.

## Reasoning-VLA (claims 91.7 PDMS NAVSIM; ~61× faster decode)

Backbone Qwen2.5-VL (3B/7B) — **the same family as DeepSight.** Trajectory is produced by a
**VL-to-Action module**:
1. The VLM hidden states are cached as KV.
2. A set of **learnable action queries** $AQ \in \mathbb{R}^{T \times N \times D}$ (T timesteps,
   N=2 coords, D=hidden) **cross-attend** to that KV, then self-attend among themselves.
3. **All T×N waypoints are produced in one parallel pass** (bidirectional mask, no causal
   AR), continuous regression (no coordinate tokenization).
4. A small Action Refinement Module (MLP+attn) smooths the output.

**Key trick — Gaussian initialization**: action queries are initialized by sampling from the
per-position **mean/variance of GT trajectories** across the training set. Ablation: learnable vs
non-learnable queries is the single biggest factor (0.30→0.23 L2); Gaussian init and ARM each add
~0.03.

### Efficiency (Table 9)
| Method | Trajectories | Steps | Time |
|---|---|---|---|
| Qwen2.5-VL AR | 10 | >20 | 5.47 s |
| **Reasoning-VLA** | 10 | **1** | **0.089 s** |

Extra trajectories are nearly free (6→10 traj: 0.081→0.089 s) — all in one pass.

## Percept-WAM four-query decoder (modality-partitioned)

A complementary design: instead of one query bank, **four** parallel MLP decoders with
attention masks — `Q_ego` (kinematics), `Q_pv` (semantic), `Q_bev` (3D geometry), `Q_full`
(final output). Each must independently produce a reasonable trajectory from its limited view,
preventing the model from ignoring any modality. Best L2 *and* best speed among Percept-WAM's
decoder variants.

## How this maps onto DeepSight

DeepSight currently emits waypoints as **text** through the shared `lm_head`
(`<answer> future waypoints: [(x,y),…]` — see [SRC_CODE_MAP.md](../../SRC_CODE_MAP.md) §2.5), which
is autoregressive and numeric-token-by-token. Swapping to a learnable-query regression head:
- Removes per-waypoint AR steps from the closed-loop control latency (directly helps Bench2Drive).
- Is a **small trainable module** — composes perfectly with a frozen/LoRA backbone (Lever 5),
  i.e. exactly the "lightweight" setup the user wants.
- Reuses the existing Qwen hidden states as KV (no new big component).
- Gaussian init from Bench2Drive GT waypoints is essentially free to compute and high-impact.

**Caveat**: one-shot queries have *no iterative refinement*, so in highly multimodal scenes they
can produce averaged trajectories. ORION's counter-argument is to keep a small generative
(VAE/diffusion) head; LinkVLA's is the 2-pass coarse-to-fine refine
([efficient-action-decoding.md](efficient-action-decoding.md)). For DeepSight's single-mode
waypoint output this is likely fine, but worth an ablation.
