# Sparse Task Queries Instead of Dense Tokens: UniDriveVLA & Percept-WAM

**Lever 2.** Where the Q-Former (Lever 1) compresses *all* visual content into generic scene
queries, this line replaces dense spatial token grids with **sparse, task-specific queries** —
and, crucially, diagnoses *why* naive dense injection breaks VLMs. Full notes:
`wiki/sources/unidrivevla.md`, `wiki/concepts/perception-for-planning.md`.

## The diagnosis you should read first (UniDriveVLA)

UniDriveVLA gives the strongest empirical reason **not** to just stuff more spatial tokens into a
shared-weight decoder. Measuring cosine similarity between LLM (text) tokens and injected
perception tokens across layers, in a **shared-weight** decoder the similarity climbs toward
**1.0** — *feature collapse*: spatial and semantic representations become indistinguishable, so
improving perception directly degrades reasoning and vice-versa.

| Architecture | General VQA↑ | DriveBench↑ | L2↓ | CR↓ |
|---|---|---|---|---|
| Shared-weight decoder | 31.1 | 50.8 | 0.641 | 0.175 |
| Mixture-of-Transformers (MoT) | **45.5** | **54.9** | **0.533** | **0.140** |

**Implication for DeepSight**: DeepSight currently fuses vision + world-query + text into *one*
shared Qwen sequence ([SRC_CODE_MAP.md](../../SRC_CODE_MAP.md) §2.5). If you compress aggressively
*and* add perception supervision into that shared stream, watch for reasoning/CoT degradation.
The fix the field converged on is **parameter or attention-mask separation**, not just fewer
tokens.

## UniDriveVLA's sparse-query design

- Backbone: Qwen3-VL (SigLIP-2 + Qwen3 LM), 6-view 960×544.
- Three **Mixture-of-Transformers** experts (separate FFN/norm/proj per expert): Understanding
  (text), Perception (sparse queries), Action (flow-matching trajectory).
- **Masked Joint Attention** routes information asymmetrically:
  - `und` → causal self-attention only (never sees per/act → cannot collapse; preserves VLM
    pretraining).
  - `per` → attends `und` + self (one-way semantic enrichment).
  - `act` → attends both.
- **Sparse perception queries** initialized from **K-Means instance banks** (dataset-level
  clustering) handle 5 tasks (3D det, HD map, ego, motion, occupancy) in *one* sparse decoder —
  **no dense BEV grid, no PointPillars, no explicit view-lifting**. Geometry is pulled from
  multi-scale 2D features via deformable attention. Tens of queries replace hundreds of dense
  BEV tokens.
- Two-pass: sparse decode → project into VLM space → masked joint attention → project back →
  refine.

Result: 78.37 DS Bench2Drive (best without PDM-Lite at publication), 0.51 m L2 nuScenes no-ego.
Cost: still 2B/8B backbone; general-VQA drops from 63→43 MMStar even *with* MoT (compression
reduces but does not eliminate adaptation damage).

## Percept-WAM's World-PV / World-BEV tokens (alternative framing)

Instead of sparse queries, Percept-WAM keeps **token reuse**: the World-PV (perspective) and
World-BEV tokens produced during the perception prefill are **directly reused** by the trajectory
decoder — no second forward pass, perception and planning share the prefill compute. Grid-
conditioned parallel AR decoding (mutually-masked grid tokens) gives a **16× detection speedup**
over sequential AR with no accuracy loss. Ablation (Table 5 in perception-for-planning concept):
**detection** query is most safety-critical (CR 0.21→0.10); **occupancy** best for L2; map/motion
add little.

## How this maps onto DeepSight

DeepSight's 6 surround views are the obvious target. Two options:
1. **Replace dense surround tokens with ~a few dozen sparse 3D queries** (UniDriveVLA-style),
   carrying an auxiliary detection/occupancy objective so they stay grounded. Biggest token win
   *and* adds the perception supervision Percept-WAM/ORION show is load-bearing.
2. If you keep DeepSight's single shared sequence, add **causal masking** so text/CoT tokens do
   not attend to the compressed perception tokens (cheap, no MoT params) — UniDriveVLA shows the
   asymmetric mask alone recovers most of the reasoning-preservation benefit.

**Caveat**: camera-only BEV view-lifting is hard (Percept-WAM: 25.0 vs 58.9 mAP without LiDAR).
DeepSight is camera-only, so prefer the UniDriveVLA "sparse queries from 2D features, no explicit
BEV grid" route over building a metric BEV bottleneck.
