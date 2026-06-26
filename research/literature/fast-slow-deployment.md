# Fast–Slow Deployment: HybridDriveVLA / DualDriveVLA

**Cross-cutting enabler.** A way to get VLM-level quality at a fraction of the *average* compute
by only invoking the expensive path when needed. Full note: `wiki/sources/hybriddriveVLA.md`
(ICML).

## The finding

Plug both a full VLM (InternVL-2B) and a vision-only backbone (ViT/ResNet) into the *same*
RecogDrive diffusion planner. At the **backbone** level their features are very different (CKA
~0.22), but the **policy/planner compresses them into a shared decision space** (CKA ~0.54). The
two policies are **complementary but long-tailed**: each decisively wins on ~2–3% of scenarios;
neither contains the other. Cross-model oracle best-of-2 = 93.58 PDMS vs single VLM 90.80 — the
diversity is real and exploitable.

## The two systems
- **HybridDriveVLA**: run both, build an 11-candidate set by interpolating along the VLM↔ViT
  style axis, score with a learned trajectory scorer → **92.10 PDMS** (NAVSIM-v1). Doubles cost.
- **DualDriveVLA (the relevant one for efficiency)**: run the **cheap ViT by default**; score its
  trajectory; only if confidence < γ, invoke the VLM + full candidate selection.
  - 100% ViT: 88.88 PDMS
  - **15% VLM invocations: 91.00 PDMS at 3.2× throughput**
  - 100% Hybrid: 92.10 PDMS

## How this maps onto DeepSight

DeepSight's adaptive-CoT is *already* a fast–slow idea ("inject external/social knowledge for
long-tail scenarios"). DualDriveVLA generalizes it to the whole model:
- Run a **cheap path** (compressed tokens, no/short CoT, regression waypoint head) on the easy
  majority; invoke the **full path** (long CoT, full surround, dense world tokens) only on
  low-confidence/long-tail steps.
- This makes the *average* closed-loop cost low even if the heavy path stays expensive — useful
  for Bench2Drive where most frames are easy.
- Needs a **confidence/scorer signal**. DeepSight could reuse the world-model prediction error or
  a small trajectory scorer (DrivoR-style, as in HybridDriveVLA) as the gate.

**Caveats**: HybridDriveVLA's interpolation occasionally violates drivable area (DAC drop on
NAVSIM-v2); the global confidence threshold is not scenario-adaptive. For a *research-iteration*
setup this is a deployment-time optimization — lower priority than Levers 1/5, but it's how you'd
reconcile "lightweight" with "keep the long-tail CoT capability."
