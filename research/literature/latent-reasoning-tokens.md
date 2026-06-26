# Latent Reasoning Tokens (Train-Time Decoders, Dropped at Inference): OneVL

**Lever 6.** DeepSight already does a version of this on the *world-model* side (1,305 prefilled
`<|bev_token_i|>` whose hidden states are projected by `vis_head` and supervised against DINOv3).
OneVL generalizes the pattern to **reasoning + future-frame** supervision and shows how to make
it stable and cheap at inference. Full note: `wiki/sources/onevl.md` (Xiaomi).

## The idea

Explicit autoregressive CoT is expensive at inference (you decode the whole think-block). OneVL
replaces it with a handful of **latent tokens** and supervises them with **two auxiliary decoders
that exist only during training**:
- **Language aux decoder** → reconstructs human-readable CoT from language latent tokens.
- **Visual aux decoder** → predicts future-frame visual tokens at +0.5 s / +1.0 s (a world-model
  objective).

At inference both decoders are **discarded**; the latent tokens are **prefilled** into the prompt
in one parallel pass, so you keep latent-reasoning benefits at **answer-only latency**.

Token budget: **4 visual latent + 2 language latent tokens** (impl. as 35 + 20 vocab tokens) vs a
full CoT block. NAVSIM 88.84 PDMS @ 4.46 s — beats explicit AR CoT (88.29 @ 6.58 s) at lower
latency.

### Ablations that matter
- Visual aux decoder: +0.87 PDMS over language-only latent supervision.
- Language aux decoder: +0.31.
- **Staged curriculum is load-bearing**: direct joint training collapses to **67.13** vs 88.84.
  (Preliminary: train visual decoder → Stage 0 warm latents on trajectory → Stage 1 freeze VLM,
  align decoders → Stage 2 joint.)
- Deployment variant: MLP regression head instead of AR waypoints → **0.24 s** at 86.83 PDMS
  (−2 PDMS for ~18× faster) — same lesson as Lever 4.

## Relationship to DeepSight (close)

DeepSight's world-query block is *already* a latent-token-with-auxiliary-decoder design:
`<|bev_token_i|>` hidden states → `vis_head` → MSE vs DINOv3 future-BEV features. Differences and
opportunities:
- **DeepSight keeps the world tokens at inference** (1,305 of them) because they are the world-
  model output, not just training scaffolding. OneVL's lesson: if the world tokens exist mainly to
  *shape* the trajectory, you could shrink them drastically (OneVL uses **4** visual latents, not
  1,305) and keep a heavier decoder **only at training time**. That is a potential **>100×** cut
  of the world-query block — *if* the closed-loop task does not need the full dense future-BEV
  prediction at inference. Verify whether Bench2Drive eval actually consumes the predicted BEV or
  only the waypoints.
- DeepSight's CoT is still explicit text. OneVL shows you can compress it to ~2 latent tokens +
  a train-only language decoder while *keeping* post-hoc interpretability. Relevant to DeepSight's
  "adaptive CoT" cost.
- **Adopt the staged curriculum** if you compress either block — OneVL's collapse-without-stages
  result is a strong warning that latent-token training is unstable end-to-end.

### Related (same family, in library)
- **DynVLA**: compact ego/environment *dynamics tokens* before action tokens (inference-time),
  `wiki/sources/dynvla.md`.
- **FutureSightDrive / FLARE / DriveVLA-W0**: future-feature/visual supervision as a dense
  training signal — same "predict the future to ground planning" principle DeepSight uses with
  DINOv3. See `wiki/concepts/world-model-for-ad.md`.
