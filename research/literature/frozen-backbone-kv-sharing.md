# Frozen Backbone + KV-Cache Sharing / Async Inference: AutoMoT

**Lever 5.** This is the single most relevant paper to the user's actual goal — *"make the
setup lightweight enough to test ideas quickly"* — because it attacks **training cost**, not just
token count. Full note: `wiki/sources/automot.md` (ICML).

## The thesis

Fine-tuning a large VLM on driving data is both expensive **and harmful**: AutoMoT shows it
causes **catastrophic forgetting** of general reasoning while barely helping planning. So freeze
the big model entirely and train only a small action expert beside it.

| Benchmark | Frozen UE | AD fine-tuned UE | Δ |
|---|---|---|---|
| OmniDrive (counterfactual planning) | 18.2 | 67.8 | +49.6 (FT helps *action* reasoning) |
| TallyQA (general reasoning) | 81.4 | 52.4 | **−35%** (FT destroys it) |
| InfographicVQA | 89.3 | 50.2 | **−44%** |

## The architecture

- **Understanding Expert (UE)**: Qwen3-VL-4B, **fully frozen, never trained on AD data.**
- **Action Expert (AE)**: ~1.6B, **trained from scratch.** Takes current RGB + (LiDAR) BEV +
  action queries.
- **Layer-wise shared KV cache**: the frozen UE produces per-layer K/V once; the AE concatenates
  its own K/V with the cached UE K/V and does joint attention. Cross-task attention is causal
  (planning conditions on understanding, not vice-versa).
- **Asynchronous inference**: the UE runs at *low* frequency; the AE reuses the **stale** cached
  UE KV at every high-frequency control step.

### The efficiency payoff (Table 5)
| Setting | L2 avg | Latency |
|---|---|---|
| Synchronized (UE every step) | 0.322 | 0.38 s |
| **Async + KV cache** | 0.326 | **0.05 s** |

**+1.24% L2 for a 7.6× speedup.** Decision accuracy essentially unchanged (53.49 → 53.10).
Trained for staleness tolerance via temporally-asynchronous samples (RGB+BEV pair selected
0.5–1 s ahead of the history frames).

Result: 87.34 DS / 70.0 SR Bench2Drive, lowest nuScenes collision (0.07) *without* fine-tuning
the UE — using only 8× A100 for the AE.

## Why this is the highest-leverage idea for "test ideas quickly"

DeepSight's pain is *"64× H20, full fine-tune, 2 epochs."* AutoMoT's recipe directly removes the
expensive part:
- **You never back-prop through the 3B Qwen.** Only a small expert (the part hosting your new
  ideas — world-feature head, trajectory head) trains. Massive memory + compute reduction;
  fits far fewer GPUs.
- The frozen VLM's general reasoning (which DeepSight's adaptive-CoT relies on for long-tail) is
  *preserved by construction*, not fought for.
- Async KV cache decouples the slow scene reasoning from the fast control loop — relevant to
  DeepSight's closed-loop Bench2Drive latency.

### How it composes with the token-compression levers
AutoMoT is **orthogonal** to Levers 1–3: you can freeze the UE *and* feed it resampled/sparse
tokens. Frozen-backbone + resampler is the lightest possible config — small trainable surface,
short sequence.

### Caveats / what to verify against DeepSight
- AutoMoT's AE uses LiDAR BEV; DeepSight is camera-only. The KV-sharing mechanism is
  modality-agnostic, but you'd condition the AE on camera/DINOv3 features instead.
- DeepSight's world-model objective (DINOv3 alignment) lives *inside* the Qwen forward today
  (`vis_head` on `<|bev_token_i|>` positions). To freeze Qwen you must move the world-feature
  prediction into the trainable expert, or train just `vis_head` + a small adapter while keeping
  the transformer blocks frozen (a lighter middle-ground: LoRA/adapter rather than full freeze).
- Staleness is unsafe in fast events (emergency brake, cut-in) — AutoMoT flags this. Keep the
  control-relevant tokens (ego state, front view) on the fast path.

## Cheaper cousins worth noting
- **OneDrive** (next note): you may not need to randomize the whole FFN — but you *should* avoid
  trusting that full fine-tuning of attention+FFN is optimal.
- **SpanVLA / Alpamayo**: same "VLM frozen / stop-gradient, train a small action expert over
  sparse KV-cache" pattern, with a flow-matching head — see
  [efficient-action-decoding.md](efficient-action-decoding.md).
