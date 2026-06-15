#!/usr/bin/env python
"""Open-loop L2 evaluation for DeepSight offline inference.

Reads the JSONL produced by `src/infer_local.py` (one {prompt, gt, pred}
object per line), parses the future waypoints out of both the ground-truth and
the predicted answer with the repo's own `parse_answer`, and reports L2 error at
1 s and 2 s (the paper's open-loop metric). Optionally saves a per-sample
trajectory plot (GT vs prediction in ego frame).

Usage:
    python src/tools/eval_l2.py --infer debug/infer_results.json
    python src/tools/eval_l2.py --infer debug/infer_results.json --plot_dir debug/traj_plots
"""
import argparse
import json
import os

import numpy as np

# reuse the parsing / metric helpers from our local copy of eval_and_visual
# (the upstream src/tools/eval_and_visual.py is left untouched)
from eval_and_visual_local import parse_answer, cal_l2_loss, print_l2_loss


def _try_parse(ans):
    try:
        trajs, _pix = parse_answer(ans)
        return np.array(trajs, dtype=float)
    except Exception:  # noqa: BLE001
        return None


def maybe_plot(gt, pred, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    fig, ax = plt.subplots(figsize=(4, 5))
    # ego forward = x, lateral = y; show lateral on horizontal axis for a top-down feel
    ax.plot(gt[:, 1], gt[:, 0], "-o", color="g", label="GT")
    ax.plot(pred[:, 1], pred[:, 0], "-x", color="r", label="pred")
    ax.scatter([0], [0], c="k", marker="s", label="ego")
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", required=True, help="JSONL output of infer_local.py")
    ap.add_argument("--plot_dir", default="", help="If set, save per-sample GT-vs-pred trajectory PNGs here.")
    args = ap.parse_args()

    lines = [l for l in open(args.infer, encoding="utf-8") if l.strip()]
    print(f"loaded {len(lines)} inferred samples from {args.infer}")
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    from collections import defaultdict
    losses_1s, losses_2s = [], []
    per_scene = defaultdict(lambda: ([], []))  # scene -> (l1 list, l2 list)
    n_ok, n_bad = 0, 0
    for idx, line in enumerate(lines):
        rec = json.loads(line)
        gt = _try_parse(rec.get("gt", ""))
        pred = _try_parse(rec.get("pred", ""))
        if gt is None or pred is None or len(gt) < 4 or len(pred) < 4:
            n_bad += 1
            continue
        gt, pred = gt[:4], pred[:4]
        l1, l2 = cal_l2_loss(gt, pred)
        losses_1s.append(l1)
        losses_2s.append(l2)
        scene = rec.get("_scene", "all")
        per_scene[scene][0].append(l1)
        per_scene[scene][1].append(l2)
        n_ok += 1
        if args.plot_dir:
            tag = f'{rec.get("_scene", "s")}_{rec.get("_frame", idx)}'
            maybe_plot(gt, pred, os.path.join(args.plot_dir, f"{tag}.png"))

    print(f"\nparsed OK: {n_ok} | unparseable/short: {n_bad}")
    if not n_ok:
        print("no parseable predictions — check the raw `pred` strings in the infer output.")
        return
    # per-scene breakdown (only if scene tags are present and there is >1 scene)
    if len(per_scene) > 1:
        for scene, (l1s, l2s) in sorted(per_scene.items()):
            print(f"\n================  scene: {scene}  (n={len(l1s)})  ================")
            print_l2_loss(l1s, l2s)
        print("\n================  ALL SCENES  ================")
    print_l2_loss(losses_1s, losses_2s)


if __name__ == "__main__":
    main()
