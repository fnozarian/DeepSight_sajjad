"""Stub for `utils.obj_utils` (missing from this repo dump).

`src/llamafactory/data/road_collator.py` imports `visual_objs` at module load
time, and `src/llamafactory/data/__init__.py` imports `RoadCollector`
unconditionally — so without this module the entire `llamafactory.data` package
(and therefore *all* training/inference entry points) fails to import.

`RoadCollector` is the road-graph / nuScenes-style collator and is NOT used by
the DeepSight Bench2Drive AD pipeline (which uses `ADCollector`). These symbols
are never actually called on that path, so a stub that satisfies the import is
sufficient. Added as a NEW file (no existing file is modified).
"""


def visual_objs(*args, **kwargs):
    raise NotImplementedError(
        "utils.obj_utils.visual_objs is a stub — RoadCollector is not used in the "
        "DeepSight Bench2Drive AD pipeline."
    )
