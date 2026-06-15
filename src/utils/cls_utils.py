"""Stub for `utils.cls_utils` (missing from this repo dump). See utils/obj_utils.py
for why these stubs exist. Only needed so `road_collator.py` imports; the symbols
are not used by the DeepSight Bench2Drive AD pipeline."""


def _stub(name):
    def _f(*args, **kwargs):
        raise NotImplementedError(
            f"utils.cls_utils.{name} is a stub — RoadCollector is not used in the "
            "DeepSight Bench2Drive AD pipeline."
        )
    return _f


merge_classes_and_ranges = _stub("merge_classes_and_ranges")
get_range_point = _stub("get_range_point")
