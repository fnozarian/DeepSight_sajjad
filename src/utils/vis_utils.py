"""Stub for `utils.vis_utils` (missing from this repo dump). See utils/obj_utils.py
for why these stubs exist. Only needed so `road_collator.py` imports; the symbols
are not used by the DeepSight Bench2Drive AD pipeline."""


def _stub(name):
    def _f(*args, **kwargs):
        raise NotImplementedError(
            f"utils.vis_utils.{name} is a stub — RoadCollector is not used in the "
            "DeepSight Bench2Drive AD pipeline."
        )
    return _f


visual_road = _stub("visual_road")
get_sub_type_color_2 = _stub("get_sub_type_color_2")
visual_line_with_arrow = _stub("visual_line_with_arrow")
