"""audit/report.py - the JSON the verdict is written from (only measured/solved values, no opinions)."""
from typing import Optional

from ..phase1.stage1_validation import ACCEPT_THRESHOLD
from ..phase1.stage2_gn_source import GN_MAX_ITER, GN_THRESH

_NAMES = {"s_lm": "landmark_consistency", "s_sym": "symmetry", "s_res": "resolution",
          "s_blur": "sharpness", "s_pose": "pose"}


def validation_dict(v) -> dict:
    scores = {name: round(float(getattr(v, k)), 3) for k, name in _NAMES.items()}
    weakest = min(scores, key=scores.get)
    return {"accepted": bool(v.accepted), "accept_threshold": ACCEPT_THRESHOLD,
            **scores, "weakest": weakest}


def residual_summary(history) -> dict:
    """Start/end residual of the source Gauss-Newton fit and whether it converged."""
    if not history:
        return {}
    start, end = history[0]["residual_before"], history[-1]["residual_after"]
    last = history[-1]
    return {"start": round(start, 1), "end": round(end, 1),
            "reduction_pct": round(100.0 * (1.0 - end / max(start, 1e-9)), 1),
            "iterations": len(history), "max_iterations": GN_MAX_ITER,
            "converged": bool(last["accepted"] and last["step_norm"] < GN_THRESH)}


def build_report(session=None, res=None, sam_measures: Optional[dict] = None,
                 missing=None, validation=None) -> dict:
    """session/res come from a finished run; for a REJECTED source pass validation= only."""
    v = validation if validation is not None else getattr(session, "validation", None)
    report = {"decision": "REJECTED" if (v is not None and not v.accepted and res is None) else None}
    if v is not None:
        report["validation"] = validation_dict(v)
    if res is not None:
        report["decision"] = res.decision
        report["quality"] = round(float(res.quality), 3)
        report["head_pose_deg"] = {k: round(float(x), 1) for k, x in res.params.head_pose_deg.items()}
        report["region_error_px"] = {k: round(float(x), 1) for k, x in (res.region_error or {}).items()}
    hist = getattr(getattr(session, "source_state", None), "residual_history", None)
    if hist:
        report["residual_curve"] = residual_summary(hist)
        report["residual_history"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items()}
                                       for h in hist]
    report["measured_from_masks"] = dict(sam_measures or {})
    report["regions_missing"] = list(missing or [])
    return report
