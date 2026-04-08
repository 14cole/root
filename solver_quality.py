from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np


def _scale_n_property(token: Any, factor: float) -> str:
    try:
        n_value = int(float(str(token).strip()))
    except (TypeError, ValueError):
        n_value = 1

    refine = max(float(factor), 1.0)
    if n_value > 0:
        return str(max(1, int(round(float(n_value) * refine))))
    if n_value < 0:
        mag = max(1, int(round(float(abs(n_value)) * refine)))
        return str(-mag)
    return "0"


def scale_snapshot_panel_density(snapshot: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """
    Return a deep-copied geometry snapshot with refined panel density controls.

    Only properties[1] (n property) is scaled. Positive values remain positive
    and negative values remain negative while increasing |n|.
    """

    scaled = copy.deepcopy(snapshot)
    refine = max(float(factor), 1.0)
    for seg in list(scaled.get("segments", []) or []):
        props = list(seg.get("properties", []) or [])
        if len(props) < 2:
            props.extend(["1"] * (2 - len(props)))
        props[1] = _scale_n_property(props[1], refine)
        seg["properties"] = props
    return scaled


def _samples_to_map(result: Dict[str, Any]) -> Dict[Tuple[float, float], float]:
    out: Dict[Tuple[float, float], float] = {}
    for row in list(result.get("samples", []) or []):
        freq = round(float(row.get("frequency_ghz", 0.0)), 12)
        elev = round(float(row.get("theta_scat_deg", 0.0)), 12)
        out[(freq, elev)] = float(row.get("rcs_db", 0.0))
    return out


def evaluate_mesh_convergence(
    base_result: Dict[str, Any],
    fine_result: Dict[str, Any],
    rms_limit_db: float = 1.0,
    max_abs_limit_db: float = 3.0,
) -> Dict[str, Any]:
    base_map = _samples_to_map(base_result)
    fine_map = _samples_to_map(fine_result)
    missing_in_fine = sorted(set(base_map.keys()) - set(fine_map.keys()))
    missing_in_base = sorted(set(fine_map.keys()) - set(base_map.keys()))

    base_md = dict(base_result.get("metadata", {}) or {})
    fine_md = dict(fine_result.get("metadata", {}) or {})
    base_panels = int(base_md.get("panel_count", 0) or 0)
    fine_panels = int(fine_md.get("panel_count", 0) or 0)

    report: Dict[str, Any] = {
        "enabled": True,
        "base_panel_count": base_panels,
        "fine_panel_count": fine_panels,
        "rms_limit_db": float(rms_limit_db),
        "max_abs_limit_db": float(max_abs_limit_db),
        "passed": False,
        "reason": "",
        "rms_db": float("inf"),
        "max_abs_db": float("inf"),
        "sample_count": 0,
        "per_frequency_rms_db": {},
        "per_frequency_max_abs_db": {},
    }

    if not base_map:
        report["reason"] = "Base solve returned no samples."
        return report
    if missing_in_fine or missing_in_base:
        report["reason"] = "Base/fine sample grids do not match."
        report["missing_in_fine_count"] = len(missing_in_fine)
        report["missing_in_base_count"] = len(missing_in_base)
        return report

    keys = sorted(base_map.keys())
    deltas = np.asarray([base_map[k] - fine_map[k] for k in keys], dtype=float)
    rms_db = float(math.sqrt(float(np.mean(deltas * deltas))))
    max_abs_db = float(np.max(np.abs(deltas)))
    report["rms_db"] = rms_db
    report["max_abs_db"] = max_abs_db
    report["sample_count"] = len(keys)

    per_freq: Dict[float, List[float]] = {}
    for (freq, _elev), delta in zip(keys, deltas.tolist()):
        per_freq.setdefault(float(freq), []).append(float(delta))
    per_freq_rms: Dict[str, float] = {}
    per_freq_max: Dict[str, float] = {}
    for freq in sorted(per_freq.keys()):
        arr = np.asarray(per_freq[freq], dtype=float)
        label = f"{freq:g}"
        per_freq_rms[label] = float(math.sqrt(float(np.mean(arr * arr))))
        per_freq_max[label] = float(np.max(np.abs(arr)))
    report["per_frequency_rms_db"] = per_freq_rms
    report["per_frequency_max_abs_db"] = per_freq_max

    pass_rms = math.isfinite(rms_db) and rms_db <= float(rms_limit_db)
    pass_max = math.isfinite(max_abs_db) and max_abs_db <= float(max_abs_limit_db)
    report["passed"] = bool(pass_rms and pass_max)
    if report["passed"]:
        report["reason"] = "Mesh convergence criteria met."
    else:
        reasons: List[str] = []
        if not pass_rms:
            reasons.append(f"rms_db={rms_db:.6g} exceeds {float(rms_limit_db):.6g}")
        if not pass_max:
            reasons.append(f"max_abs_db={max_abs_db:.6g} exceeds {float(max_abs_limit_db):.6g}")
        if fine_panels <= base_panels:
            reasons.append("fine mesh did not increase panel count")
        report["reason"] = "; ".join(reasons) if reasons else "Mesh convergence criteria not met."
    return report
