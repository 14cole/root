import json
import cmath
import math
import os
from typing import Any, Dict, List

import numpy as np

EPS = 1e-12


def _ensure_grim_ext(path: str) -> str:
    return path if path.lower().endswith(".grim") else f"{path}.grim"


def _suffix_for_incidence(theta_inc_deg: float) -> str:
    value = f"{theta_inc_deg:.6f}".rstrip("0").rstrip(".")
    value = value.replace("-", "m").replace(".", "p")
    return f"inc_{value or '0'}"


def _build_grid_for_samples(
    samples: List[Dict[str, Any]],
    polarization: str,
    source_path: str = "",
    history: str = "",
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("No samples available to export.")

    azimuths = np.asarray(sorted({float(row["theta_scat_deg"]) for row in samples}), dtype=float)
    elevations = np.asarray([0.0], dtype=float)
    frequencies = np.asarray(sorted({float(row["frequency_ghz"]) for row in samples}), dtype=float)
    polarizations = np.asarray([polarization], dtype=str)

    shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
    rcs_phase = np.full(shape, np.nan, dtype=np.float32)
    rcs_power = np.full(shape, np.nan, dtype=np.float32)

    az_index = {value: i for i, value in enumerate(azimuths)}
    f_index = {value: i for i, value in enumerate(frequencies)}

    for row in samples:
        az = float(row["theta_scat_deg"])
        freq = float(row["frequency_ghz"])
        lin = float(row.get("rcs_linear", 0.0))
        if not math.isfinite(lin) or lin < 0.0:
            lin = 0.0

        amp_real = float(row.get("rcs_amp_real", 0.0))
        amp_imag = float(row.get("rcs_amp_imag", 0.0))
        if not math.isfinite(amp_real):
            amp_real = 0.0
        if not math.isfinite(amp_imag):
            amp_imag = 0.0

        idx = (az_index[az], 0, f_index[freq], 0)
        amp_value_raw = complex(amp_real, amp_imag)
        # Normalize exported complex field so |rcs|^2 matches linear RCS power.
        phase = cmath.phase(amp_value_raw) if abs(amp_value_raw) > EPS else 0.0
        amp_value = cmath.rect(math.sqrt(max(lin, 0.0)), phase)
        existing_power = rcs_power[idx]
        if not np.isnan(existing_power):
            if abs(existing_power - lin) > EPS:
                raise ValueError(
                    f"Duplicate sample conflict at az={az}, el=0, f={freq}, pol={polarization}."
                )
            existing_phase = rcs_phase[idx]
            phase_value = float(cmath.phase(amp_value))
            if np.isfinite(existing_phase) and abs(existing_phase - phase_value) > EPS:
                raise ValueError(
                    f"Duplicate amplitude conflict at az={az}, el=0, f={freq}, pol={polarization}."
                )
            continue
        rcs_power[idx] = float(max(lin, 0.0))
        rcs_phase[idx] = float(cmath.phase(amp_value))

    return {
        "azimuths": azimuths,
        "elevations": elevations,
        "frequencies": frequencies,
        "polarizations": polarizations,
        "rcs_power": rcs_power,
        "rcs_phase": rcs_phase,
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "source_path": source_path,
        "history": history,
        "units": json.dumps({"azimuth": "deg", "elevation": "deg", "frequency": "GHz"}),
        "phase_reference": "origin=(0,0), convention=exp(-jwt), monostatic far-field amplitude",
    }


def _save_grim_npz(payload: Dict[str, Any], path: str) -> str:
    out = _ensure_grim_ext(path)
    with open(out, "wb") as f:
        np.savez(
            f,
            azimuths=payload["azimuths"],
            elevations=payload["elevations"],
            frequencies=payload["frequencies"],
            polarizations=payload["polarizations"],
            rcs_power=payload["rcs_power"],
            rcs_phase=payload["rcs_phase"],
            rcs_domain=payload["rcs_domain"],
            power_domain=payload["power_domain"],
            source_path=payload["source_path"],
            history=payload["history"],
            units=payload["units"],
            phase_reference=payload["phase_reference"],
        )
    return out


def export_result_to_grim(
    result: Dict[str, Any],
    output_path: str,
    polarization: str | None = None,
    source_path: str = "",
    history: str = "",
) -> List[str]:
    samples = result.get("samples", []) or []
    if not samples:
        raise ValueError("No solver samples were returned, nothing to export.")

    pol = (polarization or result.get("polarization") or "VV").strip().upper()
    mode = str(result.get("scattering_mode", "monostatic")).strip().lower()

    if mode != "bistatic":
        payload = _build_grid_for_samples(samples, pol, source_path=source_path, history=history)
        return [os.path.abspath(_save_grim_npz(payload, output_path))]

    by_inc: Dict[float, List[Dict[str, Any]]] = {}
    for row in samples:
        inc = float(row.get("theta_inc_deg", 0.0))
        by_inc.setdefault(inc, []).append(row)

    rootspec = _ensure_grim_ext(output_path)
    root_no_ext = rootspec[:-5]
    written: List[str] = []
    for inc in sorted(by_inc.keys()):
        payload = _build_grid_for_samples(
            by_inc[inc],
            pol,
            source_path=source_path,
            history=(history + f" | theta_inc_deg={inc:g}").strip(" |"),
        )
        out = f"{root_no_ext}_{_suffix_for_incidence(inc)}.grim"
        written.append(os.path.abspath(_save_grim_npz(payload, out)))
    return written
