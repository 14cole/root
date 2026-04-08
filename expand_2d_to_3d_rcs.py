#!/usr/bin/env python3
from __future__ import annotations

"""
Expand 2D monostatic scattering-width results into an approximate 3D monostatic RCS.

What this does:
- Reads 2D solver CSV output (frequency, angle, rcs_linear), or .grim output.
- Reads a point cloud with position + normal (x,y,z,nx,ny,nz) and optional weight.
- For each requested 3D look direction, maps local incidence angle at each point to 2D response.
- Applies finite-length extrusion factor and combines contributions into an approximate 3D RCS.

Phase support:
- If 2D input contains GRIM phase (`rcs_phase`) or CSV real/imag columns, `--combine coherent_2d_phase`
  includes that local 2D scattering phase in coherent summation.

Important assumptions:
- This is an engineering approximation, not a replacement for full 3D EM.
- Best for electrically long extrusions with slowly varying cross-section along axis.
- If your points only include x/y/z/normals, that is usable, but adding a weight column is recommended.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

C0 = 299_792_458.0


@dataclass
class Rcs2DTable:
    angle_col: str
    sigma_col: str
    freqs_ghz: np.ndarray
    by_freq: Dict[float, Tuple[np.ndarray, np.ndarray]]
    amp_by_freq: Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]] = None


@dataclass
class PointCloud:
    xyz: np.ndarray
    normals: np.ndarray
    weights: np.ndarray


def _parse_list(text: str, field_name: str) -> List[float]:
    tokens = [tok.strip() for tok in (text or "").replace(";", ",").split(",") if tok.strip()]
    if not tokens:
        raise ValueError(f"{field_name}: no values provided")
    out: List[float] = []
    for tok in tokens:
        try:
            out.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"{field_name}: invalid numeric token '{tok}'") from exc
    return out


def _parse_vec3(text: str, field_name: str) -> np.ndarray:
    vals = _parse_list(text, field_name)
    if len(vals) != 3:
        raise ValueError(f"{field_name}: expected exactly 3 values")
    v = np.asarray(vals, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 1e-15:
        raise ValueError(f"{field_name}: zero-length vector")
    return v / n


def _load_2d_rcs_csv(path: str) -> Rcs2DTable:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"2D CSV '{path}' has no rows")

    fieldnames = set(rows[0].keys())
    angle_candidates = ["theta_scat_deg", "theta_inc_deg", "theta_deg", "angle_deg"]
    sigma_candidates = ["rcs_linear", "sigma2d_linear", "sigma_linear"]

    angle_col = next((c for c in angle_candidates if c in fieldnames), None)
    sigma_col = next((c for c in sigma_candidates if c in fieldnames), None)
    if angle_col is None:
        raise ValueError(f"2D CSV '{path}' must contain one of: {angle_candidates}")
    if sigma_col is None:
        raise ValueError(f"2D CSV '{path}' must contain one of: {sigma_candidates}")
    if "frequency_ghz" not in fieldnames:
        raise ValueError("2D CSV must contain column 'frequency_ghz'")

    has_amp_cols = "rcs_amp_real" in fieldnames and "rcs_amp_imag" in fieldnames
    grouped: Dict[float, List[Tuple[float, float]]] = {}
    grouped_amp: Dict[float, List[Tuple[float, complex]]] = {}
    for r in rows:
        try:
            f_ghz = float(r["frequency_ghz"])
            ang = float(r[angle_col])
            sig = float(r[sigma_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(f_ghz) or not np.isfinite(ang) or not np.isfinite(sig):
            continue
        grouped.setdefault(f_ghz, []).append((ang, max(sig, 0.0)))
        if has_amp_cols:
            try:
                amp_r = float(r.get("rcs_amp_real", 0.0))
                amp_i = float(r.get("rcs_amp_imag", 0.0))
            except (TypeError, ValueError):
                amp_r = 0.0
                amp_i = 0.0
            if not np.isfinite(amp_r):
                amp_r = 0.0
            if not np.isfinite(amp_i):
                amp_i = 0.0
            grouped_amp.setdefault(f_ghz, []).append((ang, complex(amp_r, amp_i)))

    if not grouped:
        raise ValueError(f"2D CSV '{path}' has no valid numeric rows")

    by_freq: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    amp_by_freq: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for f_ghz, pairs in grouped.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        # Keep last value if duplicate angle appears.
        dedup: Dict[float, float] = {}
        for ang, sig in pairs_sorted:
            dedup[ang] = sig
        angs = np.asarray(sorted(dedup.keys()), dtype=float)
        sigs = np.asarray([dedup[a] for a in angs], dtype=float)
        by_freq[f_ghz] = (angs, sigs)
        if has_amp_cols:
            pairs_amp_sorted = sorted(grouped_amp.get(f_ghz, []), key=lambda x: x[0])
            dedup_amp: Dict[float, complex] = {}
            for ang, amp in pairs_amp_sorted:
                dedup_amp[ang] = amp
            amps = np.asarray(
                [
                    dedup_amp.get(a, complex(math.sqrt(max(float(dedup[a]), 0.0)), 0.0))
                    for a in angs
                ],
                dtype=np.complex128,
            )
            amp_by_freq[f_ghz] = (angs, amps)

    freqs = np.asarray(sorted(by_freq.keys()), dtype=float)
    return Rcs2DTable(
        angle_col=angle_col,
        sigma_col=sigma_col,
        freqs_ghz=freqs,
        by_freq=by_freq,
        amp_by_freq=(amp_by_freq if has_amp_cols else None),
    )


def _load_2d_rcs_grim(path: str) -> Rcs2DTable:
    """
    Load a .grim (npz) file exported by grim_io.py.

    Uses first elevation and first polarization slice.
    """
    with np.load(path, allow_pickle=False) as data:
        if "azimuths" not in data or "frequencies" not in data:
            raise ValueError(f"GRIM file '{path}' is missing required arrays (azimuths, frequencies)")
        azimuths = np.asarray(data["azimuths"], dtype=float)
        freqs = np.asarray(data["frequencies"], dtype=float)

        rcs_power = np.asarray(data["rcs_power"], dtype=float) if "rcs_power" in data else None
        rcs_phase = np.asarray(data["rcs_phase"], dtype=float) if "rcs_phase" in data else None

        # Legacy fallbacks for older GRIM schemas.
        rcs_raw = np.asarray(data["rcs"], dtype=np.complex128) if "rcs" in data else None
        if "rcs_amp" in data:
            legacy_amp = np.asarray(data["rcs_amp"], dtype=np.complex128)
        else:
            legacy_amp = None

    if rcs_power is not None and rcs_power.ndim != 4:
        raise ValueError(f"GRIM file '{path}' has unexpected rcs_power rank {rcs_power.ndim}; expected 4D")
    if rcs_phase is not None and rcs_phase.ndim != 4:
        raise ValueError(f"GRIM file '{path}' has unexpected rcs_phase rank {rcs_phase.ndim}; expected 4D")
    if rcs_raw is not None and rcs_raw.ndim != 4:
        raise ValueError(f"GRIM file '{path}' has unexpected rcs rank {rcs_raw.ndim}; expected 4D")
    if legacy_amp is not None and legacy_amp.ndim != 4:
        raise ValueError(f"GRIM file '{path}' has unexpected legacy rcs_amp rank {legacy_amp.ndim}; expected 4D")

    if rcs_power is None:
        if rcs_raw is not None:
            rcs_power = np.abs(rcs_raw) ** 2 if np.iscomplexobj(rcs_raw) else np.real(rcs_raw)
        elif legacy_amp is not None:
            rcs_power = np.abs(legacy_amp) ** 2
        else:
            raise ValueError(f"GRIM file '{path}' is missing usable RCS content")
    rcs_power = np.asarray(rcs_power, dtype=float)
    rcs_power = np.where(np.isfinite(rcs_power), np.maximum(rcs_power, 0.0), np.nan)

    if rcs_phase is None:
        if legacy_amp is not None:
            rcs_phase = np.angle(legacy_amp)
        elif rcs_raw is not None and np.iscomplexobj(rcs_raw):
            rcs_phase = np.angle(rcs_raw)
        else:
            rcs_phase = np.full(rcs_power.shape, np.nan, dtype=float)
    rcs_phase = np.asarray(rcs_phase, dtype=float)
    rcs_phase[~np.isfinite(rcs_phase)] = np.nan

    rcs_amp = np.full(rcs_power.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    amp_valid = np.isfinite(rcs_power) & np.isfinite(rcs_phase)
    if np.any(amp_valid):
        rcs_amp[amp_valid] = np.sqrt(rcs_power[amp_valid]) * np.exp(1j * rcs_phase[amp_valid])

    if len(azimuths) == 0 or len(freqs) == 0:
        raise ValueError(f"GRIM file '{path}' has empty azimuth or frequency axis")

    by_freq: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    amp_by_freq: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for fi, f_ghz in enumerate(freqs):
        vals = rcs_power[:, 0, fi, 0]
        sig = np.real(np.asarray(vals, dtype=np.complex128))
        sig = np.where(np.isfinite(sig), sig, 0.0)
        sig = np.maximum(sig, 0.0)
        by_freq[float(f_ghz)] = (azimuths.astype(float), sig.astype(float))
        if np.any(np.isfinite(rcs_phase[:, 0, fi, 0])):
            amp_vals = np.asarray(rcs_amp[:, 0, fi, 0], dtype=np.complex128)
            amp_r = np.where(np.isfinite(amp_vals.real), amp_vals.real, 0.0)
            amp_i = np.where(np.isfinite(amp_vals.imag), amp_vals.imag, 0.0)
            amp_by_freq[float(f_ghz)] = (azimuths.astype(float), amp_r + 1j * amp_i)

    return Rcs2DTable(
        angle_col="theta_scat_deg",
        sigma_col="rcs_linear",
        freqs_ghz=np.asarray(sorted(by_freq.keys()), dtype=float),
        by_freq=by_freq,
        amp_by_freq=(amp_by_freq if amp_by_freq else None),
    )


def _load_2d_rcs(path: str) -> Rcs2DTable:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".grim":
        return _load_2d_rcs_grim(path)
    return _load_2d_rcs_csv(path)


def _interp_sigma_vs_angle(angles_deg: np.ndarray, sigmas: np.ndarray, query_angle_deg: float) -> float:
    # Use clamped interpolation across available angle range.
    q = float(query_angle_deg)
    if len(angles_deg) == 1:
        return float(sigmas[0])
    q = min(max(q, float(angles_deg[0])), float(angles_deg[-1]))
    return float(np.interp(q, angles_deg, sigmas))


def _interp_complex_vs_angle(angles_deg: np.ndarray, values: np.ndarray, query_angle_deg: float) -> complex:
    q = float(query_angle_deg)
    if len(angles_deg) == 1:
        return complex(values[0])
    q = min(max(q, float(angles_deg[0])), float(angles_deg[-1]))
    real = np.interp(q, angles_deg, values.real)
    imag = np.interp(q, angles_deg, values.imag)
    return complex(real, imag)


def _lookup_sigma2d(table: Rcs2DTable, freq_ghz: float, angle_deg: float) -> float:
    freqs = table.freqs_ghz
    if len(freqs) == 1:
        a, s = table.by_freq[float(freqs[0])]
        return _interp_sigma_vs_angle(a, s, angle_deg)

    if freq_ghz <= float(freqs[0]):
        a, s = table.by_freq[float(freqs[0])]
        return _interp_sigma_vs_angle(a, s, angle_deg)
    if freq_ghz >= float(freqs[-1]):
        a, s = table.by_freq[float(freqs[-1])]
        return _interp_sigma_vs_angle(a, s, angle_deg)

    hi_idx = int(np.searchsorted(freqs, freq_ghz, side="right"))
    lo_idx = hi_idx - 1
    f0 = float(freqs[lo_idx])
    f1 = float(freqs[hi_idx])
    a0, s0 = table.by_freq[f0]
    a1, s1 = table.by_freq[f1]
    y0 = _interp_sigma_vs_angle(a0, s0, angle_deg)
    y1 = _interp_sigma_vs_angle(a1, s1, angle_deg)
    t = (freq_ghz - f0) / max(f1 - f0, 1e-15)
    return float((1.0 - t) * y0 + t * y1)


def _lookup_amp2d(table: Rcs2DTable, freq_ghz: float, angle_deg: float) -> Optional[complex]:
    amp_by_freq = table.amp_by_freq
    if not amp_by_freq:
        return None

    freqs = table.freqs_ghz
    if len(freqs) == 1:
        a, amp = amp_by_freq[float(freqs[0])]
        return _interp_complex_vs_angle(a, amp, angle_deg)

    if freq_ghz <= float(freqs[0]):
        a, amp = amp_by_freq[float(freqs[0])]
        return _interp_complex_vs_angle(a, amp, angle_deg)
    if freq_ghz >= float(freqs[-1]):
        a, amp = amp_by_freq[float(freqs[-1])]
        return _interp_complex_vs_angle(a, amp, angle_deg)

    hi_idx = int(np.searchsorted(freqs, freq_ghz, side="right"))
    lo_idx = hi_idx - 1
    f0 = float(freqs[lo_idx])
    f1 = float(freqs[hi_idx])
    a0, amp0 = amp_by_freq[f0]
    a1, amp1 = amp_by_freq[f1]
    y0 = _interp_complex_vs_angle(a0, amp0, angle_deg)
    y1 = _interp_complex_vs_angle(a1, amp1, angle_deg)
    t = (freq_ghz - f0) / max(f1 - f0, 1e-15)
    return complex((1.0 - t) * y0 + t * y1)


def _load_points_csv(path: str, normalize_weights: bool) -> PointCloud:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Points CSV '{path}' has no rows")

    required = ["x", "y", "z", "nx", "ny", "nz"]
    fields = set(rows[0].keys())
    for c in required:
        if c not in fields:
            raise ValueError(f"Points CSV is missing required column '{c}'")

    weight_col = None
    for c in ["weight", "w", "ds", "arc_len", "panel_length"]:
        if c in fields:
            weight_col = c
            break

    xyz_list: List[List[float]] = []
    nrm_list: List[List[float]] = []
    w_list: List[float] = []

    for r in rows:
        try:
            x = float(r["x"])
            y = float(r["y"])
            z = float(r["z"])
            nx = float(r["nx"])
            ny = float(r["ny"])
            nz = float(r["nz"])
        except (TypeError, ValueError):
            continue

        n = np.asarray([nx, ny, nz], dtype=float)
        nn = float(np.linalg.norm(n))
        if nn <= 1e-15:
            continue
        n = n / nn

        xyz_list.append([x, y, z])
        nrm_list.append([float(n[0]), float(n[1]), float(n[2])])

        if weight_col is None:
            w_list.append(1.0)
        else:
            try:
                w = float(r[weight_col])
            except (TypeError, ValueError):
                w = 0.0
            w_list.append(max(w, 0.0))

    if not xyz_list:
        raise ValueError(f"Points CSV '{path}' has no valid rows after parsing")

    xyz = np.asarray(xyz_list, dtype=float)
    nrms = np.asarray(nrm_list, dtype=float)
    w = np.asarray(w_list, dtype=float)

    if np.sum(w) <= 0.0:
        w = np.ones_like(w)

    if normalize_weights:
        w = w / float(np.sum(w))

    return PointCloud(xyz=xyz, normals=nrms, weights=w)


def _rhat_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(float(az_deg))
    el = math.radians(float(el_deg))
    ce = math.cos(el)
    return np.asarray([ce * math.cos(az), ce * math.sin(az), math.sin(el)], dtype=float)


def _finite_length_factor(length_m: float, wavelength_m: float, rhat: np.ndarray, axis_hat: np.ndarray) -> float:
    # |L * sinc(k*L*sin(theta))|^2 form, where sin(theta) is axial direction cosine.
    # Here u = component along axis; x = k L u.
    k = 2.0 * math.pi / max(wavelength_m, 1e-15)
    u = float(np.dot(rhat, axis_hat))
    x = k * length_m * u
    return float((length_m * np.sinc(x / math.pi)) ** 2)


def _compute_sigma3d_for_direction(
    table: Rcs2DTable,
    points: PointCloud,
    freq_ghz: float,
    az_deg: float,
    el_deg: float,
    length_m: float,
    axis_hat: np.ndarray,
    combine_mode: str,
    backface_mode: str,
    cosine_weight: bool,
) -> Tuple[float, Dict[str, float]]:
    if combine_mode == "coherent_2d_phase" and not table.amp_by_freq:
        raise ValueError(
            "combine=coherent_2d_phase requires phase-capable 2D input "
            "(GRIM with finite rcs_phase, or CSV with rcs_amp_real/rcs_amp_imag)."
        )

    rhat = _rhat_from_az_el(az_deg, el_deg)
    lam = C0 / (freq_ghz * 1e9)
    len_factor = _finite_length_factor(length_m, lam, rhat, axis_hat)

    cos_inc = points.normals @ rhat
    angles = np.degrees(np.arccos(np.clip(cos_inc, -1.0, 1.0)))

    sigma_local = np.zeros(len(points.weights), dtype=float)
    phase_local = np.zeros(len(points.weights), dtype=float)
    for i, ang in enumerate(angles):
        if backface_mode == "zero" and cos_inc[i] <= 0.0:
            s2 = 0.0
            amp2 = 0.0 + 0.0j
        else:
            s2 = _lookup_sigma2d(table, freq_ghz, float(ang))
            amp2 = _lookup_amp2d(table, freq_ghz, float(ang)) or (0.0 + 0.0j)

        if cosine_weight:
            w_cos = max(float(cos_inc[i]), 0.0)
            s2 *= w_cos
            amp2 *= math.sqrt(w_cos)
        phase_local[i] = float(np.angle(amp2))

        # Convert local 2D width to 3D finite-length estimate.
        s3 = (2.0 * s2 / max(lam, 1e-15)) * len_factor
        sigma_local[i] = s3 * float(points.weights[i])

    if combine_mode in {"coherent_zero_phase", "coherent_2d_phase"}:
        k = 2.0 * math.pi / max(lam, 1e-15)
        geom_phase = 2.0 * k * (points.xyz @ rhat)
        if combine_mode == "coherent_2d_phase":
            phase = np.exp(1j * (geom_phase + phase_local))
        else:
            phase = np.exp(1j * geom_phase)
        amp = np.sqrt(np.maximum(sigma_local, 0.0)) * phase
        sigma_total = float(np.abs(np.sum(amp)) ** 2)
    else:
        sigma_total = float(np.sum(np.maximum(sigma_local, 0.0)))

    meta = {
        "len_factor": float(len_factor),
        "illuminated_fraction": float(np.mean(cos_inc > 0.0)),
        "axial_cosine": float(np.dot(rhat, axis_hat)),
        "phase_used": float(1.0 if combine_mode == "coherent_2d_phase" else 0.0),
    }
    return max(sigma_total, 0.0), meta


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Approximate 3D RCS from 2D solver output + 3D point/normal map.")
    p.add_argument(
        "--rcs2d-csv",
        required=True,
        help="2D source file: solver CSV or .grim (uses first elevation/polarization slice).",
    )
    p.add_argument("--points-csv", required=True, help="CSV with x,y,z,nx,ny,nz and optional weight.")
    p.add_argument("--output-csv", default="rcs3d_expanded.csv", help="Output CSV path.")

    p.add_argument("--freq-list", default="", help="Optional frequency list in GHz. Default: all frequencies in rcs2d CSV.")
    p.add_argument("--az-list", required=True, help="Azimuth list in degrees, e.g. '0,15,30'.")
    p.add_argument("--el-list", default="0", help="Elevation list in degrees. Grid with az-list.")

    p.add_argument("--length-m", type=float, required=True, help="Known extrusion length in meters.")
    p.add_argument("--axis", default="0,0,1", help="Extrusion axis unit direction (x,y,z), default 0,0,1.")

    p.add_argument(
        "--combine",
        choices=["incoherent", "coherent_zero_phase", "coherent_2d_phase"],
        default="incoherent",
        help=(
            "How to combine point contributions. "
            "coherent_2d_phase uses phase from 2D GRIM rcs_phase (if available) plus geometric phase."
        ),
    )
    p.add_argument(
        "--backface",
        choices=["zero", "lookup"],
        default="zero",
        help="Backface treatment based on normal·rhat <= 0.",
    )
    p.add_argument(
        "--cosine-weight",
        action="store_true",
        help="Apply extra max(normal·rhat,0) weighting to local sigma2d lookup.",
    )
    p.add_argument(
        "--no-normalize-weights",
        action="store_true",
        help="Use point weights as-is (default normalizes to sum=1).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    table = _load_2d_rcs(args.rcs2d_csv)
    points = _load_points_csv(args.points_csv, normalize_weights=(not args.no_normalize_weights))

    axis_hat = _parse_vec3(args.axis, "axis")
    if args.length_m <= 0.0:
        raise ValueError("--length-m must be > 0")

    az_list = _parse_list(args.az_list, "az-list")
    el_list = _parse_list(args.el_list, "el-list")

    if args.freq_list.strip():
        freq_list = _parse_list(args.freq_list, "freq-list")
    else:
        freq_list = [float(f) for f in table.freqs_ghz.tolist()]

    out_rows: List[Dict[str, float]] = []
    for f_ghz in freq_list:
        for az in az_list:
            for el in el_list:
                sigma, meta = _compute_sigma3d_for_direction(
                    table=table,
                    points=points,
                    freq_ghz=float(f_ghz),
                    az_deg=float(az),
                    el_deg=float(el),
                    length_m=float(args.length_m),
                    axis_hat=axis_hat,
                    combine_mode=args.combine,
                    backface_mode=args.backface,
                    cosine_weight=bool(args.cosine_weight),
                )
                out_rows.append(
                    {
                        "frequency_ghz": float(f_ghz),
                        "az_deg": float(az),
                        "el_deg": float(el),
                        "rcs3d_linear": float(sigma),
                        "rcs3d_dbsm": float(10.0 * math.log10(max(sigma, 1e-30))),
                        "len_factor": float(meta["len_factor"]),
                        "illuminated_fraction": float(meta["illuminated_fraction"]),
                        "axial_cosine": float(meta["axial_cosine"]),
                        "phase_used": float(meta["phase_used"]),
                    }
                )

    out_rows.sort(key=lambda r: (r["frequency_ghz"], r["az_deg"], r["el_deg"]))
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frequency_ghz",
                "az_deg",
                "el_deg",
                "rcs3d_linear",
                "rcs3d_dbsm",
                "len_factor",
                "illuminated_fraction",
                "axial_cosine",
                "phase_used",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"Wrote {len(out_rows)} rows to {args.output_csv} "
        f"(combine={args.combine}, backface={args.backface}, points={len(points.weights)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
