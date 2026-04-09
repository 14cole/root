#!/usr/bin/env python3
from __future__ import annotations

"""
Expand 2D monostatic scattering-width results into an approximate 3D monostatic RCS.

Angle/direction convention (right-handed Cartesian):
- Look vector is `rhat = [cos(el)*cos(az), cos(el)*sin(az), sin(el)]`.
- `az` is measured in the x-y plane from +x toward +y.
- `el` is measured up from the x-y plane toward +z.

Direction cheat-sheet in this frame:
- Forward: +x  -> az=0, el=0
- Aft:     -x  -> az=180 (or -180), el=0
- Left:    +y  -> az=90, el=0
- Right:   -y  -> az=-90 (or 270), el=0
- Up:      +z  -> el=+90 (az ignored at the pole)
- Down:    -z  -> el=-90 (az ignored at the pole)

If your model uses a different body-frame definition of forward/left/up, rotate or
remap the input points/normals to this XYZ convention before expansion.

What this does:
- Reads 2D solver CSV output (frequency, angle, rcs_linear), or .grim output.
- Reads a point cloud with position + normal (x,y,z,nx,ny,nz) and optional weight.
- Optionally reads a STEP model (.stp/.step) to ground points and evaluate visibility shadowing.
- Optionally projects points to nearest STEP triangles and recomputes normals from the model surface.
- For each requested 3D look direction, maps local incidence angle at each point to 2D response.
- Integrates/scatters over the supplied point set directly and combines contributions.

Phase support:
- If 2D input contains GRIM phase (`rcs_phase`) or CSV real/imag columns,
  `combine="coherent_2d_phase"` includes that local 2D scattering phase in coherent summation.

Important assumptions:
- This is an engineering approximation, not a replacement for full 3D EM.
- Best when the CSV point set and weights represent the physical scattering extent.
- If your points only include x/y/z/normals, that is usable, but adding a weight column is recommended.
- When STEP shadowing is enabled, fully shadowed directions are floored by `shadow_floor_db`.
- STEP projection/shadowing requires `cadquery` and can be expensive for large meshes/point sets.
"""

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class StepTriMesh:
    vertices: np.ndarray
    faces: np.ndarray
    tri_v0: np.ndarray
    tri_e1: np.ndarray
    tri_e2: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_diag: float


@dataclass
class Rcs2DCollection:
    tables_by_pol: Dict[str, Rcs2DTable]
    source_kind: str


@dataclass
class Expand3DConfig:
    rcs2d_csv: str
    points_csv: str
    output_grim: str = "rcs3d_expanded.grim"
    model_stp: str = ""
    model_align: str = "ground_z"
    enable_model_shadowing: bool = True
    shadow_floor_db: float = -200.0
    step_tess_tol: float = 1.0e-3
    step_angular_tol_deg: float = 5.0
    project_points_to_step: bool = False
    step_projection_candidates: int = 64
    freq_list_ghz: Optional[List[float]] = None
    az_list_deg: Tuple[float, ...] = (0.0,)
    el_list_deg: Tuple[float, ...] = (0.0,)
    combine: str = "incoherent"
    backface: str = "zero"
    cosine_weight: bool = False
    normalize_weights: bool = True
    write_grim: bool = True
    history: str = ""


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


def _load_2d_rcs_grim(path: str) -> Dict[str, Rcs2DTable]:
    """
    Load a .grim (npz) file exported by grim_io.py.

    Uses first elevation slice (index 0), and keeps all polarization slices.
    """
    with np.load(path, allow_pickle=False) as data:
        if "azimuths" not in data or "frequencies" not in data:
            raise ValueError(f"GRIM file '{path}' is missing required arrays (azimuths, frequencies)")
        azimuths = np.asarray(data["azimuths"], dtype=float)
        freqs = np.asarray(data["frequencies"], dtype=float)

        rcs_power = np.asarray(data["rcs_power"], dtype=float) if "rcs_power" in data else None
        rcs_phase = np.asarray(data["rcs_phase"], dtype=float) if "rcs_phase" in data else None
        pol_labels_raw = np.asarray(data["polarizations"], dtype=str).reshape(-1) if "polarizations" in data else None

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

    pol_count = int(rcs_power.shape[3])
    if pol_labels_raw is None or len(pol_labels_raw) != pol_count:
        if pol_count == 1:
            pol_labels = ["VV"]
        else:
            pol_labels = [f"P{idx}" for idx in range(pol_count)]
    else:
        pol_labels = [str(v).strip().upper() or f"P{idx}" for idx, v in enumerate(pol_labels_raw.tolist())]

    # De-duplicate polarization labels while preserving order.
    seen: Dict[str, int] = {}
    unique_pol_labels: List[str] = []
    for idx, label in enumerate(pol_labels):
        base = label
        if base not in seen:
            seen[base] = 1
            unique_pol_labels.append(base)
            continue
        seen[base] += 1
        unique_pol_labels.append(f"{base}_{seen[base]}")

    tables_by_pol: Dict[str, Rcs2DTable] = {}
    for pi, pol_label in enumerate(unique_pol_labels):
        by_freq: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        amp_by_freq: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for fi, f_ghz in enumerate(freqs):
            vals = rcs_power[:, 0, fi, pi]
            sig = np.real(np.asarray(vals, dtype=np.complex128))
            sig = np.where(np.isfinite(sig), sig, 0.0)
            sig = np.maximum(sig, 0.0)
            by_freq[float(f_ghz)] = (azimuths.astype(float), sig.astype(float))
            if np.any(np.isfinite(rcs_phase[:, 0, fi, pi])):
                amp_vals = np.asarray(rcs_amp[:, 0, fi, pi], dtype=np.complex128)
                amp_r = np.where(np.isfinite(amp_vals.real), amp_vals.real, 0.0)
                amp_i = np.where(np.isfinite(amp_vals.imag), amp_vals.imag, 0.0)
                amp_by_freq[float(f_ghz)] = (azimuths.astype(float), amp_r + 1j * amp_i)

        tables_by_pol[pol_label] = Rcs2DTable(
            angle_col="theta_scat_deg",
            sigma_col="rcs_linear",
            freqs_ghz=np.asarray(sorted(by_freq.keys()), dtype=float),
            by_freq=by_freq,
            amp_by_freq=(amp_by_freq if amp_by_freq else None),
        )
    return tables_by_pol


def _load_2d_rcs(path: str) -> Rcs2DCollection:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".grim":
        return Rcs2DCollection(tables_by_pol=_load_2d_rcs_grim(path), source_kind="grim")
    return Rcs2DCollection(tables_by_pol={"VV": _load_2d_rcs_csv(path)}, source_kind="csv")


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
            w_list.append(float("nan"))
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

    if weight_col is None:
        # If no explicit integration weight was provided, derive spacing-based
        # weights directly from row-order point geometry.
        w = _spacing_weights_from_xyz(xyz)

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


def _spacing_weights_from_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Build point-integration weights from CSV row-order spacing when explicit
    weights are not provided.
    """

    pts = np.asarray(xyz, dtype=float)
    n = int(pts.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    seg = np.where(np.isfinite(seg) & (seg > 0.0), seg, 0.0)
    w = np.zeros(n, dtype=float)
    w[0] = 0.5 * float(seg[0])
    w[-1] = 0.5 * float(seg[-1])
    if n > 2:
        w[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    if float(np.sum(w)) <= 0.0:
        w = np.ones(n, dtype=float)
    return w


def _as_xyz_tuple(value: Any) -> Tuple[float, float, float]:
    if hasattr(value, "toTuple"):
        t = value.toTuple()
        return float(t[0]), float(t[1]), float(t[2])
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return float(value.x), float(value.y), float(value.z)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return float(value[0]), float(value[1]), float(value[2])
    raise ValueError("Unsupported vertex type from STEP tessellation.")


def _as_face_tuple(face: Any) -> Tuple[int, int, int]:
    if hasattr(face, "A") and hasattr(face, "B") and hasattr(face, "C"):
        return int(face.A), int(face.B), int(face.C)
    if isinstance(face, (list, tuple)) and len(face) >= 3:
        return int(face[0]), int(face[1]), int(face[2])
    raise ValueError("Unsupported triangle index type from STEP tessellation.")


def _load_step_tri_mesh(path: str, tess_tol: float, angular_tol_deg: float) -> StepTriMesh:
    """
    Load and tessellate a STEP model into a triangle mesh.

    Requires `cadquery` (OpenCascade-backed).
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"STEP model not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".stp", ".step"}:
        raise ValueError(f"model_stp must be a .stp/.step file, got '{path}'.")

    try:
        import cadquery as cq  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "STEP support requires cadquery. Install cadquery (or disable model_stp/model shadowing)."
        ) from exc

    shape_obj = cq.importers.importStep(path)
    shape = shape_obj.val() if hasattr(shape_obj, "val") else shape_obj

    t_tol = max(float(tess_tol), 1.0e-6)
    a_tol_deg = max(float(angular_tol_deg), 0.1)
    a_tol_rad = math.radians(a_tol_deg)

    try:
        verts_raw, faces_raw = shape.tessellate(t_tol, a_tol_rad)
    except TypeError:
        verts_raw, faces_raw = shape.tessellate(t_tol)

    vertices = np.asarray([_as_xyz_tuple(v) for v in verts_raw], dtype=float)
    faces = np.asarray([_as_face_tuple(f) for f in faces_raw], dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("STEP tessellation produced invalid/empty vertices.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("STEP tessellation produced invalid/empty faces.")

    # Handle possible 1-based triangle indexing.
    if int(np.min(faces)) == 1 and int(np.max(faces)) == len(vertices):
        faces = faces - 1

    if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
        raise ValueError("STEP tessellation returned out-of-range triangle indices.")

    tri_v0 = vertices[faces[:, 0]]
    tri_v1 = vertices[faces[:, 1]]
    tri_v2 = vertices[faces[:, 2]]
    tri_e1 = tri_v1 - tri_v0
    tri_e2 = tri_v2 - tri_v0

    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if bbox_diag <= 0.0:
        bbox_diag = 1.0

    return StepTriMesh(
        vertices=vertices,
        faces=faces,
        tri_v0=tri_v0,
        tri_e1=tri_e1,
        tri_e2=tri_e2,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_diag=bbox_diag,
    )


def _align_points_to_model(xyz: np.ndarray, model: StepTriMesh, align_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    mode = str(align_mode or "ground_z").strip().lower()
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud xyz must be N x 3.")
    shift = np.zeros(3, dtype=float)

    if mode == "none":
        return points.copy(), shift
    if mode not in {"ground_z", "center_xy_ground_z"}:
        raise ValueError("model_align must be one of: none, ground_z, center_xy_ground_z")

    p_min = np.min(points, axis=0)
    p_max = np.max(points, axis=0)
    p_ctr = 0.5 * (p_min + p_max)
    m_ctr = 0.5 * (model.bbox_min + model.bbox_max)

    shift[2] = float(model.bbox_min[2] - p_min[2])
    if mode == "center_xy_ground_z":
        shift[0] = float(m_ctr[0] - p_ctr[0])
        shift[1] = float(m_ctr[1] - p_ctr[1])

    return points + shift[None, :], shift


def _closest_point_on_triangle(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """
    Closest point on triangle ABC to point P (Christer Ericson, RTCD).
    """

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1.0e-30)
        return a + v * ab

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1.0e-30)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-30)
        return b + w * (c - b)

    denom = max(va + vb + vc, 1.0e-30)
    inv_denom = 1.0 / denom
    v = vb * inv_denom
    w = vc * inv_denom
    return a + ab * v + ac * w


def _project_points_and_normals_to_step(
    points_xyz: np.ndarray,
    points_normals: np.ndarray,
    model: StepTriMesh,
    candidate_triangles: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project each point onto its nearest STEP triangle and assign that triangle normal.

    Triangle normal is flipped to align with the original point normal when possible.
    """

    pts = np.asarray(points_xyz, dtype=float)
    nrms_in = np.asarray(points_normals, dtype=float)
    n_points = int(pts.shape[0])
    if n_points == 0:
        return pts.copy(), nrms_in.copy(), np.zeros(0, dtype=float)

    tri_v0 = model.tri_v0
    tri_v1 = model.tri_v0 + model.tri_e1
    tri_v2 = model.tri_v0 + model.tri_e2
    n_tri = int(tri_v0.shape[0])
    if n_tri == 0:
        raise ValueError("STEP mesh has no triangles for projection.")

    tri_centers = (tri_v0 + tri_v1 + tri_v2) / 3.0
    tri_normals = np.cross(model.tri_e1, model.tri_e2)
    tri_norm_mag = np.linalg.norm(tri_normals, axis=1)
    valid_norm = tri_norm_mag > 1.0e-15
    tri_normals = np.where(valid_norm[:, None], tri_normals / np.maximum(tri_norm_mag[:, None], 1.0e-30), 0.0)

    k = int(max(1, candidate_triangles))
    k = min(k, n_tri)

    proj_pts = np.zeros_like(pts)
    proj_nrm = np.zeros_like(nrms_in)
    proj_dist = np.zeros(n_points, dtype=float)

    for i in range(n_points):
        p = pts[i]
        center_d2 = np.sum((tri_centers - p[None, :]) ** 2, axis=1)
        if k < n_tri:
            cand_idx = np.argpartition(center_d2, k - 1)[:k]
        else:
            cand_idx = np.arange(n_tri, dtype=int)

        best_d2 = float("inf")
        best_cp = p
        best_n = np.asarray([0.0, 0.0, 1.0], dtype=float)

        for ti in cand_idx.tolist():
            cp = _closest_point_on_triangle(p, tri_v0[ti], tri_v1[ti], tri_v2[ti])
            d2 = float(np.dot(cp - p, cp - p))
            if d2 < best_d2:
                best_d2 = d2
                best_cp = cp
                n_tri_i = tri_normals[ti]
                if float(np.linalg.norm(n_tri_i)) <= 1.0e-15:
                    n_tri_i = np.asarray([0.0, 0.0, 1.0], dtype=float)
                best_n = n_tri_i

        n_old = nrms_in[i]
        if float(np.linalg.norm(n_old)) > 1.0e-15 and float(np.dot(best_n, n_old)) < 0.0:
            best_n = -best_n

        proj_pts[i] = best_cp
        proj_nrm[i] = best_n / max(float(np.linalg.norm(best_n)), 1.0e-30)
        proj_dist[i] = math.sqrt(max(best_d2, 0.0))

    return proj_pts, proj_nrm, proj_dist


def _compute_visibility_mask_for_direction(
    points_xyz: np.ndarray,
    model: StepTriMesh,
    rhat: np.ndarray,
) -> np.ndarray:
    """
    Visibility for monostatic directions using ray-triangle occlusion.

    Ray direction is +rhat (toward radar/view). Any hit blocks the point.
    """

    pts = np.asarray(points_xyz, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)

    d = np.asarray(rhat, dtype=float)
    dn = float(np.linalg.norm(d))
    if dn <= 1e-15:
        return np.ones(len(pts), dtype=bool)
    d = d / dn

    tri_v0 = model.tri_v0
    tri_e1 = model.tri_e1
    tri_e2 = model.tri_e2

    pvec = np.cross(np.broadcast_to(d, tri_e2.shape), tri_e2)              # (F,3)
    det = np.einsum("ij,ij->i", tri_e1, pvec)                              # (F,)
    valid = np.abs(det) > 1.0e-12
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]

    ray_offset = max(1.0e-9, 1.0e-6 * float(model.bbox_diag))
    t_eps = max(1.0e-9, 0.5 * ray_offset)
    visible = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):
        origin = pts[i] + ray_offset * d
        tvec = origin[None, :] - tri_v0                                     # (F,3)
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det                    # (F,)
        qvec = np.cross(tvec, tri_e1)                                       # (F,3)
        v = np.einsum("j,ij->i", d, qvec) * inv_det                        # (F,)
        t = np.einsum("ij,ij->i", tri_e2, qvec) * inv_det                  # (F,)

        hit = valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > t_eps)
        if np.any(hit):
            visible[i] = False

    return visible


def _ensure_grim_ext(path: str) -> str:
    return path if path.lower().endswith(".grim") else f"{path}.grim"


def _write_3d_grim(
    output_path: str,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    frequencies: np.ndarray,
    polarizations: np.ndarray,
    rcs_power: np.ndarray,
    rcs_phase: np.ndarray,
    source_path: str,
    history: str,
) -> str:
    out = _ensure_grim_ext(output_path)
    units = json.dumps({"azimuth": "deg", "elevation": "deg", "frequency": "GHz"})
    with open(out, "wb") as f:
        np.savez(
            f,
            azimuths=np.asarray(azimuths, dtype=float),
            elevations=np.asarray(elevations, dtype=float),
            frequencies=np.asarray(frequencies, dtype=float),
            polarizations=np.asarray(polarizations, dtype=str),
            rcs_power=np.asarray(rcs_power, dtype=np.float32),
            rcs_phase=np.asarray(rcs_phase, dtype=np.float32),
            rcs_domain="power_phase",
            power_domain="linear_rcs",
            source_path=str(source_path),
            history=str(history),
            units=units,
            phase_reference="origin=(0,0,0), convention=exp(-jwt), monostatic far-field amplitude",
        )
    return os.path.abspath(out)


def _compute_sigma3d_for_direction(
    table: Rcs2DTable,
    points: PointCloud,
    freq_ghz: float,
    az_deg: float,
    el_deg: float,
    combine_mode: str,
    backface_mode: str,
    cosine_weight: bool,
    visibility_mask: Optional[np.ndarray] = None,
    shadow_floor_linear: float = 1.0e-20,
) -> Tuple[float, Optional[complex], Dict[str, float]]:
    if combine_mode == "coherent_2d_phase" and not table.amp_by_freq:
        raise ValueError(
            "combine=coherent_2d_phase requires phase-capable 2D input "
            "(GRIM with finite rcs_phase, or CSV with rcs_amp_real/rcs_amp_imag)."
        )

    rhat = _rhat_from_az_el(az_deg, el_deg)
    lam = C0 / (freq_ghz * 1e9)
    vis = None if visibility_mask is None else np.asarray(visibility_mask, dtype=bool).reshape(-1)
    if vis is not None and vis.shape[0] != len(points.weights):
        raise ValueError("visibility_mask length must match point count.")

    cos_inc = points.normals @ rhat
    angles = np.degrees(np.arccos(np.clip(cos_inc, -1.0, 1.0)))

    sigma_local = np.zeros(len(points.weights), dtype=float)
    phase_local = np.zeros(len(points.weights), dtype=float)
    for i, ang in enumerate(angles):
        if vis is not None and not bool(vis[i]):
            s2 = 0.0
            amp2 = 0.0 + 0.0j
        elif backface_mode == "zero" and cos_inc[i] <= 0.0:
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

        # Convert local 2D width to 3D estimate and integrate over point weights.
        # point.weights are explicit CSV weights when present; otherwise they are
        # spacing-derived from xyz row order.
        s3 = 2.0 * s2 / max(lam, 1e-15)
        sigma_local[i] = s3 * float(points.weights[i])

    if combine_mode in {"coherent_zero_phase", "coherent_2d_phase"}:
        k = 2.0 * math.pi / max(lam, 1e-15)
        geom_phase = 2.0 * k * (points.xyz @ rhat)
        if combine_mode == "coherent_2d_phase":
            phase = np.exp(1j * (geom_phase + phase_local))
        else:
            phase = np.exp(1j * geom_phase)
        amp = np.sqrt(np.maximum(sigma_local, 0.0)) * phase
        amp_total = complex(np.sum(amp))
        sigma_total = float(np.abs(amp_total) ** 2)
    else:
        sigma_total = float(np.sum(np.maximum(sigma_local, 0.0)))
        amp_total = None

    # Shadow floor for fully shadowed directions.
    if vis is not None and (not bool(np.any(vis))):
        sigma_total = max(float(sigma_total), float(max(shadow_floor_linear, 0.0)))

    meta = {
        "illuminated_fraction": float(np.mean(cos_inc > 0.0)),
        "weight_sum": float(np.sum(points.weights)),
        "visibility_fraction": float(np.mean(vis)) if vis is not None and len(vis) > 0 else 1.0,
        "phase_used": float(1.0 if combine_mode == "coherent_2d_phase" else 0.0),
    }
    return max(sigma_total, 0.0), amp_total, meta


def expand_2d_to_3d(config: Expand3DConfig) -> List[Dict[str, float]]:
    table_collection = _load_2d_rcs(config.rcs2d_csv)
    tables_by_pol = dict(table_collection.tables_by_pol)
    if not tables_by_pol:
        raise ValueError("2D input does not contain any usable polarization data.")
    points = _load_points_csv(config.points_csv, normalize_weights=bool(config.normalize_weights))

    model_mesh: Optional[StepTriMesh] = None
    model_align_shift = np.zeros(3, dtype=float)
    projection_distances = np.zeros(0, dtype=float)
    projection_applied = False
    visibility_cache: Dict[Tuple[float, float], np.ndarray] = {}
    shadow_floor_linear = 10.0 ** (float(config.shadow_floor_db) / 10.0)
    shadow_floor_linear = max(float(shadow_floor_linear), 0.0)

    model_path = str(config.model_stp or "").strip()
    shadow_enabled = bool(config.enable_model_shadowing)
    if bool(config.project_points_to_step) and not model_path:
        raise ValueError("project_points_to_step=True requires model_stp to be set.")
    if model_path:
        model_mesh = _load_step_tri_mesh(
            path=model_path,
            tess_tol=float(config.step_tess_tol),
            angular_tol_deg=float(config.step_angular_tol_deg),
        )
        aligned_xyz, model_align_shift = _align_points_to_model(
            points.xyz,
            model=model_mesh,
            align_mode=str(config.model_align),
        )
        points = PointCloud(
            xyz=np.asarray(aligned_xyz, dtype=float),
            normals=np.asarray(points.normals, dtype=float),
            weights=np.asarray(points.weights, dtype=float),
        )
        if bool(config.project_points_to_step):
            proj_xyz, proj_nrm, proj_dist = _project_points_and_normals_to_step(
                points_xyz=points.xyz,
                points_normals=points.normals,
                model=model_mesh,
                candidate_triangles=int(config.step_projection_candidates),
            )
            points = PointCloud(
                xyz=np.asarray(proj_xyz, dtype=float),
                normals=np.asarray(proj_nrm, dtype=float),
                weights=np.asarray(points.weights, dtype=float),
            )
            projection_distances = np.asarray(proj_dist, dtype=float)
            projection_applied = True

    az_list = [float(v) for v in config.az_list_deg]
    el_list = [float(v) for v in config.el_list_deg]
    if not az_list:
        raise ValueError("az_list_deg must not be empty")
    if not el_list:
        raise ValueError("el_list_deg must not be empty")

    combine_mode = str(config.combine)
    if combine_mode not in {"incoherent", "coherent_zero_phase", "coherent_2d_phase"}:
        raise ValueError("combine must be one of: incoherent, coherent_zero_phase, coherent_2d_phase")

    backface_mode = str(config.backface)
    if backface_mode not in {"zero", "lookup"}:
        raise ValueError("backface must be one of: zero, lookup")

    if config.freq_list_ghz is not None:
        freq_list = [float(f) for f in config.freq_list_ghz]
        if not freq_list:
            raise ValueError("freq_list_ghz must not be empty when provided")
    else:
        first_pol = next(iter(tables_by_pol.keys()))
        freq_list = [float(f) for f in tables_by_pol[first_pol].freqs_ghz.tolist()]

    freq_axis = np.asarray(sorted({float(v) for v in freq_list}), dtype=float)
    az_axis = np.asarray(sorted({float(v) for v in az_list}), dtype=float)
    el_axis = np.asarray(sorted({float(v) for v in el_list}), dtype=float)
    pol_axis = np.asarray(list(tables_by_pol.keys()), dtype=str)

    if len(freq_axis) == 0 or len(az_axis) == 0 or len(el_axis) == 0 or len(pol_axis) == 0:
        raise ValueError("Expansion axes must be non-empty (frequency, azimuth, elevation, polarization).")

    az_idx = {float(v): i for i, v in enumerate(az_axis.tolist())}
    el_idx = {float(v): i for i, v in enumerate(el_axis.tolist())}
    freq_idx = {float(v): i for i, v in enumerate(freq_axis.tolist())}
    pol_idx = {str(v): i for i, v in enumerate(pol_axis.tolist())}

    shape = (len(az_axis), len(el_axis), len(freq_axis), len(pol_axis))
    rcs_power = np.full(shape, np.nan, dtype=np.float32)
    rcs_phase = np.full(shape, np.nan, dtype=np.float32)

    if model_mesh is not None and shadow_enabled:
        for az in az_axis.tolist():
            for el in el_axis.tolist():
                key = (float(az), float(el))
                visibility_cache[key] = _compute_visibility_mask_for_direction(
                    points_xyz=points.xyz,
                    model=model_mesh,
                    rhat=_rhat_from_az_el(float(az), float(el)),
                )

    out_rows: List[Dict[str, float]] = []
    for pol_name, table in tables_by_pol.items():
        for f_ghz in freq_axis.tolist():
            for az in az_axis.tolist():
                for el in el_axis.tolist():
                    vis_mask = visibility_cache.get((float(az), float(el))) if visibility_cache else None
                    sigma, amp_total, meta = _compute_sigma3d_for_direction(
                        table=table,
                        points=points,
                        freq_ghz=float(f_ghz),
                        az_deg=float(az),
                        el_deg=float(el),
                        combine_mode=combine_mode,
                        backface_mode=backface_mode,
                        cosine_weight=bool(config.cosine_weight),
                        visibility_mask=vis_mask,
                        shadow_floor_linear=shadow_floor_linear,
                    )

                    if amp_total is not None and np.isfinite(amp_total.real) and np.isfinite(amp_total.imag):
                        phase_rad = float(np.angle(amp_total))
                        amp_real = float(np.real(amp_total))
                        amp_imag = float(np.imag(amp_total))
                        phase_deg = float(np.degrees(phase_rad))
                    else:
                        phase_rad = float("nan")
                        amp_real = 0.0
                        amp_imag = 0.0
                        phase_deg = float("nan")

                    rcs_power[
                        az_idx[float(az)],
                        el_idx[float(el)],
                        freq_idx[float(f_ghz)],
                        pol_idx[str(pol_name)],
                    ] = float(max(sigma, 0.0))
                    rcs_phase[
                        az_idx[float(az)],
                        el_idx[float(el)],
                        freq_idx[float(f_ghz)],
                        pol_idx[str(pol_name)],
                    ] = phase_rad

                    out_rows.append(
                        {
                            "polarization": str(pol_name),
                            "frequency_ghz": float(f_ghz),
                            "az_deg": float(az),
                            "el_deg": float(el),
                            "rcs3d_linear": float(sigma),
                            "rcs3d_dbsm": float(10.0 * math.log10(max(sigma, 1e-30))),
                            "rcs_amp_real": amp_real,
                            "rcs_amp_imag": amp_imag,
                            "rcs_amp_phase_deg": phase_deg,
                            "illuminated_fraction": float(meta["illuminated_fraction"]),
                            "visibility_fraction": float(meta["visibility_fraction"]),
                            "weight_sum": float(meta["weight_sum"]),
                            "phase_used": float(meta["phase_used"]),
                            "shadow_floor_db": float(config.shadow_floor_db),
                            "projection_applied": float(1.0 if projection_applied else 0.0),
                            "projection_dist_mean": float(np.mean(projection_distances)) if projection_applied else 0.0,
                            "projection_dist_max": float(np.max(projection_distances)) if projection_applied else 0.0,
                        }
                    )

    out_rows.sort(key=lambda r: (r["polarization"], r["frequency_ghz"], r["az_deg"], r["el_deg"]))

    wrote_targets: List[str] = []
    if bool(config.write_grim):
        if not str(config.output_grim).strip():
            raise ValueError("output_grim must be set when write_grim=True.")
        history = str(config.history).strip()
        if not history:
            history = (
                "expanded_2d_to_3d "
                f"(combine={combine_mode}, backface={backface_mode}, "
                f"cosine_weight={bool(config.cosine_weight)}, source_kind={table_collection.source_kind}, "
                f"model_stp={'yes' if model_mesh is not None else 'no'}, "
                f"project_to_step={'yes' if projection_applied else 'no'}, "
                f"shadowing={'yes' if bool(visibility_cache) else 'no'}, "
                f"shadow_floor_db={float(config.shadow_floor_db):g})"
            )
        grim_path = _write_3d_grim(
            output_path=str(config.output_grim),
            azimuths=az_axis,
            elevations=el_axis,
            frequencies=freq_axis,
            polarizations=pol_axis,
            rcs_power=rcs_power,
            rcs_phase=rcs_phase,
            source_path=os.path.abspath(str(config.rcs2d_csv)),
            history=history,
        )
        wrote_targets.append(grim_path)

    if not wrote_targets:
        print(
            f"Computed {len(out_rows)} expanded rows but no files were written "
            "(write_grim=False)."
        )
    else:
        print(
            f"Wrote {len(out_rows)} rows across {len(wrote_targets)} output file(s): "
            + ", ".join(wrote_targets)
            + f" (combine={combine_mode}, backface={backface_mode}, points={len(points.weights)})."
        )
        if model_mesh is not None:
            print(
                "Model alignment/shadowing: "
                f"model='{model_path}', align='{str(config.model_align)}', "
                f"shift=({model_align_shift[0]:.6g}, {model_align_shift[1]:.6g}, {model_align_shift[2]:.6g}), "
                f"shadowing={'on' if bool(visibility_cache) else 'off'}."
            )
            if projection_applied:
                print(
                    "STEP projection: "
                    f"points={len(projection_distances)}, "
                    f"mean_dist={float(np.mean(projection_distances)):.6g}, "
                    f"max_dist={float(np.max(projection_distances)):.6g}, "
                    f"candidates={int(config.step_projection_candidates)}."
                )
    return out_rows


def main() -> int:
    # Edit these values when running this script directly.
    config = Expand3DConfig(
        rcs2d_csv="",
        points_csv="",
        output_grim="rcs3d_expanded.grim",
        # Optional STEP model for grounding + shadowing.
        # model_stp="model.step",
        model_align="ground_z",
        enable_model_shadowing=True,
        project_points_to_step=False,
        step_projection_candidates=64,
        shadow_floor_db=-200.0,
        step_tess_tol=1.0e-3,
        step_angular_tol_deg=5.0,
        # freq_list_ghz=[2.0, 4.0, 6.0],
        az_list_deg=(0.0, 45.0, 90.0),
        # Real 3D elevation angles to expand to:
        el_list_deg=(-30.0, 0.0, 30.0),
        combine="incoherent",
        backface="zero",
        cosine_weight=False,
        normalize_weights=True,
        write_grim=True,
    )
    if not config.rcs2d_csv or not config.points_csv:
        raise ValueError(
            "Set Expand3DConfig.rcs2d_csv and Expand3DConfig.points_csv in main(), "
            "or import this module and call expand_2d_to_3d(Expand3DConfig(...))."
        )
    expand_2d_to_3d(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
