from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import scipy  # type: ignore
except Exception:
    scipy = None

try:
    import mpmath  # type: ignore
except Exception:
    mpmath = None

from geometry_io import build_geometry_snapshot, parse_geometry
from rcs_solver import solve_monostatic_rcs_2d


def _parse_list(text: str, field_name: str) -> List[float]:
    tokens = [tok for tok in re.split(r"[,\s]+", (text or "").strip()) if tok]
    if not tokens:
        raise ValueError(f"{field_name}: no values were provided.")
    out: List[float] = []
    for tok in tokens:
        try:
            out.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"{field_name}: invalid numeric token '{tok}'.") from exc
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "sha256": "",
            "bytes": 0,
        }
    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _normalize_units(units: str) -> str:
    u = str(units).strip().lower()
    if u in {"in", "inch"}:
        return "inches"
    if u in {"m", "meter"}:
        return "meters"
    if u not in {"inches", "meters"}:
        raise ValueError("units must be one of: inches, meters, in, inch, m, meter")
    return u


def _normalize_pol(pol: str) -> str:
    p = str(pol).strip().upper()
    if p in {"TE", "VV", "V", "VERTICAL"}:
        return "TE"
    if p in {"TM", "HH", "H", "HORIZONTAL"}:
        return "TM"
    raise ValueError("polarization must be TE/TM (or VV/HH aliases)")


def _resolve_fort_candidates(base_dir: Path, flag: int) -> List[Path]:
    name = f"fort.{int(flag)}"
    return [base_dir / name, Path.cwd() / name]


def _discover_material_files(geo_path: Path) -> List[Dict[str, Any]]:
    with open(geo_path, "r") as f:
        text = f.read()
    _, _, ibcs_entries, dielectric_entries = parse_geometry(text)

    flags: List[Tuple[str, int]] = []
    for row in ibcs_entries:
        if not row:
            continue
        try:
            flag = int(round(float(row[0])))
        except (TypeError, ValueError):
            continue
        if flag > 50:
            flags.append(("ibc", flag))
    for row in dielectric_entries:
        if not row:
            continue
        try:
            flag = int(round(float(row[0])))
        except (TypeError, ValueError):
            continue
        if flag > 50:
            flags.append(("dielectric", flag))

    out: List[Dict[str, Any]] = []
    base_dir = geo_path.parent
    for kind, flag in sorted(set(flags), key=lambda t: (t[1], t[0])):
        candidates = _resolve_fort_candidates(base_dir, flag)
        found: Path | None = None
        for c in candidates:
            if c.is_file():
                found = c
                break
        out.append(
            {
                "kind": kind,
                "flag": int(flag),
                "resolved": str(found) if found is not None else "",
                "candidates": [str(c) for c in candidates],
                "file": _file_info(found) if found is not None else _file_info(candidates[0]),
            }
        )
    return out


def _collect_code_hashes(solver_root: Path) -> Dict[str, Dict[str, Any]]:
    names = [
        "rcs_solver.py",
        "headless_solver.py",
        "geometry_io.py",
        "grim_io.py",
        "solver_quality.py",
        "solver_tab.py",
        "main.py",
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        out[name] = _file_info(solver_root / name)
    return out


def _build_sample_signature(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = sorted(
        [
            (
                float(s.get("frequency_ghz", 0.0)),
                float(s.get("theta_scat_deg", 0.0)),
                round(float(s.get("rcs_db", 0.0)), 10),
                round(float(s.get("rcs_amp_phase_deg", 0.0)), 10),
            )
            for s in samples
        ],
        key=lambda t: (t[0], t[1]),
    )
    blob = json.dumps(rows, separators=(",", ":"))
    sig = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {
        "sample_count": len(rows),
        "signature_sha256": sig,
        "rows": [
            {
                "frequency_ghz": r[0],
                "theta_scat_deg": r[1],
                "rcs_db": r[2],
                "rcs_amp_phase_deg": r[3],
            }
            for r in rows
        ],
    }


def _run_probe(
    geometry_path: Path,
    units: str,
    polarization: str,
    freqs: List[float],
    elevs: List[float],
) -> Dict[str, Any]:
    with open(geometry_path, "r") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
    snapshot = build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    result = solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=freqs,
        elevations_deg=elevs,
        polarization=polarization,
        geometry_units=units,
        material_base_dir=str(geometry_path.parent),
        compute_condition_number=True,
    )
    metadata = dict(result.get("metadata", {}) or {})
    samples = list(result.get("samples", []) or [])
    sig = _build_sample_signature(samples)
    return {
        "metadata": metadata,
        "sample_signature": sig,
        "sample_stats": {
            "rcs_db_min": float(min((float(s.get("rcs_db", 0.0)) for s in samples), default=0.0)),
            "rcs_db_max": float(max((float(s.get("rcs_db", 0.0)) for s in samples), default=0.0)),
            "rcs_db_mean": float(np.mean([float(s.get("rcs_db", 0.0)) for s in samples])) if samples else 0.0,
        },
    }


def _make_report(
    geometry: str,
    units: str,
    polarization: str,
    freq_list: str,
    elev_list: str,
) -> Dict[str, Any]:
    geo_path = Path(geometry).expanduser().resolve()
    if not geo_path.is_file():
        raise FileNotFoundError(f"Geometry file not found: {geo_path}")

    units_norm = _normalize_units(units)
    pol_norm = _normalize_pol(polarization)
    freqs = _parse_list(freq_list, "Frequencies")
    elevs = _parse_list(elev_list, "Elevations")
    if any(float(f) <= 0.0 for f in freqs):
        raise ValueError("Frequencies must be positive GHz values.")

    solver_root = Path(__file__).resolve().parent
    report = {
        "report_type": "2dsolver_repro_check",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "inputs": {
            "geometry_path": str(geo_path),
            "units": units_norm,
            "polarization": pol_norm,
            "frequencies_ghz": [float(v) for v in freqs],
            "elevations_deg": [float(v) for v in elevs],
        },
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": getattr(scipy, "__version__", None),
            "mpmath_version": getattr(mpmath, "__version__", None),
        },
        "files": {
            "geometry": _file_info(geo_path),
            "material_files": _discover_material_files(geo_path),
            "code_hashes": _collect_code_hashes(solver_root),
        },
    }
    report["probe"] = _run_probe(
        geometry_path=geo_path,
        units=units_norm,
        polarization=pol_norm,
        freqs=freqs,
        elevs=elevs,
    )
    return report


def _compare_rows(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    a_map = {(float(r["frequency_ghz"]), float(r["theta_scat_deg"])): float(r["rcs_db"]) for r in a_rows}
    b_map = {(float(r["frequency_ghz"]), float(r["theta_scat_deg"])): float(r["rcs_db"]) for r in b_rows}
    a_keys = set(a_map.keys())
    b_keys = set(b_map.keys())
    common = sorted(a_keys & b_keys)
    missing = sorted(a_keys - b_keys)
    extra = sorted(b_keys - a_keys)
    if not common:
        return {
            "common_points": 0,
            "missing_points": len(missing),
            "extra_points": len(extra),
            "max_abs_rcs_db_delta": None,
            "mean_abs_rcs_db_delta": None,
        }
    deltas = [abs(b_map[k] - a_map[k]) for k in common]
    return {
        "common_points": len(common),
        "missing_points": len(missing),
        "extra_points": len(extra),
        "max_abs_rcs_db_delta": float(max(deltas)),
        "mean_abs_rcs_db_delta": float(sum(deltas) / len(deltas)),
    }


def _compare_reports(path_a: Path, path_b: Path) -> Dict[str, Any]:
    with open(path_a, "r") as f:
        a = json.load(f)
    with open(path_b, "r") as f:
        b = json.load(f)

    diffs: List[str] = []

    def check(label: str, va: Any, vb: Any) -> None:
        if va != vb:
            diffs.append(label)

    check("inputs.units", a.get("inputs", {}).get("units"), b.get("inputs", {}).get("units"))
    check("inputs.polarization", a.get("inputs", {}).get("polarization"), b.get("inputs", {}).get("polarization"))
    check("inputs.frequencies_ghz", a.get("inputs", {}).get("frequencies_ghz"), b.get("inputs", {}).get("frequencies_ghz"))
    check("inputs.elevations_deg", a.get("inputs", {}).get("elevations_deg"), b.get("inputs", {}).get("elevations_deg"))
    check("files.geometry.sha256", a.get("files", {}).get("geometry", {}).get("sha256"), b.get("files", {}).get("geometry", {}).get("sha256"))
    check("probe.metadata.bessel_backend", a.get("probe", {}).get("metadata", {}).get("bessel_backend"), b.get("probe", {}).get("metadata", {}).get("bessel_backend"))
    check(
        "probe.metadata.complex_hankel_backend",
        a.get("probe", {}).get("metadata", {}).get("complex_hankel_backend"),
        b.get("probe", {}).get("metadata", {}).get("complex_hankel_backend"),
    )
    check(
        "probe.sample_signature.signature_sha256",
        a.get("probe", {}).get("sample_signature", {}).get("signature_sha256"),
        b.get("probe", {}).get("sample_signature", {}).get("signature_sha256"),
    )

    code_a = a.get("files", {}).get("code_hashes", {}) or {}
    code_b = b.get("files", {}).get("code_hashes", {}) or {}
    code_diff: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(code_a.keys()) | set(code_b.keys())):
        ha = (code_a.get(key, {}) or {}).get("sha256", "")
        hb = (code_b.get(key, {}) or {}).get("sha256", "")
        if ha != hb:
            code_diff[key] = {"a": ha, "b": hb}
    if code_diff:
        diffs.append("files.code_hashes")

    rows_a = list(a.get("probe", {}).get("sample_signature", {}).get("rows", []) or [])
    rows_b = list(b.get("probe", {}).get("sample_signature", {}).get("rows", []) or [])
    row_cmp = _compare_rows(rows_a, rows_b)

    return {
        "report_type": "2dsolver_repro_compare",
        "report_a": str(path_a.resolve()),
        "report_b": str(path_b.resolve()),
        "match": len(diffs) == 0 and (row_cmp.get("max_abs_rcs_db_delta") in {0.0, None}),
        "differences": diffs,
        "code_hash_differences": code_diff,
        "sample_rcs_db_delta": row_cmp,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate/compare reproducibility reports for 2D solver outputs across machines."
    )
    p.add_argument("--geometry", default="", help="Path to input .geo file for probe solve.")
    p.add_argument("--units", default="meters", help="Geometry units: meters/inches.")
    p.add_argument("--pol", "--polarization", dest="polarization", default="TM", help="Polarization: TM/TE.")
    p.add_argument("--freq-list", default="4.0", help="Comma/space-separated frequency list in GHz.")
    p.add_argument("--elev-list", default="0,45,90", help="Comma/space-separated elevation list in degrees.")
    p.add_argument("--json-output", default="", help="Optional report output path.")
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("REPORT_A", "REPORT_B"),
        help="Compare two previously generated repro reports.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.compare:
        path_a = Path(args.compare[0]).expanduser().resolve()
        path_b = Path(args.compare[1]).expanduser().resolve()
        report = _compare_reports(path_a, path_b)
        if args.json_output:
            out = Path(args.json_output).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return 0 if bool(report.get("match", False)) else 2

    if not str(args.geometry).strip():
        raise ValueError("--geometry is required unless --compare is used.")

    report = _make_report(
        geometry=str(args.geometry),
        units=str(args.units),
        polarization=str(args.polarization),
        freq_list=str(args.freq_list),
        elev_list=str(args.elev_list),
    )
    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
