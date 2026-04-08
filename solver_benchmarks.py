from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from geometry_io import build_geometry_snapshot, parse_geometry
from rcs_solver import solve_monostatic_rcs_2d


def _make_pec_circle_snapshot(radius: float, n_per_arc: int) -> Dict[str, Any]:
    r = float(radius)
    n = int(max(2, n_per_arc))
    geo_text = f"""Title: PEC Circle Benchmark
Segment: circle arc
properties: 2 {n} 90.0 0 0 0
{r:.12g} 0.0 0.0 {r:.12g}
0.0 {r:.12g} {-r:.12g} 0.0
{-r:.12g} 0.0 0.0 {-r:.12g}
0.0 {-r:.12g} {r:.12g} 0.0
IBCS:
1 0.0 0.0 0.0
Dielectrics:
1 0.0 0.0 0.0 0.0
"""
    title, segments, ibcs_entries, dielectric_entries = parse_geometry(geo_text)
    return build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)


def _solve_curve(
    snapshot: Dict[str, Any],
    frequency_ghz: float,
    elevations_deg: List[float],
    polarization: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    result = solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[frequency_ghz],
        elevations_deg=elevations_deg,
        polarization=polarization,
        geometry_units="meters",
        material_base_dir=".",
    )
    samples = sorted(result.get("samples", []), key=lambda row: float(row["theta_scat_deg"]))
    elevs = np.asarray([float(row["theta_scat_deg"]) for row in samples], dtype=float)
    rcs_db = np.asarray([float(row["rcs_db"]) for row in samples], dtype=float)
    return elevs, rcs_db, result


def _rms_delta_db(a_db: np.ndarray, b_db: np.ndarray) -> float:
    if len(a_db) != len(b_db):
        raise ValueError("Benchmark vectors have mismatched lengths.")
    diff = a_db - b_db
    return float(np.sqrt(np.mean(diff * diff)))


def run_pec_circle_benchmark_suite(
    radius_m: float = 0.5,
    frequency_ghz: float = 1.0,
    elevations_step_deg: float = 5.0,
    mesh_levels: List[int] | None = None,
    pols: List[str] | None = None,
) -> Dict[str, Any]:
    if mesh_levels is None:
        mesh_levels = [6, 12, 24]
    if pols is None:
        pols = ["TM", "TE"]

    elevs = np.arange(0.0, 180.0 + 1e-9, float(elevations_step_deg), dtype=float).tolist()
    report: Dict[str, Any] = {
        "benchmark": "pec_circle",
        "radius_m": float(radius_m),
        "frequency_ghz": float(frequency_ghz),
        "elevations_count": len(elevs),
        "mesh_levels": list(mesh_levels),
        "polarizations": {},
    }

    pass_all = True
    for pol in pols:
        pol_key = str(pol).upper()
        curves: Dict[int, np.ndarray] = {}
        panel_counts: Dict[int, int] = {}
        residual_means: Dict[int, float] = {}

        for n in mesh_levels:
            snap = _make_pec_circle_snapshot(radius_m, n)
            _, curve_db, result = _solve_curve(snap, frequency_ghz, elevs, pol_key)
            curves[n] = curve_db
            md = result.get("metadata", {}) or {}
            panel_counts[n] = int(md.get("panel_count", 0))
            residual_means[n] = float(md.get("residual_norm_mean", 0.0))

        finest_n = mesh_levels[-1]
        finest_curve = curves[finest_n]
        isotropy_std_db = float(np.std(finest_curve))
        isotropy_peak_to_peak_db = float(np.max(finest_curve) - np.min(finest_curve))

        convergence_rms: Dict[str, float] = {}
        convergence_maxabs: Dict[str, float] = {}
        pair_order: List[str] = []
        for i in range(len(mesh_levels) - 1):
            n_lo = mesh_levels[i]
            n_hi = mesh_levels[i + 1]
            key = f"{n_lo}->{n_hi}"
            delta = curves[n_lo] - curves[n_hi]
            convergence_rms[key] = _rms_delta_db(curves[n_lo], curves[n_hi])
            convergence_maxabs[key] = float(np.max(np.abs(delta)))
            pair_order.append(key)

        # Practical thresholds for current solver maturity.
        isotropy_std_limit = 1.5
        mesh_rms_limit = 3.0
        iso_pass = isotropy_std_db <= isotropy_std_limit
        finest_pair = pair_order[-1] if pair_order else ""
        finest_pair_rms = convergence_rms.get(finest_pair, 0.0)
        conv_pass = finest_pair_rms <= mesh_rms_limit
        pol_pass = bool(iso_pass and conv_pass)
        pass_all = pass_all and pol_pass

        report["polarizations"][pol_key] = {
            "panel_counts": panel_counts,
            "residual_norm_mean": residual_means,
            "isotropy_std_db": isotropy_std_db,
            "isotropy_peak_to_peak_db": isotropy_peak_to_peak_db,
            "isotropy_std_limit_db": isotropy_std_limit,
            "convergence_rms_db": convergence_rms,
            "convergence_max_abs_db": convergence_maxabs,
            "convergence_rms_limit_db": mesh_rms_limit,
            "convergence_eval_pair": finest_pair,
            "pass": pol_pass,
        }

    report["pass"] = pass_all
    return report


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run solver benchmark suite for PEC circle.")
    parser.add_argument("--radius-m", type=float, default=0.5)
    parser.add_argument("--freq-ghz", type=float, default=1.0)
    parser.add_argument("--elev-step", type=float, default=5.0)
    parser.add_argument("--mesh-levels", default="6,12,24", help="Comma-separated n-per-arc levels.")
    parser.add_argument("--pols", default="TM,TE", help="Comma-separated polarizations.")
    parser.add_argument("--json-output", default="", help="Optional output path for benchmark JSON.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    mesh_levels = [int(tok.strip()) for tok in args.mesh_levels.split(",") if tok.strip()]
    pols = [tok.strip().upper() for tok in args.pols.split(",") if tok.strip()]
    report = run_pec_circle_benchmark_suite(
        radius_m=float(args.radius_m),
        frequency_ghz=float(args.freq_ghz),
        elevations_step_deg=float(args.elev_step),
        mesh_levels=mesh_levels,
        pols=pols,
    )
    print(json.dumps(report, indent=2))
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if bool(report.get("pass", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
