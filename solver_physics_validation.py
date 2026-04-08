from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from geometry_io import build_geometry_snapshot, parse_geometry
from rcs_solver import solve_monostatic_rcs_2d
from solver_benchmarks import run_pec_circle_benchmark_suite


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    geometry_rel: str
    reference_csv_rel: str
    units: str = "meters"
    polarization: str = "TM"


REFERENCE_CASES: List[ReferenceCase] = [
    ReferenceCase(
        name="layer_pec",
        geometry_rel="examples/magram_layer_demo/circle_pec.geo",
        reference_csv_rel="examples/magram_layer_demo/pec.csv",
    ),
    ReferenceCase(
        name="layer_ram_tuned",
        geometry_rel="examples/magram_layer_demo/circle_ram_tuned.geo",
        reference_csv_rel="examples/magram_layer_demo/ram_tuned.csv",
    ),
    ReferenceCase(
        name="layer_ram_offtuned",
        geometry_rel="examples/magram_layer_demo/circle_ram_offtuned.geo",
        reference_csv_rel="examples/magram_layer_demo/ram_offtuned.csv",
    ),
    ReferenceCase(
        name="volume_diel_ref",
        geometry_rel="examples/magram_volume_demo/circle_dielectric_reference.geo",
        reference_csv_rel="examples/magram_volume_demo/diel_ref.csv",
    ),
    ReferenceCase(
        name="volume_diel_magram",
        geometry_rel="examples/magram_volume_demo/circle_dielectric_magram.geo",
        reference_csv_rel="examples/magram_volume_demo/diel_magram.csv",
    ),
    ReferenceCase(
        name="coating_ref",
        geometry_rel="examples/magram_coating_demo/circle_coated_reference.geo",
        reference_csv_rel="examples/magram_coating_demo/coat_ref.csv",
    ),
    ReferenceCase(
        name="coating_magram",
        geometry_rel="examples/magram_coating_demo/circle_coated_magram.geo",
        reference_csv_rel="examples/magram_coating_demo/coat_magram.csv",
    ),
]


def _key(freq_ghz: float, elev_deg: float) -> Tuple[float, float]:
    return (round(float(freq_ghz), 12), round(float(elev_deg), 12))


def _parse_list(text: str, field: str) -> List[float]:
    tokens = [tok for tok in re.split(r"[,\s]+", str(text).strip()) if tok]
    if not tokens:
        raise ValueError(f"{field}: no values were provided.")
    out: List[float] = []
    for tok in tokens:
        try:
            out.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"{field}: invalid numeric token '{tok}'.") from exc
    return out


def _read_reference_csv(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None

    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frequency_ghz", "theta_scat_deg", "rcs_db"}
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Reference CSV {path} missing required columns: {joined}")
        for raw in reader:
            rows.append(
                {
                    "frequency_ghz": float(raw["frequency_ghz"]),
                    "theta_scat_deg": float(raw["theta_scat_deg"]),
                    "theta_inc_deg": float(raw.get("theta_inc_deg", raw["theta_scat_deg"])),
                    "rcs_linear": float(raw.get("rcs_linear", 0.0) or 0.0),
                    "rcs_db": float(raw["rcs_db"]),
                    "rcs_amp_real": float(raw.get("rcs_amp_real", 0.0) or 0.0),
                    "rcs_amp_imag": float(raw.get("rcs_amp_imag", 0.0) or 0.0),
                    "rcs_amp_phase_deg": float(raw.get("rcs_amp_phase_deg", 0.0) or 0.0),
                    "linear_residual": float(raw.get("linear_residual", 0.0) or 0.0),
                }
            )

    if not rows:
        raise ValueError(f"Reference CSV is empty: {path}")

    data_map: Dict[Tuple[float, float], float] = {}
    for row in rows:
        k = _key(row["frequency_ghz"], row["theta_scat_deg"])
        data_map[k] = float(row["rcs_db"])
    freqs = sorted({float(r["frequency_ghz"]) for r in rows})
    elevs = sorted({float(r["theta_scat_deg"]) for r in rows})
    return {
        "rows": rows,
        "map_db": data_map,
        "frequencies_ghz": freqs,
        "elevations_deg": elevs,
    }


def _write_reference_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    sorted_rows = sorted(
        list(rows),
        key=lambda row: (
            float(row.get("frequency_ghz", 0.0)),
            float(row.get("theta_scat_deg", 0.0)),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frequency_ghz",
                "theta_inc_deg",
                "theta_scat_deg",
                "rcs_linear",
                "rcs_db",
                "rcs_amp_real",
                "rcs_amp_imag",
                "rcs_amp_phase_deg",
                "linear_residual",
            ]
        )
        for row in sorted_rows:
            writer.writerow(
                [
                    float(row.get("frequency_ghz", 0.0)),
                    float(row.get("theta_inc_deg", row.get("theta_scat_deg", 0.0))),
                    float(row.get("theta_scat_deg", 0.0)),
                    float(row.get("rcs_linear", 0.0)),
                    float(row.get("rcs_db", 0.0)),
                    float(row.get("rcs_amp_real", 0.0)),
                    float(row.get("rcs_amp_imag", 0.0)),
                    float(row.get("rcs_amp_phase_deg", 0.0)),
                    float(row.get("linear_residual", 0.0)),
                ]
            )


def _solve_case(
    case: ReferenceCase,
    frequencies_ghz: List[float],
    elevations_deg: List[float],
    quality_thresholds: Dict[str, float | int] | None = None,
) -> Dict[str, Any]:
    geo_path = ROOT / case.geometry_rel
    with open(geo_path, "r") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
    snapshot = build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    base_dir = str(geo_path.parent)
    result = solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=frequencies_ghz,
        elevations_deg=elevations_deg,
        polarization=case.polarization,
        geometry_units=case.units,
        material_base_dir=base_dir,
        quality_thresholds=quality_thresholds,
        strict_quality_gate=False,
        compute_condition_number=True,
    )
    return result


def _sample_map_db(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[float, float], float]:
    out: Dict[Tuple[float, float], float] = {}
    for row in rows:
        out[_key(float(row.get("frequency_ghz", 0.0)), float(row.get("theta_scat_deg", 0.0)))] = float(
            row.get("rcs_db", 0.0)
        )
    return out


def _compare_case(
    case: ReferenceCase,
    ref_map: Dict[Tuple[float, float], float],
    new_map: Dict[Tuple[float, float], float],
    rms_limit_db: float,
    max_abs_limit_db: float,
    require_quality_gate: bool,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    ref_keys = set(ref_map.keys())
    new_keys = set(new_map.keys())
    missing = sorted(ref_keys - new_keys)
    extra = sorted(new_keys - ref_keys)
    common = sorted(ref_keys & new_keys)

    deltas = np.asarray([float(new_map[k] - ref_map[k]) for k in common], dtype=float) if common else np.zeros((0,))
    if len(deltas) > 0:
        rms_db = float(math.sqrt(float(np.mean(deltas * deltas))))
        max_abs_db = float(np.max(np.abs(deltas)))
        mean_bias_db = float(np.mean(deltas))
    else:
        rms_db = float("inf")
        max_abs_db = float("inf")
        mean_bias_db = float("inf")

    top_indices: List[int] = []
    if len(deltas) > 0:
        top_indices = list(np.argsort(-np.abs(deltas))[:3])
    worst_points: List[Dict[str, float]] = []
    for idx in top_indices:
        freq, elev = common[int(idx)]
        worst_points.append(
            {
                "frequency_ghz": float(freq),
                "elevation_deg": float(elev),
                "delta_db": float(deltas[int(idx)]),
            }
        )

    rms_pass = bool(math.isfinite(rms_db) and rms_db <= float(rms_limit_db))
    max_pass = bool(math.isfinite(max_abs_db) and max_abs_db <= float(max_abs_limit_db))
    shape_pass = (len(missing) == 0 and len(extra) == 0 and len(common) > 0)
    qg = dict(metadata.get("quality_gate", {}) or {})
    qg_pass = bool(qg.get("passed", True))
    if not require_quality_gate:
        qg_pass = True
    passed = bool(shape_pass and rms_pass and max_pass and qg_pass)

    fail_reasons: List[str] = []
    if not shape_pass:
        fail_reasons.append(
            f"sample_grid_mismatch(missing={len(missing)}, extra={len(extra)}, common={len(common)})"
        )
    if shape_pass and not rms_pass:
        fail_reasons.append(f"rms_db={rms_db:.6g} exceeds {float(rms_limit_db):.6g}")
    if shape_pass and not max_pass:
        fail_reasons.append(f"max_abs_db={max_abs_db:.6g} exceeds {float(max_abs_limit_db):.6g}")
    if require_quality_gate and not qg_pass:
        viol = qg.get("violations", []) or []
        fail_reasons.append("quality_gate_fail: " + ("; ".join(str(v) for v in viol) if viol else "unknown"))

    return {
        "case": case.name,
        "geometry": str(ROOT / case.geometry_rel),
        "reference_csv": str(ROOT / case.reference_csv_rel),
        "reference_available": True,
        "comparison_skipped": False,
        "units": case.units,
        "polarization": case.polarization,
        "sample_count": len(common),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "rms_db": rms_db,
        "max_abs_db": max_abs_db,
        "mean_bias_db": mean_bias_db,
        "rms_limit_db": float(rms_limit_db),
        "max_abs_limit_db": float(max_abs_limit_db),
        "worst_points": worst_points,
        "quality_gate_passed": qg_pass,
        "panel_count": int(metadata.get("panel_count", 0) or 0),
        "residual_norm_max": float(metadata.get("residual_norm_max", 0.0) or 0.0),
        "condition_est_max": float(metadata.get("condition_est_max", 0.0) or 0.0),
        "passed": passed,
        "fail_reasons": fail_reasons,
    }


def _case_report_no_reference(
    case: ReferenceCase,
    metadata: Dict[str, Any],
    sample_count: int,
    require_quality_gate: bool,
    fallback_frequencies: List[float],
    fallback_elevations: List[float],
) -> Dict[str, Any]:
    qg = dict(metadata.get("quality_gate", {}) or {})
    qg_pass = bool(qg.get("passed", True))
    if not require_quality_gate:
        qg_pass = True

    solved = int(sample_count) > 0
    passed = bool(solved and qg_pass)
    fail_reasons: List[str] = []
    if not solved:
        fail_reasons.append("no_samples_returned")
    if require_quality_gate and not qg_pass:
        viol = qg.get("violations", []) or []
        fail_reasons.append("quality_gate_fail: " + ("; ".join(str(v) for v in viol) if viol else "unknown"))

    return {
        "case": case.name,
        "geometry": str(ROOT / case.geometry_rel),
        "reference_csv": str(ROOT / case.reference_csv_rel),
        "reference_available": False,
        "comparison_skipped": True,
        "skip_reason": "reference_csv_missing",
        "units": case.units,
        "polarization": case.polarization,
        "sample_count": int(sample_count),
        "fallback_frequencies_ghz": [float(v) for v in fallback_frequencies],
        "fallback_elevations_deg": [float(v) for v in fallback_elevations],
        "rms_db": None,
        "max_abs_db": None,
        "mean_bias_db": None,
        "worst_points": [],
        "quality_gate_passed": qg_pass,
        "panel_count": int(metadata.get("panel_count", 0) or 0),
        "residual_norm_max": float(metadata.get("residual_norm_max", 0.0) or 0.0),
        "condition_est_max": float(metadata.get("condition_est_max", 0.0) or 0.0),
        "passed": passed,
        "fail_reasons": fail_reasons,
    }


def _select_cases(requested: List[str]) -> List[ReferenceCase]:
    if not requested:
        return list(REFERENCE_CASES)
    by_name = {case.name: case for case in REFERENCE_CASES}
    out: List[ReferenceCase] = []
    for name in requested:
        key = str(name).strip()
        if key not in by_name:
            valid = ", ".join(sorted(by_name.keys()))
            raise ValueError(f"Unknown case '{key}'. Valid cases: {valid}")
        out.append(by_name[key])
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-based physics validation for the 2D solver. "
            "Compares fresh solves to golden CSV curves and runs PEC-circle benchmark."
        )
    )
    parser.add_argument(
        "--rms-max-db",
        type=float,
        default=0.05,
        help="Per-case tolerance for RMS dB error vs reference CSV.",
    )
    parser.add_argument(
        "--max-abs-max-db",
        type=float,
        default=0.2,
        help="Per-case tolerance for max absolute dB error vs reference CSV.",
    )
    parser.add_argument(
        "--quality-residual-max",
        type=float,
        default=1.0e-2,
        help="Quality gate threshold for metadata.residual_norm_max during validation solves.",
    )
    parser.add_argument(
        "--quality-condition-max",
        type=float,
        default=1.0e6,
        help="Quality gate threshold for metadata.condition_est_max during validation solves.",
    )
    parser.add_argument(
        "--quality-warnings-max",
        type=int,
        default=10,
        help="Quality gate threshold for number of metadata warnings during validation solves.",
    )
    parser.add_argument(
        "--ignore-quality-gate",
        action="store_true",
        help="Do not fail reference cases when solver metadata quality gate fails.",
    )
    parser.add_argument(
        "--skip-pec-benchmark",
        action="store_true",
        help="Skip canonical PEC-circle isotropy/convergence benchmark.",
    )
    parser.add_argument(
        "--benchmark-mesh-levels",
        default="6,12,24",
        help="Mesh levels for PEC-circle benchmark.",
    )
    parser.add_argument(
        "--benchmark-elev-step",
        type=float,
        default=5.0,
        help="Elevation step (deg) for PEC-circle benchmark.",
    )
    parser.add_argument(
        "--benchmark-freq-ghz",
        type=float,
        default=1.0,
        help="Frequency (GHz) for PEC-circle benchmark.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named reference case (repeatable).",
    )
    parser.add_argument(
        "--update-references",
        action="store_true",
        help="Overwrite reference CSV files using current solver outputs before comparison.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional output path for full JSON report.",
    )
    parser.add_argument(
        "--fallback-freq-list",
        default="2,4,6",
        help="Frequencies (GHz) used when reference CSV is unavailable.",
    )
    parser.add_argument(
        "--fallback-elev-list",
        default="0,15,30,45,60,75,90,105,120,135,150,165,180",
        help="Elevations (deg) used when reference CSV is unavailable.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only final pass/fail line.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cases = _select_cases(list(args.case or []))
    quality_thresholds = {
        "residual_norm_max": float(args.quality_residual_max),
        "condition_est_max": float(args.quality_condition_max),
        "warnings_max": int(args.quality_warnings_max),
    }

    report: Dict[str, Any] = {
        "validation": "solver_physics_reference_suite",
        "cwd": os.getcwd(),
        "root": str(ROOT),
        "settings": {
            "rms_max_db": float(args.rms_max_db),
            "max_abs_max_db": float(args.max_abs_max_db),
            "quality_thresholds": quality_thresholds,
            "ignore_quality_gate": bool(args.ignore_quality_gate),
            "skip_pec_benchmark": bool(args.skip_pec_benchmark),
            "update_references": bool(args.update_references),
            "fallback_freq_list": str(args.fallback_freq_list),
            "fallback_elev_list": str(args.fallback_elev_list),
            "selected_cases": [c.name for c in cases],
        },
        "cases": [],
    }
    fallback_freqs = _parse_list(args.fallback_freq_list, "Fallback frequencies")
    fallback_elevs = _parse_list(args.fallback_elev_list, "Fallback elevations")
    if any(float(f) <= 0.0 for f in fallback_freqs):
        raise ValueError("Fallback frequencies must be positive GHz values.")

    case_pass = True
    for case in cases:
        ref_path = ROOT / case.reference_csv_rel
        ref = _read_reference_csv(ref_path)
        if ref is not None:
            solve_freqs = list(ref["frequencies_ghz"])
            solve_elevs = list(ref["elevations_deg"])
        else:
            solve_freqs = [float(v) for v in fallback_freqs]
            solve_elevs = [float(v) for v in fallback_elevs]

        result = _solve_case(
            case=case,
            frequencies_ghz=solve_freqs,
            elevations_deg=solve_elevs,
            quality_thresholds=quality_thresholds,
        )

        if args.update_references:
            _write_reference_csv(result.get("samples", []), ref_path)
            ref = _read_reference_csv(ref_path)

        metadata = dict(result.get("metadata", {}) or {})
        if ref is None:
            case_report = _case_report_no_reference(
                case=case,
                metadata=metadata,
                sample_count=len(list(result.get("samples", []) or [])),
                require_quality_gate=not bool(args.ignore_quality_gate),
                fallback_frequencies=solve_freqs,
                fallback_elevations=solve_elevs,
            )
        else:
            new_map = _sample_map_db(result.get("samples", []))
            case_report = _compare_case(
                case=case,
                ref_map=dict(ref["map_db"]),
                new_map=new_map,
                rms_limit_db=float(args.rms_max_db),
                max_abs_limit_db=float(args.max_abs_max_db),
                require_quality_gate=not bool(args.ignore_quality_gate),
                metadata=metadata,
            )
        report["cases"].append(case_report)
        case_pass = bool(case_pass and case_report.get("passed", False))

    bench_report: Dict[str, Any] = {
        "enabled": not bool(args.skip_pec_benchmark),
        "pass": True,
    }
    if not bool(args.skip_pec_benchmark):
        mesh_levels = [int(tok.strip()) for tok in str(args.benchmark_mesh_levels).split(",") if tok.strip()]
        bench = run_pec_circle_benchmark_suite(
            radius_m=0.5,
            frequency_ghz=float(args.benchmark_freq_ghz),
            elevations_step_deg=float(args.benchmark_elev_step),
            mesh_levels=mesh_levels,
            pols=["TM", "TE"],
        )
        bench_report = {
            "enabled": True,
            "pass": bool(bench.get("pass", False)),
            "report": bench,
        }
    report["pec_circle_benchmark"] = bench_report

    overall_pass = bool(case_pass and bench_report.get("pass", True))
    report["pass"] = overall_pass

    if args.json_output:
        out_path = Path(args.json_output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    if not args.quiet:
        print(json.dumps(report, indent=2))
    print("PASS" if overall_pass else "FAIL")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
