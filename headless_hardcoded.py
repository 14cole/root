from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from headless_solver import _parse_sweep, run_headless
from solver_benchmarks import run_pec_circle_benchmark_suite

THIS_DIR = Path(__file__).resolve().parent


# -------------------------------
# COMMON SETTINGS (edit these)
# -------------------------------
# This section is meant for typical day-to-day runs.
COMMON: Dict[str, Any] = {
    # Input / output
    "input_geometry": "square.geo",
    "output_grim": "hardcoded_run.grim",
    "inout_units": "inches",  # inches or meters
    "polarization": "TM",  # TM / TE (HH/VV aliases also work downstream)

    # Frequency setup
    "frequency_mode": "list",  # "list" or "sweep"
    "frequency_list_ghz": [1.0, 2.0, 3.0],
    "frequency_sweep_ghz": {"start": 1.0, "stop": 10.0, "step": 0.5},

    # Azimuth setup (incoming "coming-from" angles in degrees):
    # 0 = from right, +90 = from top, -90 = from bottom
    "azimuth_mode": "list",  # "list" or "sweep"
    "azimuth_list_deg": [0.0, 45.0, 90.0, 135.0, 180.0],
    "azimuth_sweep_deg": {"start": 0.0, "stop": 180.0, "step": 2.0},
}


# -------------------------------
# ADVANCED SETTINGS (optional)
# -------------------------------
# Most users should not need to change these often.
ADVANCED: Dict[str, Any] = {
    "workers": 1,  # >1 enables multiprocessing across frequencies
    "quiet": False,
    "history": "solver=headless-hardcoded",
    "csv_output_path": "hardcoded_run.csv",  # set to None or "" to disable
    "json_summary_path": "hardcoded_run_summary.json",  # set to None or "" to disable

    # Numeric quality gate (same meaning as GUI/headless CLI)
    "quality_thresholds": {
        "residual_norm_max": 1.0e-2,
        "condition_est_max": 1.0e6,
        "warnings_max": 10,
    },
    "strict_quality_gate": False,

    # Mesh convergence gate
    "mesh_convergence": False,
    "mesh_fine_factor": 1.5,
    "mesh_rms_limit_db": 1.0,
    "mesh_max_abs_limit_db": 3.0,
    "strict_mesh_convergence": False,

    # Solver performance knobs
    "compute_condition_number": "auto",  # auto/true/false
    "parallel_elevations": True,
    "max_elevation_workers": 0,  # 0 = auto
    "reuse_angle_invariant_matrix": True,

    # Optional canonical benchmark
    "run_benchmarks": False,
    "benchmark_json_path": "hardcoded_benchmarks.json",
    "benchmark": {
        "radius_m": 0.5,
        "frequency_ghz": 1.0,
        "elevations_step_deg": 5.0,
        "mesh_levels": [6, 12, 24],
        "pols": ["TM", "TE"],
    },
}


def _build_frequency_values(common: Dict[str, Any]) -> List[float]:
    mode = str(common.get("frequency_mode", "list")).strip().lower()
    if mode == "sweep":
        sweep = dict(common.get("frequency_sweep_ghz", {}) or {})
        return _parse_sweep(
            float(sweep.get("start", 1.0)),
            float(sweep.get("stop", 10.0)),
            float(sweep.get("step", 1.0)),
            "Frequencies",
        )
    values = list(common.get("frequency_list_ghz", []) or [])
    return [float(v) for v in values]


def _build_azimuth_values(common: Dict[str, Any]) -> List[float]:
    mode = str(common.get("azimuth_mode", "list")).strip().lower()
    if mode == "sweep":
        sweep = dict(common.get("azimuth_sweep_deg", {}) or {})
        return _parse_sweep(
            float(sweep.get("start", 0.0)),
            float(sweep.get("stop", 180.0)),
            float(sweep.get("step", 1.0)),
            "Azimuths",
        )
    values = list(common.get("azimuth_list_deg", []) or [])
    return [float(v) for v in values]


def _quality_thresholds(advanced: Dict[str, Any]) -> Dict[str, float | int] | None:
    raw = dict(advanced.get("quality_thresholds", {}) or {})
    if not raw:
        return None
    return {
        "residual_norm_max": float(raw.get("residual_norm_max", 1.0e-2)),
        "condition_est_max": float(raw.get("condition_est_max", 1.0e6)),
        "warnings_max": int(raw.get("warnings_max", 10)),
    }


def _resolve_geometry_path(path_text: str) -> str:
    raw = str(path_text).strip()
    if not raw:
        raise ValueError("COMMON['input_geometry'] must be a non-empty path.")
    candidate = Path(raw)
    if candidate.is_file():
        return str(candidate)
    local = THIS_DIR / raw
    if local.is_file():
        return str(local)
    raise FileNotFoundError(
        f"Could not find geometry file '{raw}'. Checked '{candidate}' and '{local}'."
    )


def main() -> int:
    common = dict(COMMON)
    advanced = dict(ADVANCED)
    benchmark_cfg = dict(advanced.get("benchmark", {}) or {})
    cond_mode = advanced.get("compute_condition_number", "auto")
    if isinstance(cond_mode, str) and cond_mode.strip().lower() == "auto":
        cond_compute: bool | None = None
    else:
        cond_compute = bool(cond_mode)

    if bool(advanced.get("run_benchmarks", False)):
        report = run_pec_circle_benchmark_suite(
            radius_m=float(benchmark_cfg.get("radius_m", 0.5)),
            frequency_ghz=float(benchmark_cfg.get("frequency_ghz", 1.0)),
            elevations_step_deg=float(benchmark_cfg.get("elevations_step_deg", 5.0)),
            mesh_levels=[int(v) for v in benchmark_cfg.get("mesh_levels", [6, 12, 24])],
            pols=[str(v).upper() for v in benchmark_cfg.get("pols", ["TM", "TE"])],
        )
        print(json.dumps({"benchmarks": report}, indent=2))
        benchmark_path = str(advanced.get("benchmark_json_path", "")).strip()
        if benchmark_path:
            with open(benchmark_path, "w") as f:
                json.dump(report, f, indent=2)

    frequencies = _build_frequency_values(common)
    azimuths = _build_azimuth_values(common)
    geometry_path = _resolve_geometry_path(str(common["input_geometry"]))

    payload = run_headless(
        geometry_path=geometry_path,
        output_path=str(common["output_grim"]),
        frequencies_ghz=frequencies,
        elevations_deg=azimuths,
        units=str(common.get("inout_units", "inches")),
        polarization=str(common.get("polarization", "TE")),
        workers=int(advanced.get("workers", 1)),
        csv_output_path=(str(advanced.get("csv_output_path", "")).strip() or None),
        history=str(advanced.get("history", "")),
        quiet=bool(advanced.get("quiet", False)),
        quality_thresholds=_quality_thresholds(advanced),
        strict_quality_gate=bool(advanced.get("strict_quality_gate", False)),
        mesh_convergence=bool(advanced.get("mesh_convergence", False)),
        mesh_fine_factor=float(advanced.get("mesh_fine_factor", 1.5)),
        mesh_rms_limit_db=float(advanced.get("mesh_rms_limit_db", 1.0)),
        mesh_max_abs_limit_db=float(advanced.get("mesh_max_abs_limit_db", 3.0)),
        strict_mesh_convergence=bool(advanced.get("strict_mesh_convergence", False)),
        compute_condition_number=cond_compute,
        parallel_elevations=bool(advanced.get("parallel_elevations", True)),
        max_elevation_workers=max(0, int(advanced.get("max_elevation_workers", 0))),
        reuse_angle_invariant_matrix=bool(advanced.get("reuse_angle_invariant_matrix", True)),
    )

    summary = {
        "input_geometry": geometry_path,
        "output_grim": str(common["output_grim"]),
        "units": str(common.get("inout_units", "inches")),
        "polarization": str(common.get("polarization", "TE")),
        "frequency_count": len(frequencies),
        "azimuth_count": len(azimuths),
        "workers": payload["workers"],
        "sample_count": len(payload["result"].get("samples", [])),
        "grim_files": payload["grim_files"],
        "csv_file": payload["csv_file"],
        "metadata": payload["result"].get("metadata", {}),
    }
    print(json.dumps(summary, indent=2))

    summary_path = str(advanced.get("json_summary_path", "")).strip()
    if summary_path:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
