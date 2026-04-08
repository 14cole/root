from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

from geometry_io import build_geometry_snapshot, parse_geometry
from grim_io import export_result_to_grim
from rcs_solver import evaluate_quality_gate, solve_monostatic_rcs_2d
from solver_quality import evaluate_mesh_convergence, scale_snapshot_panel_density


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


def _parse_sweep(start: float, stop: float, step: float, field_name: str) -> List[float]:
    step_abs = abs(float(step))
    if step_abs <= 0.0:
        raise ValueError(f"{field_name}: step must be > 0.")
    direction = 1.0 if stop >= start else -1.0
    signed = step_abs * direction
    values: List[float] = []
    current = float(start)
    for _ in range(200_000):
        if direction > 0 and current > stop + 1e-9:
            break
        if direction < 0 and current < stop - 1e-9:
            break
        values.append(round(current, 12))
        current += signed
    if not values or abs(values[-1] - stop) > 1e-9:
        values.append(float(stop))
    return values


def _sorted_samples(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(samples),
        key=lambda row: (
            float(row.get("frequency_ghz", 0.0)),
            float(row.get("theta_scat_deg", 0.0)),
        ),
    )


def _write_csv(samples: List[Dict[str, Any]], path: str) -> str:
    rows = _sorted_samples(samples)
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
        for row in rows:
            writer.writerow(
                [
                    float(row.get("frequency_ghz", 0.0)),
                    float(row.get("theta_inc_deg", 0.0)),
                    float(row.get("theta_scat_deg", 0.0)),
                    float(row.get("rcs_linear", 0.0)),
                    float(row.get("rcs_db", 0.0)),
                    float(row.get("rcs_amp_real", 0.0)),
                    float(row.get("rcs_amp_imag", 0.0)),
                    float(row.get("rcs_amp_phase_deg", 0.0)),
                    float(row.get("linear_residual", 0.0)),
                ]
            )
    return os.path.abspath(path)


def _solve_one_frequency(
    snapshot: Dict[str, Any],
    base_dir: str,
    freq_ghz: float,
    elevations_deg: List[float],
    polarization: str,
    units: str,
    quality_thresholds: Dict[str, float | int] | None,
    strict_quality_gate: bool,
    compute_condition_number: bool,
    parallel_elevations: bool,
    max_elevation_workers: int,
    reuse_angle_invariant_matrix: bool,
) -> Dict[str, Any]:
    return solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(freq_ghz)],
        elevations_deg=elevations_deg,
        polarization=polarization,
        geometry_units=units,
        material_base_dir=base_dir,
        quality_thresholds=quality_thresholds,
        strict_quality_gate=strict_quality_gate,
        compute_condition_number=compute_condition_number,
        parallel_elevations=parallel_elevations,
        max_elevation_workers=max_elevation_workers,
        reuse_angle_invariant_matrix=reuse_angle_invariant_matrix,
    )


def _print(msg: str, quiet: bool) -> None:
    if quiet:
        return
    print(msg, file=sys.stderr, flush=True)


def run_headless(
    geometry_path: str,
    output_path: str,
    frequencies_ghz: List[float],
    elevations_deg: List[float],
    units: str = "inches",
    polarization: str = "TE",
    workers: int = 1,
    csv_output_path: str | None = None,
    history: str = "",
    quiet: bool = False,
    quality_thresholds: Dict[str, float | int] | None = None,
    strict_quality_gate: bool = False,
    mesh_convergence: bool = False,
    mesh_fine_factor: float = 1.5,
    mesh_rms_limit_db: float = 1.0,
    mesh_max_abs_limit_db: float = 3.0,
    strict_mesh_convergence: bool = False,
    compute_condition_number: bool | None = None,
    parallel_elevations: bool = True,
    max_elevation_workers: int = 0,
    reuse_angle_invariant_matrix: bool = True,
) -> Dict[str, Any]:
    if any(f <= 0 for f in frequencies_ghz):
        raise ValueError("Frequencies must be positive GHz values.")
    if not elevations_deg:
        raise ValueError("At least one elevation angle is required.")
    if mesh_convergence and float(mesh_fine_factor) <= 1.0:
        raise ValueError("Mesh convergence requires --mesh-fine-factor > 1.0.")
    compute_cond = bool(strict_quality_gate) if compute_condition_number is None else bool(compute_condition_number)

    geo_path_abs = os.path.abspath(geometry_path)
    with open(geo_path_abs, "r") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
    snapshot = build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    base_dir = os.path.dirname(geo_path_abs)

    workers = max(1, int(workers))
    results_by_freq: Dict[float, Dict[str, Any]] = {}
    
    def run_serial_result() -> Dict[str, Any]:
        last_pct = {"value": -1}

        def cb(done: int, total: int, message: str) -> None:
            if quiet:
                return
            pct = int(round(100.0 * float(done) / float(total))) if total > 0 else 0
            pct = max(0, min(100, pct))
            if pct == last_pct["value"] and pct not in {0, 100}:
                return
            last_pct["value"] = pct
            _print(f"[{pct:3d}%] {message}", quiet=False)

        return solve_monostatic_rcs_2d(
            geometry_snapshot=snapshot,
            frequencies_ghz=frequencies_ghz,
            elevations_deg=elevations_deg,
            polarization=polarization,
            geometry_units=units,
            material_base_dir=base_dir,
            progress_callback=cb if not quiet else None,
            quality_thresholds=quality_thresholds,
            strict_quality_gate=strict_quality_gate,
            compute_condition_number=compute_cond,
            parallel_elevations=parallel_elevations,
            max_elevation_workers=max_elevation_workers,
            reuse_angle_invariant_matrix=reuse_angle_invariant_matrix,
        )

    if workers == 1 or len(frequencies_ghz) == 1:
        result = run_serial_result()
    else:
        _print(
            (
                f"Launching multiprocessing: workers={workers}, "
                f"frequency_tasks={len(frequencies_ghz)}. "
                f"Consider setting OMP_NUM_THREADS=1 to avoid oversubscription."
            ),
            quiet=quiet,
        )
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(
                        _solve_one_frequency,
                        snapshot,
                        base_dir,
                        float(freq),
                        elevations_deg,
                        polarization,
                        units,
                        quality_thresholds,
                        strict_quality_gate,
                        compute_cond,
                        parallel_elevations,
                        max_elevation_workers,
                        reuse_angle_invariant_matrix,
                    ): float(freq)
                    for freq in frequencies_ghz
                }
                done = 0
                total = len(futures)
                for future in as_completed(futures):
                    freq = futures[future]
                    results_by_freq[freq] = future.result()
                    done += 1
                    _print(f"[{done}/{total}] Solved {freq:g} GHz", quiet=quiet)
        except (OSError, PermissionError) as exc:
            _print(
                f"Multiprocessing unavailable ({type(exc).__name__}: {exc}); falling back to serial solve.",
                quiet=quiet,
            )
            result = run_serial_result()
            workers = 1
            results_by_freq = {}

        if results_by_freq:
            samples: List[Dict[str, Any]] = []
            template: Dict[str, Any] | None = None
            for freq in sorted(results_by_freq.keys()):
                res = results_by_freq[freq]
                if template is None:
                    template = res
                samples.extend(res.get("samples", []))
            if template is None:
                raise RuntimeError("No results produced by multiprocessing run.")
            result = {
                "title": template.get("title", snapshot.get("title", "Geometry")),
                "scattering_mode": template.get("scattering_mode", "monostatic"),
                "polarization": template.get("polarization", "VV"),
                "solver_polarization": template.get("solver_polarization", polarization),
                "samples": _sorted_samples(samples),
                "metadata": {
                    **(template.get("metadata", {}) or {}),
                    "frequency_count": len(frequencies_ghz),
                    "elevation_count": len(elevations_deg),
                    "workers": workers,
                    "mode": "headless-multiprocess",
                },
            }
            md = result.get("metadata", {}) or {}
            residual_max = 0.0
            residual_means: List[float] = []
            cond_max = 0.0
            cond_means: List[float] = []
            warning_union: List[str] = []
            seen_warn: set[str] = set()
            for freq in sorted(results_by_freq.keys()):
                rmd = (results_by_freq[freq].get("metadata", {}) or {})
                residual_max = max(residual_max, float(rmd.get("residual_norm_max", 0.0) or 0.0))
                cond_max = max(cond_max, float(rmd.get("condition_est_max", 0.0) or 0.0))
                residual_means.append(float(rmd.get("residual_norm_mean", 0.0) or 0.0))
                cond_means.append(float(rmd.get("condition_est_mean", 0.0) or 0.0))
                for msg in list(rmd.get("warnings", []) or []):
                    text = str(msg)
                    if text in seen_warn:
                        continue
                    seen_warn.add(text)
                    warning_union.append(text)
            md["residual_norm_max"] = float(residual_max)
            md["residual_norm_mean"] = (
                float(sum(residual_means) / len(residual_means)) if residual_means else 0.0
            )
            md["condition_est_max"] = float(cond_max)
            md["condition_est_mean"] = (
                float(sum(cond_means) / len(cond_means)) if cond_means else 0.0
            )
            md["warnings"] = warning_union
            md["quality_gate"] = evaluate_quality_gate(md, thresholds=quality_thresholds)
            result["metadata"] = md
            if strict_quality_gate and not bool(md["quality_gate"].get("passed", False)):
                msg = "; ".join(md["quality_gate"].get("violations", []) or ["quality gate failed"])
                raise ValueError(f"Quality gate failed: {msg}")

    if mesh_convergence:
        _print(
            (
                "Running mesh convergence check "
                f"(fine_factor={float(mesh_fine_factor):.3g}, "
                f"rms_limit_db={float(mesh_rms_limit_db):.3g}, "
                f"max_abs_limit_db={float(mesh_max_abs_limit_db):.3g})..."
            ),
            quiet=quiet,
        )
        fine_snapshot = scale_snapshot_panel_density(snapshot, float(mesh_fine_factor))

        fine_last_pct = {"value": -1}

        def fine_cb(done: int, total: int, message: str) -> None:
            if quiet:
                return
            pct = int(round(100.0 * float(done) / float(total))) if total > 0 else 0
            pct = max(0, min(100, pct))
            if pct == fine_last_pct["value"] and pct not in {0, 100}:
                return
            fine_last_pct["value"] = pct
            _print(f"[mesh {pct:3d}%] {message}", quiet=False)

        fine_result = solve_monostatic_rcs_2d(
            geometry_snapshot=fine_snapshot,
            frequencies_ghz=frequencies_ghz,
            elevations_deg=elevations_deg,
            polarization=polarization,
            geometry_units=units,
            material_base_dir=base_dir,
            progress_callback=fine_cb if not quiet else None,
            quality_thresholds=quality_thresholds,
            strict_quality_gate=False,
            compute_condition_number=False,
            parallel_elevations=parallel_elevations,
            max_elevation_workers=max_elevation_workers,
            reuse_angle_invariant_matrix=reuse_angle_invariant_matrix,
        )
        mesh_gate = evaluate_mesh_convergence(
            base_result=result,
            fine_result=fine_result,
            rms_limit_db=float(mesh_rms_limit_db),
            max_abs_limit_db=float(mesh_max_abs_limit_db),
        )
        mesh_gate["fine_factor"] = float(mesh_fine_factor)
        md = result.get("metadata", {}) or {}
        md["mesh_convergence"] = mesh_gate
        result["metadata"] = md

        if strict_mesh_convergence and not bool(mesh_gate.get("passed", False)):
            reason = str(mesh_gate.get("reason", "") or "mesh convergence failed")
            raise ValueError(f"Mesh convergence gate failed: {reason}")

    grim_files = export_result_to_grim(
        result,
        output_path,
        polarization=result.get("polarization", "VV"),
        source_path=geo_path_abs,
        history=history or "solver=headless",
    )

    csv_path_abs = None
    if csv_output_path:
        csv_path_abs = _write_csv(result.get("samples", []), csv_output_path)

    payload = {
        "result": result,
        "grim_files": grim_files,
        "csv_file": csv_path_abs,
        "geometry_path": geo_path_abs,
        "workers": workers,
    }
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless 2D monostatic BIE/MoM solver runner (HPC/CLI friendly)."
    )
    parser.add_argument("--geometry", required=True, help="Path to input .geo geometry file.")
    parser.add_argument("--output", required=True, help="Output .grim path (extension auto-added if missing).")
    parser.add_argument(
        "--units",
        default="inches",
        choices=["inches", "meters", "inch", "meter", "in", "m"],
        help="Units of geometry coordinates.",
    )
    parser.add_argument(
        "--pol",
        "--polarization",
        default="TE",
        dest="polarization",
        choices=["TE", "TM", "VV", "HH", "V", "H", "vertical", "horizontal"],
        help="Polarization to solve.",
    )

    freq_group = parser.add_mutually_exclusive_group(required=True)
    freq_group.add_argument(
        "--freq-list",
        help="Comma/space-separated frequencies in GHz, e.g. '1,2,3.5'.",
    )
    freq_group.add_argument(
        "--freq-sweep",
        nargs=3,
        type=float,
        metavar=("START_GHZ", "STOP_GHZ", "STEP_GHZ"),
        help="Frequency sweep start stop step in GHz.",
    )

    elev_group = parser.add_mutually_exclusive_group(required=True)
    elev_group.add_argument(
        "--elev-list",
        help="Comma/space-separated elevations in deg, e.g. '0,30,60,90'.",
    )
    elev_group.add_argument(
        "--elev-sweep",
        nargs=3,
        type=float,
        metavar=("START_DEG", "STOP_DEG", "STEP_DEG"),
        help="Elevation sweep start stop step in deg.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of processes across frequencies. "
            "Use 1 for serial. Higher values can help for large frequency sets."
        ),
    )
    parser.add_argument("--csv-output", default="", help="Optional CSV output path for sample table.")
    parser.add_argument("--history", default="", help="Optional history string stored in .grim metadata.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Fail the run if numeric quality gate thresholds are exceeded.",
    )
    parser.add_argument(
        "--quality-residual-max",
        type=float,
        default=1.0e-2,
        help="Quality gate threshold for metadata.residual_norm_max.",
    )
    parser.add_argument(
        "--quality-condition-max",
        type=float,
        default=1.0e6,
        help="Quality gate threshold for metadata.condition_est_max.",
    )
    parser.add_argument(
        "--quality-warnings-max",
        type=int,
        default=10,
        help="Quality gate threshold for number of metadata warnings.",
    )
    parser.add_argument(
        "--compute-condition-number",
        action="store_true",
        help="Force condition-number computation (auto-enabled by --strict-quality).",
    )
    parser.add_argument(
        "--no-parallel-elevations",
        action="store_true",
        help="Disable per-elevation parallel execution inside each frequency solve.",
    )
    parser.add_argument(
        "--max-elevation-workers",
        type=int,
        default=0,
        help="Max workers for per-elevation parallelism (0 = auto).",
    )
    parser.add_argument(
        "--no-matrix-reuse",
        action="store_true",
        help="Disable angle-invariant matrix/factorization reuse optimization.",
    )
    parser.add_argument(
        "--json-summary",
        default="",
        help="Optional JSON summary path with metadata and output file paths.",
    )
    parser.add_argument(
        "--mesh-convergence",
        action="store_true",
        help="Run a fine-mesh solve and compare dB deltas against the base solve.",
    )
    parser.add_argument(
        "--strict-mesh-convergence",
        action="store_true",
        help="Fail the run if mesh convergence criteria are not met.",
    )
    parser.add_argument(
        "--mesh-fine-factor",
        type=float,
        default=1.5,
        help="Multiplier applied to geometry n-property for fine-mesh convergence solve.",
    )
    parser.add_argument(
        "--mesh-rms-max-db",
        type=float,
        default=1.0,
        help="Mesh convergence threshold for RMS(|base_dB - fine_dB|).",
    )
    parser.add_argument(
        "--mesh-max-abs-max-db",
        type=float,
        default=3.0,
        help="Mesh convergence threshold for max abs dB delta.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.freq_list:
        freqs = _parse_list(args.freq_list, "Frequencies")
    else:
        f_start, f_stop, f_step = args.freq_sweep
        freqs = _parse_sweep(f_start, f_stop, f_step, "Frequencies")

    if args.elev_list:
        elevs = _parse_list(args.elev_list, "Elevations")
    else:
        e_start, e_stop, e_step = args.elev_sweep
        elevs = _parse_sweep(e_start, e_stop, e_step, "Elevations")

    units = args.units.lower()
    if units in {"inch", "in"}:
        units = "inches"
    if units in {"meter", "m"}:
        units = "meters"

    pol = str(args.polarization).strip().upper()
    if pol == "V":
        pol = "VV"
    if pol == "H":
        pol = "HH"
    if pol == "VERTICAL":
        pol = "TE"
    if pol == "HORIZONTAL":
        pol = "TM"

    payload = run_headless(
        geometry_path=args.geometry,
        output_path=args.output,
        frequencies_ghz=freqs,
        elevations_deg=elevs,
        units=units,
        polarization=pol,
        workers=max(1, int(args.workers)),
        csv_output_path=(args.csv_output.strip() or None),
        history=args.history,
        quiet=bool(args.quiet),
        quality_thresholds={
            "residual_norm_max": float(args.quality_residual_max),
            "condition_est_max": float(args.quality_condition_max),
            "warnings_max": int(args.quality_warnings_max),
        },
        strict_quality_gate=bool(args.strict_quality),
        mesh_convergence=bool(args.mesh_convergence),
        mesh_fine_factor=float(args.mesh_fine_factor),
        mesh_rms_limit_db=float(args.mesh_rms_max_db),
        mesh_max_abs_limit_db=float(args.mesh_max_abs_max_db),
        strict_mesh_convergence=bool(args.strict_mesh_convergence),
        compute_condition_number=(True if bool(args.compute_condition_number) else None),
        parallel_elevations=not bool(args.no_parallel_elevations),
        max_elevation_workers=max(0, int(args.max_elevation_workers)),
        reuse_angle_invariant_matrix=not bool(args.no_matrix_reuse),
    )

    summary = {
        "geometry_path": payload["geometry_path"],
        "workers": payload["workers"],
        "sample_count": len(payload["result"].get("samples", [])),
        "grim_files": payload["grim_files"],
        "csv_file": payload["csv_file"],
        "metadata": payload["result"].get("metadata", {}),
    }

    if args.json_summary:
        with open(args.json_summary, "w") as f:
            json.dump(summary, f, indent=2)

    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
