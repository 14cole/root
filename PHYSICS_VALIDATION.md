# 2D Solver Physics Validation

Use `solver_physics_validation.py` to run reference-based regression checks:

- Re-solves canonical `.geo` cases (layer, volume, coating examples).
- Compares solved `rcs_db` values against stored reference CSV curves.
- Enforces tolerance-based pass/fail (`RMS dB` and `max abs dB` limits).
- Runs the PEC-circle benchmark suite (isotropy + mesh convergence).

If a reference CSV is missing in deployment, that case no longer crashes:

- The case still solves using fallback sweeps.
- Direct curve-comparison metrics are marked as skipped for that case.
- Quality gate checks still apply to the solve.

## Quick Start

```bash
python3 2dsolver/solver_physics_validation.py
```

Or via the unified entrypoint:

```bash
python3 2dsolver/main.py --validate-physics
```

Exit code:

- `0`: all checks passed
- `2`: at least one check failed

## Useful Options

```bash
python3 2dsolver/solver_physics_validation.py \
  --rms-max-db 0.05 \
  --max-abs-max-db 0.2 \
  --json-output /tmp/physics_validation_report.json
```

Fallback sweeps for missing CSV references:

```bash
python3 2dsolver/solver_physics_validation.py \
  --fallback-freq-list "2,4,6" \
  --fallback-elev-list "0,15,30,45,60,75,90,105,120,135,150,165,180"
```

Run one case only:

```bash
python3 2dsolver/solver_physics_validation.py --case layer_ram_tuned
```

Refresh reference CSVs from current solver output:

```bash
python3 2dsolver/solver_physics_validation.py --update-references
```
