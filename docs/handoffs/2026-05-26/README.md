# Shimizu 20260526 Handoff

This directory stores the 2026-05-26 recovery artifacts created before destroying the current server instance.

## Archives

The following local archives are expected here:

```text
handoff_20260526/shimizu_20260526_router_best_and_eval.tar.zst
handoff_20260526/shimizu_20260526_synthetic_router_final_300x4.tar.zst
handoff_20260526/SHA256SUMS.txt
```

## What Is Stored

### Router best-and-eval archive

This archive keeps only the current best router training result and the materials needed to review or resume from it:

- best router training run summary files
- selected checkpoint files for the current best router
- current router test outputs
- before/after fallback evaluation outputs
- 2026-05-26 report assets and analysis docs

### Synthetic final 300x4 archive

This archive keeps only the final Gemini synthetic dataset used at the end of the session:

- `outputs/synthetic_router_pipeline_nb2_promptgen_300x4_c10`

Intermediate synthetic experiments such as `20x4`, `100x4`, `400x4`, concurrency benchmarks, and dry runs are intentionally excluded.

## Restore

From the repo root:

```bash
tar --zstd -xf handoff_20260526/shimizu_20260526_router_best_and_eval.tar.zst -C .
tar --zstd -xf handoff_20260526/shimizu_20260526_synthetic_router_final_300x4.tar.zst -C .
sha256sum -c handoff_20260526/SHA256SUMS.txt
```

For full recovery notes, see:

```text
docs/final_handoff_20260526.md
```
