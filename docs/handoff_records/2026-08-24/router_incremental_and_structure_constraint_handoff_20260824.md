# 2026-08-24 Router incremental training and structure constraint handoff

## Scope

This compact incremental handoff preserves today's Gemini router annotations,
router training results, reconstruction scripts, and the `digitalappv4`
building-structure constraint implementation. Raw images and materialized
training datasets are intentionally excluded; restore the referenced old data
and new source images separately before rebuilding a dataset view.
No API key, GitHub token, password, or credential is included.

## Model status

| Status | Artifact | Meaning |
|---|---|---|
| `baseline` | `selected_precision_p090_classwise_epoch004_brace_balanced_v2.pth` | Current production baseline. Keep deployed. SHA256 `48486312670c2f09343254176ea79f2364e77210e8cccd2097acf5b9282c81b6`. |
| `failed` | `checkpoint_epoch_000.pth` | Representative full fine-tune failure, retained only for reproduction and analysis. **Do not deploy.** |
| `experimental` | `router_5class_incremental_balanced_shared_ft_a010_20260824.pth` | Shared-parameter interpolation candidate. Precision remains above 0.90 after calibration, but old-test recall/F1 decrease. **Do not replace the baseline without acceptance.** |

The failed full fine-tune measured `P=0.8742, R=0.6928, F1=0.7730` on the
frozen old test and `P=0.4264, R=0.1971, F1=0.2696` on the new holdout. It
improved new brace samples but regressed the old test and column-base behavior.

The experimental shared-parameter candidate measured `P=0.9008, R=0.7247,
F1=0.8032` on the old test after threshold calibration. The new holdout
aggregate improves slightly, but an individual column-base gain is not proven.

The invalid class-row-only attempt is documented in the development record but
its weights are intentionally excluded: RF-DETR rebuilt the model and changed
508 of 509 tensors, so it was not a valid frozen-parameter experiment.

## Annotations and reconstruction

- `annotations/gemini_router_annotations_20260824.json` is the single annotation
  artifact. It contains the accepted Gemini records, excluded records, summary,
  model identity, and source-relative image names.
- No image binary and no materialized train/valid/test dataset is included.
- `source/Shimizu-2026/systems/rfdetr/scripts/` contains the incremental-view,
  oversampling, training, threshold-search, evaluation, grafting, and checkpoint
  interpolation utilities used today.
- `source/Shimizu-2026/systems/gemini/scripts/` contains the annotation-batch,
  retry, merge, and Gemini annotation utilities.
- Dataset reconstruction requires the earlier reviewed baseline dataset and the
  separately retained source images referenced by `image_rel_path`.

## Application pipeline

The `digitalappv4` source snapshot contains:

- building structure values `rc`, `steel`, and `unknown`;
- mobile/admin structure selection and API propagation;
- Phase 1 deterministic allowed-class filtering, enabled by default;
- suppressed-result audit output;
- reversible Phase 2 raw-logit masking, enabled only with
  `SHIMIZU_STRUCTURE_CONSTRAINT_MODE=constrained_logits`;
- evaluation scripts, tests, meeting questions, summaries, and comparison JPGs.

Phase 2 is intentionally off by default. Current holdout results do not justify
making it the production default.

## Restore and verification

Extract the archive from `/workspace`, then verify all files with:

```bash
sha256sum -c SHA256SUMS.txt
```

Use the baseline checkpoint and its manifest/configuration for production.
Failed and experimental weights exist to reproduce conclusions, not as release
candidates.

The earlier 5 GB self-contained archive is superseded by the compact incremental
archive `shimizu_20260824_router_incremental_compact_handoff.tar.zst`. The large
archive duplicated historical JPEG data and is not the routine download target.
