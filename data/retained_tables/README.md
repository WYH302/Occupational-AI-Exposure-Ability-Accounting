# Recovered historical evidence subset

This directory preserves the machine-readable result files that were recoverable from the deep historical workspace snapshot dated 2026-07-27. CSV rows and values match the source snapshot; line endings were normalized when the files were restored. `PROVENANCE.csv` records both the original-source SHA-256 and the restored-file SHA-256 so that this normalization is explicit.

Status: **historical evidence, not a newly rerun result tree**. These files support numerical cross-checking of the displayed pair, version-drift, and external-boundary tables. They do not replace the missing full `results/` directory, raw/licensed inputs, input-version manifest, coverage report, or end-to-end regeneration scripts. Consequently, the complete experiment remains `CANNOT_VERIFY` in the current workspace even though this subset is internally consistent with the archived 2026-07-28 audit.

No values in this directory were inferred or recomputed during recovery.

`tables/table_profile_caliper_sensitivity_display.csv` is a display-precision recovery rather than a full-precision machine result. Its six rows were transcribed and independently checked against Supplementary Table S5 on `__恢复附件_可视化/019f6eac-c887-7b32-bedd-3e148e1cdc2d/supp_acceptance_qa_20260728/supp-06.png` (SHA-256 `cbcb372ef02d9bb1197d3c342df8955c6aae827914a62a7b23e3b5c941e9e431`) and against the formatting schema in `analysis/build_two_layer_assets.py`. The unavailable unrounded all-eligible values were not reconstructed.
