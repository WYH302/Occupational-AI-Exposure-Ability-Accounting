"""Targeted check of the manuscript's FRS mapping definition against its pinned archive.

Downloads public inputs to a separate evidence directory; never replaces historical
inputs or results. This is not an end-to-end replication of the outcome analyses.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.two_layer_accounting import _frs_ability_scores, build_frs_replication_accounting

COMMIT = "adca5fc2cd0e9a659ff05278b7fa7a53f4f324c1"
FILES = {
    "Input/mturk_mapping_matrix.dta": "frs_mturk_mapping_matrix.dta",
    "Input/application_list.dta": "frs_application_list.dta",
    "Input/abilities_2020.dta": "frs_abilities_2020.dta",
    "Input/AIOE, AIIE, and AIGE Construction.do": "frs_aioe_construction.do",
    "AIOE_DataAppendix.xlsx": "AIOE_DataAppendix.xlsx",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/AIOE-Data/AIOE/git/trees/{COMMIT}?recursive=1"
    with urllib.request.urlopen(tree_url, timeout=45) as response:
        tree = {item["path"]: item for item in json.load(response)["tree"]}
    records = []
    for remote, local in FILES.items():
        url = f"https://raw.githubusercontent.com/AIOE-Data/AIOE/{COMMIT}/{urllib.parse.quote(remote)}"
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
        git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert git_sha == tree[remote]["sha"], remote
        (args.evidence_dir / local).write_bytes(data)
        records.append({"file": local, "url": url, "git_blob_sha1": git_sha,
                        "sha256": hashlib.sha256(data).hexdigest()})
    mapping = pd.read_stata(args.evidence_dir / "frs_mturk_mapping_matrix.dta")
    applications = pd.read_stata(args.evidence_dir / "frs_application_list.dta")
    cols = [c for c in mapping if c not in {"application_id", "applications"}]
    matrix = mapping[cols].to_numpy(dtype=float)
    assert matrix.shape == (10, 52)
    assert np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix <= 1).all()
    fractional = (matrix > 0) & (matrix < 1)
    assert fractional.any()
    assert (applications["binary_all"] == 1).all()
    scores = _frs_ability_scores(applications, mapping)
    np.testing.assert_allclose(scores["ability_score"], matrix.sum(axis=0), rtol=1e-6, atol=1e-7)
    ledger, occupations = build_frs_replication_accounting(args.evidence_dir)
    sums = ledger.groupby("soc")["standardized_contribution"].sum()
    rebuilt = occupations.set_index("soc")["standardized_score"]
    identity = float((sums - rebuilt).abs().max())
    residual = occupations["published_residual"]
    r = float(occupations["standardized_score"].corr(occupations["published_aioe"]))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    assert len(occupations) == 774 and len(ledger) == 40248
    assert identity < 3e-15
    assert round(r, 6) == 0.999994 and round(rmse, 5) == 0.00359
    assert round(float(residual.abs().max()), 5) == 0.03793
    report = {
        "scope": "Fresh targeted reconstruction from the cited immutable public commit, not recovery of missing historical inputs or a full experimental rerun",
        "commit": COMMIT, "inputs": records,
        "implementation_sha256": hashlib.sha256((Path(__file__).parent / "two_layer_accounting.py").read_bytes()).hexdigest(),
        "matrix_shape": list(matrix.shape), "mapping_min": float(matrix.min()),
        "mapping_max": float(matrix.max()), "fractional_entries": int(fractional.sum()),
        "zero_entries": int((matrix == 0).sum()), "one_entries": int((matrix == 1).sum()),
        "application_binary_all": applications["binary_all"].tolist(),
        "ability_score_min": float(scores.ability_score.min()),
        "ability_score_max": float(scores.ability_score.max()),
        "occupations": len(occupations), "ledger_rows": len(ledger),
        "max_identity_error": identity, "published_pearson_r": r,
        "published_rmse": rmse, "max_published_residual": float(residual.abs().max()),
        "conclusion": "The implementation preserves fractional relatedness; the binary mapping description is incorrect. Targeted reconstruction matches reported accounting/reconciliation at display precision."
    }
    (args.evidence_dir / "mapping_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
