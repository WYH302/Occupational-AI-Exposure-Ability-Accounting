from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
RETAINED_UNUSED_INPUTS = {
    "O*NET 30.2 abilities",
    "O*NET 30.2 job zones",
    "O*NET 30.2 education",
    "IPUMS CPS extract",
}
RESTRICTED_INPUT_LOCATORS = {
    "IPUMS USA ACS extract": "ipums://usa/acs-extract.csv",
    "IPUMS CPS extract": "ipums://cps/cps-extract.csv",
}


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def find_one(pattern: str, root: Path = RAW_DIR) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one input matching {pattern!r}; found {matches}")
    return matches[0]


def input_files() -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = [
        ("FRS ability and occupation exposure", "Felten--Raj--Seamans appendix", RAW_DIR / "AIOE_DataAppendix.xlsx"),
        ("FRS archived ability ratings", "Exact-accounting importance and level inputs", RAW_DIR / "frs_abilities_2020.dta"),
        ("FRS archived application list", "Exact-accounting AI application indicator", RAW_DIR / "frs_application_list.dta"),
        ("FRS archived application mapping", "Exact-accounting application--ability mapping", RAW_DIR / "frs_mturk_mapping_matrix.dta"),
        ("FRS archived occupation titles", "Archived reconstruction occupation labels", RAW_DIR / "frs_occ_title_2020.dta"),
        ("FRS archived construction code", "Authoritative aggregation and standardization specification", RAW_DIR / "frs_aioe_construction.do"),
        ("O*NET 24.3 abilities", "Frozen-origin version-drift baseline", RAW_DIR / "db_24_3_excel" / "db_24_3_excel" / "Abilities.xlsx"),
        ("O*NET 27.1 abilities", "Primary November 2022 measurement snapshot", RAW_DIR / "db_27_1_excel" / "Abilities.xlsx"),
        ("O*NET 27.1 job zones", "Primary occupation control", RAW_DIR / "db_27_1_excel" / "Job Zones.xlsx"),
        ("O*NET 27.1 education", "Primary occupation control", RAW_DIR / "db_27_1_excel" / "Education, Training, and Experience.xlsx"),
        ("O*NET 30.2 abilities", "February 2026 version sensitivity", RAW_DIR / "db_30_2_excel" / "Abilities.xlsx"),
        ("O*NET 30.2 job zones", "February 2026 version sensitivity", RAW_DIR / "db_30_2_excel" / "Job Zones.xlsx"),
        ("O*NET 30.2 education", "February 2026 version sensitivity", RAW_DIR / "db_30_2_excel" / "Education, Training, and Experience.xlsx"),
        ("O*NET 30.3 abilities", "May 2026 version sensitivity", RAW_DIR / "db_30_3_excel" / "Abilities.xlsx"),
        ("O*NET 30.3 job zones", "May 2026 version sensitivity", RAW_DIR / "db_30_3_excel" / "Job Zones.xlsx"),
        ("O*NET 30.3 education", "May 2026 version sensitivity", RAW_DIR / "db_30_3_excel" / "Education.xlsx"),
        ("Census occupation crosswalk", "Exact Census-to-SOC mapping", EXTERNAL_DIR / "crosswalks" / "2018-occupation-code-list-and-crosswalk.xlsx"),
        ("IPUMS USA ACS extract", "Restricted-redistribution annual-earnings microdata", find_one("2.usa_00001.csv")),
        ("IPUMS CPS extract", "Restricted-redistribution secondary employment microdata", find_one("1.cps_00001.csv")),
    ]
    for year in range(2019, 2025):
        files.append(
            (
                f"OEWS May {year}",
                "Auxiliary published national estimate",
                RAW_DIR / f"oesm{str(year)[2:]}nat" / f"national_M{year}_dl.xlsx",
            )
        )
    return files


def main() -> None:
    rows = []
    for label, role, path in input_files():
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "Input": label,
                "Analysis status": (
                    "Retained legacy input; not used in final two-layer results"
                    if label in RETAINED_UNUSED_INPUTS
                    else "Used in final two-layer analysis"
                ),
                "Role": role,
                "Input locator": RESTRICTED_INPUT_LOCATORS.get(
                    label, path.relative_to(ROOT).as_posix()
                ),
                "Bytes": path.stat().st_size,
                "SHA-256": sha256(path),
            }
        )
        print(f"hashed {label}: {path.stat().st_size:,} bytes")
    output = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    output.to_csv(ROOT / "results" / "input_version_manifest.csv", index=False)


if __name__ == "__main__":
    main()
