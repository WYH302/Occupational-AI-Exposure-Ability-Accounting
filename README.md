# Occupational-AI-Exposure-Ability-Accounting

Research code, axiomatic decomposition utilities, retained benchmark tables, and interactive capability-balance-sheet explorer for **Similar AI Exposure Scores, Different Ability Contributions: Evidence from U.S. Occupations** (*Uncovering the Iso-Exposure Illusion via Signed Capability Accounting*).

**Repository maintainer:** Yonghao Wu ([WYH302](https://github.com/WYH302))

---

## Core Conceptual Framework

Standard composite indices of artificial intelligence (AI) exposure project high-dimensional occupational skill portfolios onto a single scalar axis. Because linear aggregation is fully compensatory, scalar rankings create an **iso-exposure illusion**: occupations residing on the same iso-exposure contour share virtually identical headline scores while facing fundamentally different technological pressures in 52-dimensional skill space.

1. **Signed Capability Balance Sheet ($Z_o = P_o - N_o = \sum_{a=1}^{52} C_{oa}$)**:
   Under locality (ability separability), zero-baseline consistency, and degree-one homogeneity, the unique local additive allocation rule $C_{oa} = v_{oa}(A_a - \mu_R)/\sigma_R$ decomposes each occupation's standardized exposure score into countervailing structural forces:
   - **Exposure-Amplifying Capabilities ($P_o$, $A_a > \mu_R$)**: Cognitive, analytical, and language-intensive abilities that pull occupational exposure upward;
   - **Exposure-Buffering Capabilities ($N_o$, $A_a < \mu_R$)**: Manual dexterity, psychomotor, and physical-coordination bottlenecks where low AI applicability insulates the occupation and pulls net exposure downward.
2. **Chronological Occupational Restructuring Yardstick**:
   Across **983 cross-major occupation pairs** differing by at most $0.025$ standard deviations on both published and reconstructed Felten–Raj–Seamans (AIOE) scores, the median base-2 Jensen–Shannon profile distance is **0.1731** ($P_{90} = 0.2612$). Calibrated against historical within-occupation skill updates across O*NET releases 24.3–27.1 ($N = 152$) and 24.3–30.3 ($N = 226$), **67.7%** and **61.3%** of near-score pairs diverge more in their underlying capability portfolios than the 90th percentile of occupations undergoing multi-year taxonomy revisions.
3. **Bifurcated Adjustment Regime**:
   Linking occupational capability profiles to American Community Survey (ACS, $N = 318$) and Occupational Employment and Wage Statistics (OEWS, $N = 648$) panels reveals a functional division: scalar exposure governs the *between-occupation pecuniary margin* (short-run equilibrium wage sorting), whereas the 52-dimensional capability balance sheet governs the *within-occupation structural margin* (workflow redesign and targeted reskilling).

---

## Start Here: Zero External Dependencies

Use Python 3.10 or later. The following three commands run out of the box using **only the Python standard library** (no third-party packages, GPU, or API keys required):

```bash
# 1. Run the automated unit test suite (verifies SHA-256 provenance, axiomatic identities, and JSD metric properties)
python -m unittest discover -s tests -v

# 2. Verify all retained CSV tables and generate analysis/reproduced/verification_report.json
python analysis/analyze_retained_bundle.py

# 3. Launch the local interactive HTML Capability Balance Sheet Explorer
python demo.py
```

On a headless machine, run `python demo.py --no-open` and open `demo_output/index.html` in any browser.

---

## Repository Structure and Scope

| Path | Description | Runtime / Input Requirements |
| :--- | :--- | :--- |
| `demo.py` | Self-contained HTML interactive explorer for benchmark distributions, quartile pairs, caliper sensitivity, and wage regressions | Python 3.10+ standard library only |
| `analysis/analyze_retained_bundle.py` | Offline SHA-256 provenance verifier and axiomatic capability balance sheet calculator | Python 3.10+ standard library only |
| `analysis/two_layer_accounting.py` | Full-precision FRS ability accounting, base-2 Jensen–Shannon distance, general commonality ($G_o^+$), and effective ability count ($N_o^{eff}$) | Requires `numpy` and `pandas` (`requirements-analysis.txt`) |
| `analysis/revision_statistics.py` | CPI-U log-wage deflation, O*NET ability alias normalization, HC1 robust joint Wald tests, and Benjamini–Hochberg FDR adjustment | Requires `numpy`, `pandas`, `statsmodels` |
| `analysis/revision_experiments.py` | Longitudinal O*NET 24.3/27.1/30.3 benchmark procedures and ACS/OEWS econometric specifications | Requires raw public O*NET, BLS OEWS, and IPUMS USA ACS extracts |
| `analysis/verify_frs_mapping.py`, `build_two_layer_assets.py`, `rebuild_vector_figure_assets.py` | FRS relatedness matrix checks and table/figure asset generation procedures | Require full upstream input directory layout |
| `data/retained_tables/` | Provenance-tracked summary CSV tables (`PROVENANCE.csv` + 9 summary tables) | Included directly in this repository |
| `tests/test_retained_bundle.py` | Standard-library unit test suite | Python 3.10+ standard library only |

To install optional scientific packages for the full `numpy`/`pandas`/`statsmodels` modules:

```bash
python -m pip install -r requirements-analysis.txt
```

---

## Data Provenance and Availability

- **Included in this repository (`data/retained_tables/`)**: Provenance-tracked summary tables covering the 983 cross-major near-score occupation pairs, the four quartile illustrative pairs, O*NET 24.3–27.1 and 24.3–30.3 longitudinal revision benchmarks (both updated-only and unchanged-title screens), score-proximity caliper sensitivity ($\delta \in \{0.010, 0.025, 0.050\}$), and ACS/OEWS short-run wage regression summaries. All eight provenance-tracked CSV files match their recorded SHA-256 checksums in `data/retained_tables/PROVENANCE.csv`.
- **External public datasets**: Raw O*NET database releases (24.3, 27.1, 30.3), Felten–Raj–Seamans AIOE repository matrices, U.S. Census Bureau SOC crosswalks, and BLS OEWS/CPI-U tables are publicly available from their respective agencies. Individual-level American Community Survey (ACS) microdata are accessible through IPUMS USA (`https://usa.ipums.org`) under standard terms of use and cannot be redistributed directly.

---

## Attribution and Funding

- **Maintainer:** Yonghao Wu ([WYH302](https://github.com/WYH302)), School of Electronic Science and Engineering (School of Microelectronics), South China Normal University. Co-author contact details are omitted from repository metadata for privacy; please refer to the published/author-approved manuscript for full bibliographic citation.
- **Funding:** Supported by the 2025 "Chong Bu Qiang" Provincial Financial Special Project of Guangdong Ocean University (Grant No. `080507202501`).

---

## 中文说明（Chinese Overview）

本仓库公开 **《Similar AI Exposure Scores, Different Ability Contributions: Evidence from U.S. Occupations》** 的核心分析代码、公理化“有符号能力资产负债表（Signed Capability Balance Sheet）”分解工具、保留的基准结果汇总表（含 SHA-256 溯源校验）以及零依赖本地交互式可视化浏览器（`demo.py`）。

- **快速验证（仅需 Python 标准库，无需安装第三方包或调用 API）**：
  1. `python -m unittest discover -s tests -v`（运行单元测试套件）
  2. `python analysis/analyze_retained_bundle.py`（核验全部汇总表的 SHA-256 哈希与公理化分解恒等式）
  3. `python demo.py`（生成并打开本地交互式 HTML 结果探索器）
