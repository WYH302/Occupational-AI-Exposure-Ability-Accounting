"""Standard-library unit tests for the capability-accounting repository."""

from __future__ import annotations

import math
import unittest

from analysis.analyze_retained_bundle import (
    decompose_signed_capability_balance_sheet,
    jensen_shannon_distance_base2,
    verify_retained_bundle,
)
from demo import build_explorer_html


class TestCapabilityAccountingRepository(unittest.TestCase):
    def test_provenance_and_retained_tables(self) -> None:
        report = verify_retained_bundle()
        self.assertEqual(report["provenance_files_verified"], 8)
        self.assertEqual(report["near_score_cross_major_pairs"], 983)
        self.assertAlmostEqual(report["near_score_median_jsd"], 0.1731, places=4)
        self.assertAlmostEqual(report["near_score_p90_jsd"], 0.2612, places=4)
        self.assertAlmostEqual(report["exceedance_share_24_3_to_27_1_pct"], 67.7, places=1)
        self.assertAlmostEqual(report["exceedance_share_24_3_to_30_3_pct"], 61.3, places=1)

    def test_signed_capability_balance_sheet_identity(self) -> None:
        res_o = decompose_signed_capability_balance_sheet([0.50, 0.25, 0.25], [1.0, 3.0, 4.0], 2.0, 1.0)
        res_k = decompose_signed_capability_balance_sheet([0.40, 0.55, 0.05], [1.0, 3.0, 4.0], 2.0, 1.0)
        self.assertAlmostEqual(res_o["z_score"], 0.25, places=12)
        self.assertAlmostEqual(res_k["z_score"], 0.25, places=12)
        self.assertAlmostEqual(res_o["net_balance_P_minus_N"], res_o["z_score"], places=12)
        self.assertAlmostEqual(res_k["net_balance_P_minus_N"], res_k["z_score"], places=12)
        self.assertLess(res_o["add_up_abs_error"], 1e-15)

    def test_jensen_shannon_distance_bounds_and_symmetry(self) -> None:
        p = [0.50, 0.25, 0.25]
        q = [0.40, 0.55, 0.05]
        self.assertAlmostEqual(jensen_shannon_distance_base2(p, p), 0.0, places=12)
        d_pq = jensen_shannon_distance_base2(p, q)
        d_qp = jensen_shannon_distance_base2(q, p)
        self.assertAlmostEqual(d_pq, d_qp, places=12)
        self.assertTrue(0.0 < d_pq <= 1.0)
        disjoint_a = [1.0, 0.0]
        disjoint_b = [0.0, 1.0]
        self.assertAlmostEqual(jensen_shannon_distance_base2(disjoint_a, disjoint_b), 1.0, places=12)

    def test_demo_html_generation(self) -> None:
        out_path = build_explorer_html()
        self.assertTrue(out_path.exists())
        text = out_path.read_text(encoding="utf-8")
        self.assertIn("Signed Capability Balance Sheet", text)
        self.assertIn("0.1731", text)


if __name__ == "__main__":
    unittest.main()
