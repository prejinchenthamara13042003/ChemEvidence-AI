"""
Unit tests for ChemCalculator (core/chem_calculator.py).
Tests:
- Formula parsing and stoichiometric MW calculation with standard IUPAC weights
- Subscript character normalization (e.g. C₁₉H₂₂N₄O₂)
- Space tolerance in formula parsing (e.g. C 19 H 22 N 4 O 2)
- Ion-to-neutral formula conversion ([M+H]+ -> neutral)
- Document-wide formula and MW binding from tables and characterization text
"""

import unittest
from core.models import CompoundInfo, ExtractedTable, EvidenceCitation
from core.pdf_parser import PageContent
from core.chem_calculator import ChemCalculator


class TestChemCalculator(unittest.TestCase):

    def test_stoichiometric_molecular_weight(self):
        # C19H22N4O2: 19*12.011 + 22*1.008 + 4*14.007 + 2*15.999 = 338.41 g/mol
        mw_3b = ChemCalculator.calculate_molecular_weight("C19H22N4O2")
        self.assertIsNotNone(mw_3b)
        self.assertAlmostEqual(mw_3b, 338.41, delta=0.1)

        # C18H20N4O
        mw_3a = ChemCalculator.calculate_molecular_weight("C18H20N4O")
        self.assertIsNotNone(mw_3a)
        self.assertAlmostEqual(mw_3a, 308.38, delta=0.1)

        # C22H23N3O4 (Erlotinib)
        mw_erlotinib = ChemCalculator.calculate_molecular_weight("C22H23N3O4")
        self.assertIsNotNone(mw_erlotinib)
        self.assertAlmostEqual(mw_erlotinib, 393.44, delta=0.1)

    def test_unicode_subscripts_and_spaces(self):
        # Subscript normalization
        mw_sub = ChemCalculator.calculate_molecular_weight("C₁₉H₂₂N₄O₂")
        self.assertIsNotNone(mw_sub)
        self.assertAlmostEqual(mw_sub, 338.41, delta=0.1)

        # Spaced format
        norm = ChemCalculator.normalize_formula("C 19 H 22 N 4 O 2")
        self.assertEqual(norm, "C19H22N4O2")

    def test_neutral_from_ion(self):
        # [M+H]+ ion C19H23N4O2 -> neutral C19H22N4O2
        neutral = ChemCalculator.neutral_formula_from_ion("C19H23N4O2")
        self.assertEqual(neutral, "C19H22N4O2")

    def test_document_wide_binding(self):
        compounds = [
            CompoundInfo(
                compound_id="Compound 7J",
                name="Quinazoline Compound 7J",
                evidence=EvidenceCitation(page_number=2, section="Synthesis", verbatim_quote="Synthesis of 7j...", confidence=0.95)
            ),
            CompoundInfo(
                compound_id="Compound 4A",
                name="Kinase Inhibitor Compound 4A",
                evidence=EvidenceCitation(page_number=1, section="Abstract", verbatim_quote="Notably 4a...", confidence=0.95)
            )
        ]

        table = ExtractedTable(
            table_id="table_1",
            title="Table 1: Synthetic Compounds",
            page_number=2,
            headers=["Compound", "Formula", "Yield (%)"],
            rows=[
                ["Compound 7J", "C24H27N5O3", "82%"],
                ["Compound 4A", "C17H19N5O", "68%"]
            ],
            evidence=EvidenceCitation(
                page_number=2,
                section="Table 1",
                verbatim_quote="Table 1: Synthetic Compounds",
                confidence=0.95
            )
        )

        pages = [
            PageContent(page_number=1, text="Compound 4a was evaluated against EGFR.", sections=[]),
            PageContent(page_number=2, text="Compound 7j: HRMS (ESI) m/z calcd for C24H28N5O3 [M+H]+ 434.2187", sections=[])
        ]

        ChemCalculator.bind_formulas_and_weights_document_wide(compounds, [table], pages)

        # Compound 7J should have formula and calculated MW
        comp_7j = compounds[0]
        self.assertEqual(comp_7j.formula, "C24H27N5O3")
        self.assertIsNotNone(comp_7j.molecular_weight)
        self.assertAlmostEqual(comp_7j.molecular_weight, 433.5, delta=0.5)

        # Compound 4A should have formula and calculated MW
        comp_4a = compounds[1]
        self.assertEqual(comp_4a.formula, "C17H19N5O")
        self.assertIsNotNone(comp_4a.molecular_weight)
        self.assertAlmostEqual(comp_4a.molecular_weight, 309.37, delta=0.1)


if __name__ == "__main__":
    unittest.main()
