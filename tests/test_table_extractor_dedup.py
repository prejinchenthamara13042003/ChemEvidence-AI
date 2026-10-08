"""
Unit tests for TableExtractor duplicate column name handling,
PyArrow compatibility, and narrative prose rejection.
"""

import unittest
import pandas as pd
import pyarrow as pa
from core.models import ExtractedTable, EvidenceCitation
from core.table_extractor import TableExtractor
from core.pdf_parser import PageContent


class TestTableExtractorDuplicateColumns(unittest.TestCase):

    def test_duplicate_column_deduplication(self):
        # Create a table with duplicate headers matching the user's issue
        headers = ["Compound", "IC50", "Compound", "Yield", "IC50", "Notes"]
        rows = [
            ["1a", "12 nM", "1a", "78%", "12 nM", "active"],
            ["2b", "45 nM", "2b", "65%", "45 nM", "moderate"]
        ]
        cit = EvidenceCitation(page_number=1, section="SAR", verbatim_quote="Test", confidence=0.99)
        table = ExtractedTable(
            table_id="Table 1",
            title="Table 1. In vitro assays",
            headers=headers,
            rows=rows,
            page_number=1,
            section="SAR",
            evidence=cit
        )

        df = TableExtractor.table_to_dataframe(table)
        self.assertFalse(df.empty)
        # Check that all column names are strictly unique
        cols = list(df.columns)
        self.assertEqual(len(cols), len(set(cols)))
        self.assertIn("Compound", cols)
        self.assertIn("Compound_1", cols)
        self.assertIn("IC50", cols)
        self.assertIn("IC50_1", cols)

        # Ensure PyArrow can convert this DataFrame without ValueError: Duplicate column names found
        arrow_table = pa.Table.from_pandas(df)
        self.assertEqual(arrow_table.num_columns, len(cols))
        self.assertEqual(arrow_table.num_rows, 2)

    def test_narrative_paragraph_not_extracted_as_table(self):
        # A page containing a narrative paragraph like the one reported in user error
        prose_text = (
            "Results and Discussion\n\n"
            "To examine how the structure of the target compounds affects their activity\n"
            "against EGFR kinase, compounds 4a–c and 6a–c were synthesized, and the\n"
            "structure–activity relationships (SARs) of the novel scaffold were investigated.\n"
            "We synthesized compounds 8a–c, which had no acrylamide structure.\n"
            "All compounds containing the acrylamide structure demonstrated better inhibitory activities\n"
            "against the EGFRwt kinase compared to compound 7j (IC50 = 25.69 nM), which we have previously\n"
            "reported as a reversible EGFR inhibitor. Among them, compound 6c displayed the strongest\n"
            "inhibitory activity against EGFRwt kinase (IC50 = 10.76 nM) and was superior to the reference\n"
            "compound gefitinib (IC50 = 11.75 nM).\n"
        )
        page = PageContent(page_number=3, text=prose_text, sections=[{"name": "Results and Discussion", "page": 3}])
        tables = TableExtractor.extract_tables_from_pages([page])
        # Prose text should NOT be parsed into a fake 42-column table
        self.assertEqual(len(tables), 0)


if __name__ == "__main__":
    unittest.main()
