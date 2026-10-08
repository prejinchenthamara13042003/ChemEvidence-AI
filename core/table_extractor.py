"""
Automated Chemistry Table Extraction Engine.
Extracts structured grid tables, pipe-delimited tables, whitespace-aligned
data matrices, and vertical PDF-wrapped tables from chemistry research papers
with exact page and section attribution.
"""

import re
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from core.pdf_parser import PageContent
from core.models import ExtractedTable, EvidenceCitation


class TableExtractor:
    """Detects and reconstructs scientific chemistry tables into structured formats."""

    @staticmethod
    def extract_tables_from_pages(pages: List[PageContent]) -> List[ExtractedTable]:
        """Scans all pages in the document and parses tabular data."""
        tables: List[ExtractedTable] = []
        table_counter = 1

        for page in pages:
            page_tables = TableExtractor._find_tables_in_page(page, table_counter)
            tables.extend(page_tables)
            table_counter += len(page_tables)

        return tables

    @staticmethod
    def _find_tables_in_page(page: PageContent, start_idx: int) -> List[ExtractedTable]:
        tables: List[ExtractedTable] = []
        text = page.text
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        i = 0
        section = page.sections[0]["name"] if page.sections else "Main Text"

        while i < len(lines):
            line = lines[i]
            
            # Check for table title/header marker (e.g., "Table 1:", "Table 2.")
            # Must NOT be an in-text narrative citation (e.g., "Table 1 shows that...", "as listed in Table 1")
            title_match = re.match(r"^(Table\s+\d+[:\.\-]?\s*(?:[A-Za-z0-9\s\(\)\-\,\;\/\%]+)?)", line, re.IGNORECASE)
            is_in_text_ref = re.search(r"\b(shows|summarizes|demonstrates|illustrates|lists|exhibits|displays|depicts|indicates|reveals|reported|determined)\b", line, re.IGNORECASE) is not None

            if title_match and not is_in_text_ref and len(line) < 140:
                full_title_line = line
                table_id_match = re.search(r"Table\s+\d+", full_title_line, re.IGNORECASE)
                table_id = table_id_match.group(0).title() if table_id_match else f"Table {start_idx + len(tables)}"

                # Collect candidate lines following Table X
                candidate_lines = []
                j = i + 1
                while j < len(lines):
                    next_l = lines[j].strip()
                    # Stop if next section, next table, scheme, or figure starts
                    if re.match(r"^(Table\s+\d+|Scheme\s+\d+|Figure\s+\d+|Section\b|Experimental|References|Conclusions|Biological\s+Evaluation)", next_l, re.IGNORECASE) and len(candidate_lines) >= 2:
                        break
                    # Stop if obvious narrative paragraph opening starts
                    if re.match(r"^(As shown|These results|Following|In summary|In conclusion|We synthesized|To examine|Moreover|Furthermore|It was observed|Collectively)", next_l, re.IGNORECASE):
                        break
                    # Stop if narrative sentence ends with period and has narrative verbs
                    if (next_l.endswith(".") or next_l.endswith(";")) and len(next_l.split()) >= 6:
                        if re.search(r"\b(we|were|was|synthesized|investigated|demonstrated|observed|compared|exhibited|ranging|affects)\b", next_l, re.IGNORECASE):
                            break
                    candidate_lines.append(next_l)
                    if len(candidate_lines) > 80:
                        break
                    j += 1

                if candidate_lines:
                    parsed_t = TableExtractor._parse_table_block(
                        title=full_title_line,
                        table_id=table_id,
                        lines=candidate_lines,
                        page_number=page.page_number,
                        section="Table / " + section
                    )
                    if parsed_t and len(parsed_t.rows) > 0:
                        tables.append(parsed_t)
                        i = j - 1
            i += 1

        # Fallback: scan for markdown-style pipe tables if none found via title pattern
        if not tables:
            pipe_tables = TableExtractor._extract_pipe_tables(page, start_idx)
            tables.extend(pipe_tables)

        return tables

    @staticmethod
    def _parse_table_block(
        title: str,
        table_id: str,
        lines: List[str],
        page_number: int,
        section: str
    ) -> Optional[ExtractedTable]:
        """Parses a list of table text lines into columns and rows."""
        if not lines:
            return None

        headers: List[str] = []
        rows: List[List[str]] = []

        # 1. Pipe-delimited check
        if any("|" in l for l in lines[:4]):
            for l in lines:
                if "|" in l:
                    cells = [c.strip() for c in l.split("|") if c.strip()]
                    if set(l.replace("|", "").strip()) <= {"-", ":", " "}:
                        continue
                    if not headers and len(cells) >= 2:
                        headers = cells
                    elif headers and cells:
                        if len(cells) < len(headers):
                            cells += ["-"] * (len(headers) - len(cells))
                        rows.append(cells[:len(headers)])

        # 2. Whitespace / tab delimited check
        elif "\t" in lines[0] or re.search(r"\s{2,}", lines[0]):
            headers = [h.strip() for h in re.split(r"\s{2,}|\t", lines[0]) if h.strip()]
            for r_line in lines[1:]:
                cells = [c.strip() for c in re.split(r"\s{2,}|\t", r_line) if c.strip()]
                if len(cells) >= 2:
                    if len(cells) < len(headers):
                        cells += ["-"] * (len(headers) - len(cells))
                    rows.append(cells[:len(headers)])

        # 3. Vertical wrapped PDF format check
        if (not headers or not rows) and len(lines) >= 4:
            first_row_idx = None
            # Scan only the first 2-8 lines for the first row anchor
            for idx, l in enumerate(lines[:8]):
                clean_l = l.strip()
                # A row anchor is typically a clean compound number, entry code, or drug name
                if re.match(r"^(?:Compound\s+[0-9]+[a-z]?\b|Derivative\s+[0-9]+[a-z]?\b|Entry\s+[0-9]+\b|^[0-9]{1,3}$|Erlotinib\b|Gefitinib\b|Osimertinib\b)", clean_l, re.IGNORECASE):
                    if len(clean_l) < 35 and not clean_l.endswith((".", ";", ":")):
                        first_row_idx = idx
                        break

            if first_row_idx and 2 <= first_row_idx <= 8:
                candidate_headers = lines[:first_row_idx]
                # Validate candidate headers are not prose sentences
                valid = True
                for h in candidate_headers:
                    h_clean = h.strip()
                    if len(h_clean) > 35 or len(h_clean.split()) > 5 or h_clean.endswith((".", ";")):
                        valid = False
                        break
                    if re.search(r"\b(we|synthesized|were|was|demonstrated|investigated|observed|compared|affects)\b", h_clean, re.IGNORECASE):
                        valid = False
                        break

                if valid:
                    headers = candidate_headers
                    k = len(headers)
                    row_cells = lines[first_row_idx:]
                    for chunk_idx in range(0, len(row_cells), k):
                        row_chunk = row_cells[chunk_idx:chunk_idx+k]
                        if len(row_chunk) == k:
                            rows.append(row_chunk)
                        elif len(row_chunk) >= 2:
                            rows.append(row_chunk + ["-"] * (k - len(row_chunk)))

        if not headers or not rows:
            return None

        # Sanity validation: table must have 2 to 15 columns, at least 1 row
        if not (2 <= len(headers) <= 15) or len(rows) < 1:
            return None

        # Reject if headers are narrative sentences
        for h in headers:
            if len(h) > 40 or re.search(r"\b(synthesized|investigated|demonstrated|observed|compared|exhibited|affects)\b", h, re.IGNORECASE):
                return None

        verbatim_snippet = f"{title}\n" + "\n".join(lines[:8])
        cit = EvidenceCitation(
            page_number=page_number,
            section=section,
            verbatim_quote=verbatim_snippet[:350],
            confidence=0.98
        )

        return ExtractedTable(
            table_id=table_id,
            title=title,
            headers=headers,
            rows=rows,
            page_number=page_number,
            section=section,
            evidence=cit
        )

    @staticmethod
    def _extract_pipe_tables(page: PageContent, start_idx: int) -> List[ExtractedTable]:
        tables: List[ExtractedTable] = []
        lines = page.text.split("\n")
        table_lines = []
        
        for line in lines:
            if "|" in line:
                table_lines.append(line.strip())
            else:
                if len(table_lines) >= 3:
                    parsed = TableExtractor._parse_table_block(
                        title=f"Extracted Table {start_idx + len(tables)}",
                        table_id=f"Table {start_idx + len(tables)}",
                        lines=table_lines,
                        page_number=page.page_number,
                        section="Table Data"
                    )
                    if parsed:
                        tables.append(parsed)
                table_lines = []

        if len(table_lines) >= 3:
            parsed = TableExtractor._parse_table_block(
                title=f"Extracted Table {start_idx + len(tables)}",
                table_id=f"Table {start_idx + len(tables)}",
                lines=table_lines,
                page_number=page.page_number,
                section="Table Data"
            )
            if parsed:
                tables.append(parsed)

        return tables

    @staticmethod
    def table_to_dataframe(table: ExtractedTable) -> pd.DataFrame:
        """Converts an ExtractedTable to a formatted pandas DataFrame with guaranteed unique column names."""
        if not table.headers or not table.rows:
            return pd.DataFrame()
        
        # Deduplicate and sanitize headers to guarantee PyArrow compatibility
        unique_headers = []
        counts: Dict[str, int] = {}
        for idx, h in enumerate(table.headers):
            clean_h = str(h).strip() if h else f"Col_{idx+1}"
            # Normalize whitespace and newlines
            clean_h = re.sub(r"\s+", " ", clean_h)
            if not clean_h:
                clean_h = f"Col_{idx+1}"
            if clean_h in counts:
                counts[clean_h] += 1
                unique_headers.append(f"{clean_h}_{counts[clean_h]}")
            else:
                counts[clean_h] = 0
                unique_headers.append(clean_h)

        # Normalize row lengths to match headers
        norm_rows = []
        for r in table.rows:
            if len(r) < len(unique_headers):
                norm_rows.append(r + ["-"] * (len(unique_headers) - len(r)))
            else:
                norm_rows.append(r[:len(unique_headers)])
                
        df = pd.DataFrame(norm_rows, columns=unique_headers)
        return df
