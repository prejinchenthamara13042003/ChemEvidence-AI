"""
Page-aware PDF and Document Parser for Chemistry Literature.
Preserves exact page numbering, detects scientific sections,
renders high-resolution page previews via PyMuPDF (fitz), and highlights verbatim quotes.
"""

import io
import re
import html
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from pypdf import PdfReader
from core.models import PaperMetadata


class PageContent:
    """Represents text and metadata for an individual page."""
    def __init__(self, page_number: int, text: str, sections: Optional[List[Dict[str, str]]] = None):
        self.page_number = page_number
        self.text = text
        self.sections = sections or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "sections": self.sections
        }


class DocumentParser:
    """Parses PDF or text research papers with page-level fidelity and high-resolution rendering."""

    SECTION_PATTERNS = [
        (r"(?i)\babstract\b", "Abstract"),
        (r"(?i)\bintroduction\b", "Introduction"),
        (r"(?i)\bresults\s+and\s+discussion\b", "Results & Discussion"),
        (r"(?i)\bresults\b", "Results"),
        (r"(?i)\bdiscussion\b", "Discussion"),
        (r"(?i)\bbiological\s+evaluation\b", "Biological Evaluation"),
        (r"(?i)\bbioactivity\b", "Bioactivity"),
        (r"(?i)\bchemistry\b", "Chemistry & Synthesis"),
        (r"(?i)\bexperimental\s+section\b", "Experimental Section"),
        (r"(?i)\bexperimental\s+procedures\b", "Experimental Procedures"),
        (r"(?i)\bmaterials\s+and\s+methods\b", "Materials & Methods"),
        (r"(?i)\bconclusions?\b", "Conclusions"),
        (r"(?i)\backnowledg(e)?ments?\b", "Acknowledgments"),
        (r"(?i)\bsupporting\s+information\b", "Supporting Information"),
        (r"(?i)\breferences?\b", "References"),
        (r"(?i)\btable\s+\d+[:\.]?", "Table"),
        (r"(?i)\bscheme\s+\d+[:\.]?", "Scheme"),
    ]

    @classmethod
    def parse_pdf_bytes(cls, pdf_bytes: bytes) -> List[PageContent]:
        """Extracts text page-by-page from raw PDF bytes using PyMuPDF or pypdf."""
        pages: List[PageContent] = []

        if fitz is not None:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for idx, page in enumerate(doc):
                    page_num = idx + 1
                    raw_text = page.get_text() or ""
                    cleaned_text = cls._clean_text(raw_text)
                    sections = cls._identify_sections(cleaned_text)
                    pages.append(PageContent(page_number=page_num, text=cleaned_text, sections=sections))
                if pages:
                    return pages
            except Exception:
                pages = []

        # Fallback to pypdf if fitz encounters an error
        stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(stream)
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            cleaned_text = cls._clean_text(raw_text)
            sections = cls._identify_sections(cleaned_text)
            pages.append(PageContent(page_number=page_num, text=cleaned_text, sections=sections))

        return pages

    @classmethod
    def parse_text_content(cls, text: str) -> List[PageContent]:
        """
        Parses text content. If explicit page markers (e.g. '--- Page 1 ---' or '[Page 1]')
        exist, uses them. Otherwise chunks into realistic page lengths (~3000 chars).
        """
        cleaned_text = cls._clean_text(text)
        page_split_pattern = r"(?:^|\n)(?:---+\s*Page\s*(\d+)\s*---+|\[Page\s*(\d+)\])"
        matches = list(re.finditer(page_split_pattern, cleaned_text, re.IGNORECASE))

        if matches:
            pages: List[PageContent] = []
            for i in range(len(matches)):
                start = matches[i].end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
                p_num = int(matches[i].group(1) or matches[i].group(2) or (i + 1))
                page_body = cleaned_text[start:end].strip()
                sections = cls._identify_sections(page_body)
                pages.append(PageContent(page_number=p_num, text=page_body, sections=sections))
            return pages

        # Fallback: Split by character windows respecting paragraph breaks
        paragraphs = cleaned_text.split("\n\n")
        pages = []
        current_page_text = []
        current_char_count = 0
        current_page_num = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_char_count + len(para) > 3000 and current_page_text:
                full_page = "\n\n".join(current_page_text)
                pages.append(PageContent(
                    page_number=current_page_num,
                    text=full_page,
                    sections=cls._identify_sections(full_page)
                ))
                current_page_num += 1
                current_page_text = [para]
                current_char_count = len(para)
            else:
                current_page_text.append(para)
                current_char_count += len(para)

        if current_page_text:
            full_page = "\n\n".join(current_page_text)
            pages.append(PageContent(
                page_number=current_page_num,
                text=full_page,
                sections=cls._identify_sections(full_page)
            ))

        if not pages:
            pages.append(PageContent(page_number=1, text=cleaned_text, sections=cls._identify_sections(cleaned_text)))

        return pages

    @classmethod
    def extract_metadata(cls, pages: List[PageContent]) -> PaperMetadata:
        """Extracts publication title, authors, DOI, and page stats."""
        total_chars = sum(len(p.text) for p in pages)
        total_words = sum(len(p.text.split()) for p in pages)
        page_count = len(pages)

        first_page = pages[0].text if pages else ""
        lines = [line.strip() for line in first_page.split("\n") if line.strip()]

        # Heuristic title extraction (often the first significant line)
        title = "Scientific Chemistry Literature"
        authors = "Not specified"
        doi = None

        for line in lines[:5]:
            if len(line) > 15 and not line.lower().startswith(("http", "doi", "issn", "page")):
                title = line
                break

        # Check for DOI
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", first_page)
        if doi_match:
            doi = doi_match.group(0)

        # Look for author indications
        for line in lines[1:6]:
            if any(term in line.lower() for term in ["university", "department", "laboratory", "institute"]):
                break
            if "," in line and len(line) < 120 and line != title:
                authors = line
                break

        return PaperMetadata(
            title=title,
            authors=authors,
            journal_or_doi=doi,
            page_count=page_count,
            total_characters=total_chars,
            total_words=total_words
        )

    # ---------------------------------------------------------
    # PyMuPDF High-Resolution Page Rendering & Amber Highlighting
    # ---------------------------------------------------------

    @classmethod
    def get_pdf_page_count(cls, pdf_bytes: bytes) -> int:
        """Returns the total number of pages in the PDF."""
        if fitz is None:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            return len(reader.pages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return len(doc)

    @classmethod
    def find_quote_rects(cls, page: Any, quote: str) -> List[Any]:
        """
        Locates quote rectangles on a PyMuPDF page using multi-strategy search.
        Handles hyphenation, whitespace variance, quotation marks, and ellipses.
        """
        if not quote or fitz is None:
            return []

        # 1. Clean quote string
        q = quote.strip().strip('"\'“”`')
        q = re.sub(r"\.\.\.+$", "", q).strip()
        if not q:
            return []

        # Strategy A: Direct search
        rects = page.search_for(q)
        if rects:
            return rects

        # Strategy B: Whitespace-normalized search
        q_norm = " ".join(q.split())
        rects = page.search_for(q_norm)
        if rects:
            return rects

        # Strategy C: Sentence / punctuation split
        sentences = [s.strip() for s in re.split(r"[.;\n]", q_norm) if len(s.strip()) > 15]
        for s in sentences:
            rects = page.search_for(s)
            if rects:
                return rects

        # Strategy D: Multi-word phrase search (chunks of 6-8 words)
        words = q_norm.split()
        if len(words) > 5:
            first_phrase = " ".join(words[:min(8, len(words))])
            rects = page.search_for(first_phrase)
            if rects:
                all_rects = list(rects)
                for i in range(8, len(words), 8):
                    chunk = " ".join(words[i:i+8])
                    if len(chunk.split()) >= 3:
                        sub_rects = page.search_for(chunk)
                        all_rects.extend(sub_rects)
                return all_rects

        # Strategy E: Sliding window of 4-5 words
        if len(words) >= 4:
            for i in range(0, min(len(words) - 3, 8)):
                sub = " ".join(words[i:i+4])
                rects = page.search_for(sub)
                if rects:
                    return rects

        return []

    @classmethod
    def render_pdf_page_image(
        cls,
        pdf_bytes: bytes,
        page_number: int,
        zoom: float = 2.0,
        highlight_quote: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Renders an uploaded PDF page into a high-resolution PNG image directly
        inside the application using PyMuPDF (fitz).
        Optionally highlights verbatim quotes using PyMuPDF amber highlight box annotations.
        """
        if fitz is None:
            return None

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            if total_pages == 0:
                return None

            # Clamp page number (1-indexed to 0-indexed)
            idx = max(0, min(page_number - 1, total_pages - 1))
            page = doc[idx]

            # Apply Amber Highlight Box Annotations if quote is provided
            if highlight_quote and highlight_quote.strip():
                rects = cls.find_quote_rects(page, highlight_quote)
                if rects:
                    try:
                        # 1. Native PyMuPDF Highlight Annotation
                        annot = page.add_highlight_annot(rects)
                        annot.set_colors(stroke=(1.0, 0.72, 0.0))  # Warm Amber
                        annot.update()
                    except Exception:
                        pass

                    try:
                        # 2. Crisp overlay bounding box with glowing amber border
                        shape = page.new_shape()
                        for r in rects:
                            padded = fitz.Rect(r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2)
                            shape.draw_rect(padded)
                        shape.finish(
                            color=(0.96, 0.62, 0.04),      # Amber outline (#F59E0B)
                            fill=(1.0, 0.85, 0.25),        # Golden highlight fill
                            fill_opacity=0.38,
                            width=1.6
                        )
                        shape.commit()
                    except Exception:
                        pass

            # High-Resolution Pixmap Rendering (zoom=2.0 gives ~150-200 DPI crispness)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            return pix.tobytes("png")

        except Exception as e:
            print(f"Error rendering PDF page image: {e}")
            return None

    @classmethod
    def highlight_quote_in_text(cls, text: str, quote: Optional[str] = None) -> str:
        """
        Wraps the verbatim quote snippet inside glowing yellow <mark> tags
        for real-time visual grounding in text reader view.
        """
        if not text:
            return ""

        escaped_text = html.escape(text)
        if not quote or not quote.strip():
            return escaped_text

        q_clean = quote.strip().strip('"\'“”`')
        q_clean = re.sub(r"\.\.\.+$", "", q_clean).strip()
        if not q_clean:
            return escaped_text

        # Try exact search on escaped text
        escaped_quote = html.escape(q_clean)
        pattern = re.compile(re.escape(escaped_quote), re.IGNORECASE)
        if pattern.search(escaped_text):
            return pattern.sub(
                lambda m: f'<mark class="glowing-citation-mark">{m.group(0)}</mark>',
                escaped_text
            )

        # Try whitespace normalized
        q_words = escaped_quote.split()
        if len(q_words) >= 3:
            flexible_regex = r"\s+".join([re.escape(w) for w in q_words])
            try:
                pattern2 = re.compile(flexible_regex, re.IGNORECASE)
                if pattern2.search(escaped_text):
                    return pattern2.sub(
                        lambda m: f'<mark class="glowing-citation-mark">{m.group(0)}</mark>',
                        escaped_text
                    )
            except Exception:
                pass

            # Try first 5-8 words if long quote
            first_phrase = r"\s+".join([re.escape(w) for w in q_words[:min(7, len(q_words))]])
            try:
                pattern3 = re.compile(first_phrase, re.IGNORECASE)
                if pattern3.search(escaped_text):
                    return pattern3.sub(
                        lambda m: f'<mark class="glowing-citation-mark">{m.group(0)}</mark>',
                        escaped_text
                    )
            except Exception:
                pass

        return escaped_text

    @classmethod
    def synthesize_pdf_from_pages(cls, pages: List[PageContent]) -> Optional[bytes]:
        """
        Synthesizes a clean PDF from parsed text pages using PyMuPDF.
        Enables high-resolution page viewing even for text manuscripts or pasted literature.
        """
        if fitz is None or not pages:
            return None

        try:
            doc = fitz.open()
            for p in pages:
                page = doc.new_page(width=595, height=842)
                
                header_text = f"ChemEvidence AI Document Preview • Page {p.page_number} of {len(pages)}"
                page.insert_text((40, 40), header_text, fontsize=9, color=(0.4, 0.45, 0.5))

                if p.sections:
                    sec_names = " | ".join([s["name"] for s in p.sections])
                    page.insert_text((40, 56), f"Sections: {sec_names}", fontsize=8.5, color=(0.0, 0.5, 0.75))

                rect = fitz.Rect(40, 70, 555, 800)
                page.insert_textbox(rect, p.text, fontsize=10, fontname="helv", lineheight=1.35, color=(0.1, 0.12, 0.15))

            pdf_stream = io.BytesIO()
            doc.save(pdf_stream)
            return pdf_stream.getvalue()
        except Exception as e:
            print(f"Error synthesizing PDF from pages: {e}")
            return None

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Normalize whitespace and line endings while preserving formatting."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove null bytes or control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text.strip()

    @classmethod
    def _identify_sections(cls, text: str) -> List[Dict[str, str]]:
        """Identify section headings present on the page."""
        sections = []
        for pattern, section_name in cls.SECTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                sections.append({
                    "name": section_name,
                    "position": match.start()
                })
        # Sort by position
        sections.sort(key=lambda s: s["position"])
        return sections
