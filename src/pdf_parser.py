"""
High-Resolution PDF Page Viewer and Document Parser for Chemistry Literature.
Converts uploaded research paper PDFs into high-resolution page previews directly
inside the application using PyMuPDF (fitz), with page navigation controls
and amber quote highlighting.
"""

from core.pdf_parser import PageContent, DocumentParser

__all__ = ["PageContent", "DocumentParser"]
