"""
Generates multi-page PDF versions of sample chemistry papers
using ReportLab with scientific styling, headers, and page counters.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_pdf_from_text(txt_path: str, pdf_path: str):
    """Parses a sample paper text file separated by '--- Page X ---' and generates a PDF."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    pages = content.split("--- Page ")
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=8
    )
    author_style = ParagraphStyle(
        'DocAuthor',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569'),
        spaceAfter=14
    )
    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    table_cell_style = ParagraphStyle(
        'DocTableCell',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )

    story = []

    for idx, page_raw in enumerate(pages):
        if not page_raw.strip():
            continue
        
        # Clean page header
        lines = page_raw.strip().split("\n")
        # first line might be page number "1 ---"
        body_lines = lines[1:] if "---" in lines[0] else lines

        if idx > 1:
            story.append(PageBreak())

        in_table = False
        table_rows = []

        for line in body_lines:
            line_str = line.strip()
            if not line_str:
                if in_table and table_rows:
                    # Render table
                    t = Table(table_rows, hAlign='LEFT')
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 8))
                    table_rows = []
                    in_table = False
                story.append(Spacer(1, 4))
                continue

            # Detect table rows with |
            if "|" in line_str and not line_str.startswith("#"):
                in_table = True
                cells = [Paragraph(c.strip(), table_cell_style) for c in line_str.split("|")]
                table_rows.append(cells)
                continue
            elif in_table and table_rows:
                t = Table(table_rows, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
                table_rows = []
                in_table = False

            # Check headings
            if line_str.endswith(":") and len(line_str) < 60:
                story.append(Paragraph(line_str, heading_style))
            elif idx == 1 and line_str == body_lines[0]:
                story.append(Paragraph(line_str, title_style))
            elif idx == 1 and line_str == body_lines[1]:
                story.append(Paragraph(line_str, author_style))
            else:
                story.append(Paragraph(line_str, body_style))

        if in_table and table_rows:
            t = Table(table_rows, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t)

    doc.build(story)
    print(f"Generated PDF: {pdf_path}")


def main():
    base_dir = "/Users/prejin/Thesis/sample_papers"
    papers = [
        ("paper1_kinase_inhibitors.txt", "paper1_kinase_inhibitors.pdf"),
        ("paper2_catalytic_synthesis_conflicts.txt", "paper2_catalytic_synthesis_conflicts.pdf"),
        ("paper3_natural_product_sar.txt", "paper3_natural_product_sar.pdf"),
    ]

    for txt_name, pdf_name in papers:
        txt_p = os.path.join(base_dir, txt_name)
        pdf_p = os.path.join(base_dir, pdf_name)
        if os.path.exists(txt_p):
            create_pdf_from_text(txt_p, pdf_p)


if __name__ == "__main__":
    main()
