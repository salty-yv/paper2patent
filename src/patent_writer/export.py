from __future__ import annotations

import csv
import json
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from .schema import EvidenceReport, PatentDraft, SourceDocument


def _set_font(style, size: float, bold: bool = False) -> None:
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().rFonts.set(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "宋体"
    )


def export_docx(draft: PatentDraft, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    title = " ".join(draft.title.split()) or "专利说明书草稿"
    markdown = [f"# {title}", ""]
    for item in draft.sections:
        markdown.extend([f"## {item.heading}", ""])
        if item.paragraphs:
            for paragraph in item.paragraphs:
                markdown.extend([paragraph.text, ""])
        else:
            markdown.extend(["【待发明人补充该部分技术内容】", ""])
    # Pandoc turns $...$ and $$...$$ into native, editable Word equations.
    pypandoc.convert_text(
        "\n".join(markdown), "docx", format="markdown+tex_math_dollars",
        outputfile=str(destination),
    )
    document = Document(destination)
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.76)
    section.bottom_margin = Inches(0.70)
    section.left_margin = Inches(0.92)
    section.right_margin = Inches(0.82)

    styles = document.styles
    _set_font(styles["Normal"], 11)
    normal = styles["Normal"].paragraph_format
    normal.space_after = Pt(5)
    normal.line_spacing = 1.28
    normal.first_line_indent = Pt(22)
    _set_font(styles["Title"], 16, bold=True)
    styles["Title"].paragraph_format.space_after = Pt(18)
    styles["Title"].paragraph_format.first_line_indent = Pt(0)
    title_properties = styles["Title"].element.get_or_add_pPr()
    for child in list(title_properties):
        if child.tag == qn("w:pBdr"):
            title_properties.remove(child)
    _set_font(styles["Heading 1"], 12.5, bold=True)
    styles["Heading 1"].paragraph_format.space_before = Pt(16)
    styles["Heading 1"].paragraph_format.space_after = Pt(8)
    styles["Heading 1"].paragraph_format.first_line_indent = Pt(0)

    if document.paragraphs:
        document.paragraphs[0].style = styles["Title"]
        document.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for paragraph in document.paragraphs[1:]:
        if paragraph.style.name.startswith("Heading"):
            paragraph.style = styles["Heading 1"]
        else:
            paragraph.style = styles["Normal"]
    document.save(destination)


def export_review_files(draft: PatentDraft, source: SourceDocument, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    source_map = source.block_map()
    with (directory / "证据对照表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["章节", "段落", "正文", "概念编号", "原文块", "原文页码", "原文摘录"])
        for section in draft.sections:
            for index, paragraph in enumerate(section.paragraphs, 1):
                for sid in paragraph.source_ids:
                    block = source_map.get(sid)
                    writer.writerow([
                        section.heading, index, paragraph.text, "、".join(paragraph.node_ids), sid,
                        block.page if block else "", block.text if block else "",
                    ])
    lines = [draft.title, ""]
    for section in draft.sections:
        lines.extend([f"## {section.heading}", ""])
        lines.extend(p.text + "\n" for p in section.paragraphs)
    (directory / "说明书草稿.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "待确认事项.json").write_text(
        json.dumps({"questions": draft.questions, "warnings": draft.warnings}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_evidence_report(report: EvidenceReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "逐条证据核对.json").write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (directory / "逐条证据核对.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["主张编号", "章节", "段落", "草稿主张", "核对结果", "判断理由", "原文块", "页码", "原文引句", "引句逐字匹配", "程序检查", "待核对的补证候选", "页面核对", "页面核对人", "页面核对备注"])
        for finding in report.findings:
            if finding.evidence:
                for item in finding.evidence:
                    writer.writerow([
                        finding.id, finding.heading, finding.paragraph, finding.claim,
                        finding.verdict, finding.reason, item.source_id, item.page,
                        item.quote, "是" if item.exact_match else "否", "; ".join(finding.checks),
                        "、".join(x.source_id for x in finding.candidates),
                        finding.page_review, finding.page_review_by, finding.page_review_note,
                    ])
            else:
                writer.writerow([
                    finding.id, finding.heading, finding.paragraph, finding.claim,
                    finding.verdict, finding.reason, "", "", "", "", "; ".join(finding.checks),
                    "、".join(x.source_id for x in finding.candidates),
                    finding.page_review, finding.page_review_by, finding.page_review_note,
                ])
