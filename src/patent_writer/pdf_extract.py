from __future__ import annotations

import hashlib
import re

import pymupdf as fitz

from .schema import SourceBlock, SourceDocument


_FIGURE = re.compile(r"^(?:Figure|Fig\.?|图)\s*\d+\b", re.IGNORECASE)
_TABLE = re.compile(r"^(?:Table|表)\s*\d+\b", re.IGNORECASE)


def _margin_line_numbers(page: fitz.Page) -> set[str]:
    """Find printed line numbers by their small type and left-margin position."""
    candidates = [
        span["text"].strip()
        for block in page.get_text("dict")["blocks"] if "lines" in block
        for line in block["lines"] for span in line["spans"]
        if span["text"].strip().isdigit()
        and len(span["text"].strip()) <= 3
        and span["size"] < 8
        and span["bbox"][0] < page.rect.width * 0.18
    ]
    # A few isolated small numbers may be footnotes or data, not line numbers.
    return set(candidates) if len(candidates) >= 8 else set()


def _clean(text: str, line_numbers: set[str] | None = None) -> str:
    if line_numbers:
        text = "\n".join(
            line for line in text.splitlines() if line.strip() not in line_numbers
        )
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _block_line_numbers(block: tuple, page_width: float, line_numbers: set[str]) -> set[str]:
    # Printed line numbers extend a body-text block into the left margin.
    # Centered equations and table cells can contain the same digits as data.
    return line_numbers if block[0] < page_width * 0.2 else set()


def _reading_order(raw_blocks: list[tuple], page_width: float, line_numbers: set[str]) -> tuple[list[tuple], bool]:
    blocks = [
        b for b in raw_blocks
        if len(b) > 6 and b[6] == 0
        and _clean(b[4], _block_line_numbers(b, page_width, line_numbers))
    ]
    middle = page_width / 2
    narrow_left = [b for b in blocks if b[2] < middle + 12 and b[0] < middle]
    narrow_right = [b for b in blocks if b[0] > middle - 12 and b[2] > middle]
    two_columns = len(narrow_left) >= 3 and len(narrow_right) >= 3
    if not two_columns:
        return sorted(blocks, key=lambda b: (round(b[1] / 5), b[0])), False

    full = [b for b in blocks if b[2] - b[0] > page_width * 0.67]
    full.sort(key=lambda b: b[1])
    remaining = [b for b in blocks if b not in full]
    ordered: list[tuple] = []
    lower = -1.0
    for boundary in [*full, None]:
        upper = boundary[1] if boundary is not None else float("inf")
        band = [b for b in remaining if lower <= b[1] < upper]
        left = [b for b in band if (b[0] + b[2]) / 2 < middle]
        right = [b for b in band if b not in left]
        ordered.extend(sorted(left, key=lambda b: (b[1], b[0])))
        ordered.extend(sorted(right, key=lambda b: (b[1], b[0])))
        if boundary is not None:
            ordered.append(boundary)
            lower = boundary[3]
    # A full-width figure can overlap narrow text blocks. Preserve those blocks too.
    missing = [b for b in blocks if b not in ordered]
    ordered.extend(sorted(missing, key=lambda b: (b[1], b[0])))
    return ordered, True


def extract_pdf(content: bytes, filename: str) -> SourceDocument:
    if not content.startswith(b"%PDF"):
        raise ValueError("文件不是有效的 PDF。")
    blocks: list[SourceBlock] = []
    warnings: list[str] = []
    two_column_pages: list[int] = []
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"无法读取 PDF：{exc}") from exc
    with document:
        if document.is_encrypted:
            raise ValueError("PDF 已加密，请先提供可读取的版本。")
        for index, page in enumerate(document):
            line_numbers = _margin_line_numbers(page)
            ordered, two_columns = _reading_order(page.get_text("blocks"), page.rect.width, line_numbers)
            if two_columns:
                two_column_pages.append(index + 1)
            page_chars = 0
            for raw in ordered:
                text = _clean(raw[4], _block_line_numbers(raw, page.rect.width, line_numbers))
                if not text or (text.isdigit() and len(text) < 5):
                    continue
                page_chars += len(text)
                kind = "figure_caption" if _FIGURE.match(text) else "table_caption" if _TABLE.match(text) else "text"
                blocks.append(
                    SourceBlock(
                        id=f"B{len(blocks) + 1:04d}",
                        page=index + 1,
                        text=text,
                        bbox=tuple(round(float(v), 1) for v in raw[:4]),
                        kind=kind,
                    )
                )
            if page_chars < 60:
                warnings.append(f"第 {index + 1} 页可提取文字很少，可能需要 OCR 或人工检查。")
        if two_column_pages:
            shown = "、".join(map(str, two_column_pages[:10]))
            warnings.append(f"检测到双栏页面（{shown}），请检查阅读顺序。")
        first_page = [b for b in blocks if b.page == 1]
        title_parts: list[str] = []
        for block in first_page[:4]:
            if re.search(r"author|abstract|submission|摘要|作者", block.text, re.IGNORECASE):
                break
            if len(block.text) > 15 and len(" ".join(title_parts)) < 160:
                title_parts.append(block.text)
        title = " ".join(title_parts)[:180] or filename.rsplit(".", 1)[0]
        return SourceDocument(
            filename=filename,
            sha256=hashlib.sha256(content).hexdigest(),
            title=title,
            page_count=len(document),
            blocks=blocks,
            warnings=warnings,
        )


def source_context(source: SourceDocument, max_chars: int = 140_000) -> str:
    lines = [f"[{block.id} | 第{block.page}页] {block.text}" for block in source.blocks]
    text = "\n".join(lines)
    if len(text) > max_chars:
        raise ValueError(f"论文提取后约 {len(text)} 字符，超过当前单次处理上限 {max_chars}；请先拆分或精简输入。")
    if len(text) < 200:
        raise ValueError("PDF 中可读取文字不足，需先进行 OCR。")
    return text


def expand_table_citations(source: SourceDocument, source_ids: list[str]) -> list[str]:
    """Keep a numeric table row together with its caption and column headings."""
    blocks = source.blocks
    positions = {block.id: index for index, block in enumerate(blocks)}
    result = list(dict.fromkeys(source_ids))
    for source_id in source_ids:
        index = positions.get(source_id)
        if index is None:
            continue
        row = blocks[index]
        if len(re.findall(r"\d+(?:\.\d+)?", row.text)) < 4 or len(row.text) > 320:
            continue
        for prior in range(index - 1, max(-1, index - 21), -1):
            caption = blocks[prior]
            if caption.page != row.page:
                break
            if caption.kind == "table_caption":
                for block in blocks[prior:min(prior + 3, index)]:
                    if block.page == row.page and block.id not in result:
                        result.append(block.id)
                break
    return result
