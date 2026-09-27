from __future__ import annotations

import re
from difflib import SequenceMatcher

from .pdf_extract import expand_table_citations
from .schema import ConceptGraph, EvidenceCandidate, EvidenceReport, PatentDraft, SourceDocument


_HAN = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[a-z][a-z0-9-]*", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![a-z0-9])\d+(?:\.\d+)?%?(?![a-z0-9])", re.IGNORECASE)
_BOILERPLATE = re.compile(r"本发明|本实施例|所述|具体而言|此外|一种|方法|技术领域")


def _terms(value: str) -> set[str]:
    value = _BOILERPLATE.sub("", value.casefold())
    han = "".join(_HAN.findall(value))
    return {han[i:i + 2] for i in range(len(han) - 1)} | set(_LATIN.findall(value))


def _similarity(claim: str, concept: str) -> float:
    a, b = _terms(claim), _terms(concept)
    overlap = len(a & b) / max(1, len(a | b))
    plain_a = _BOILERPLATE.sub("", claim)
    plain_b = _BOILERPLATE.sub("", concept)
    sequence = SequenceMatcher(None, plain_a, plain_b).ratio()
    return 0.6 * overlap + 0.4 * sequence


def _search_anchors(value: str) -> tuple[set[str], set[str]]:
    # Decimal values are useful anchors in result tables; English method names
    # help when the claim is Chinese and the source is English.
    numbers = {x for x in _NUMBER.findall(value.casefold()) if "." in x or "%" in x}
    words = {x.casefold() for x in _LATIN.findall(value) if len(x) >= 3}
    return numbers, words


def attach_candidate_sources(
    report: EvidenceReport, source: SourceDocument, draft: PatentDraft,
    graph: ConceptGraph,
) -> EvidenceReport:
    """Suggest source blocks for human review; never change a verdict."""
    block_map = source.block_map()
    paragraphs = {
        (section.heading, index): paragraph
        for section in draft.sections
        for index, paragraph in enumerate(section.paragraphs, 1)
    }
    for finding in report.findings:
        if finding.verdict == "supported":
            finding.candidates = []
            continue
        paragraph = paragraphs.get((finding.heading, finding.paragraph))
        if paragraph is None:
            continue
        cited = set(paragraph.source_ids)
        suggestions: dict[str, tuple[float, str]] = {}

        for sid in expand_table_citations(source, paragraph.source_ids):
            if sid not in cited and sid in block_map:
                suggestions[sid] = (2.0, "同页表题或列标题；需与数据行一起核对")

        for node in graph.nodes:
            score = _similarity(finding.claim, node.text)
            if score < 0.24:
                continue
            for sid in node.source_ids:
                if sid in cited or sid not in block_map:
                    continue
                score_with_context = score + (0.08 if node.id in paragraph.node_ids else 0.0)
                if score_with_context > suggestions.get(sid, (0.0, ""))[0]:
                    suggestions[sid] = (score_with_context, f"与概念 {node.id} 内容相近；请核对原文")

        claim_numbers, claim_words = _search_anchors(finding.claim)
        if claim_numbers or len(claim_words) >= 2:
            for block in source.blocks:
                if block.id in cited or block.id in suggestions:
                    continue
                block_numbers, block_words = _search_anchors(block.text)
                number_hits = len(claim_numbers & block_numbers)
                word_hits = len(claim_words & block_words)
                if number_hits >= 2 or (number_hits >= 1 and word_hits >= 1) or word_hits >= 3:
                    suggestions[block.id] = (
                        0.25 + 0.08 * number_hits + 0.03 * word_hits,
                        "全文数值或英文术语匹配；需核对语义和表头",
                    )

        finding.candidates = [
            EvidenceCandidate(source_id=sid, page=block_map[sid].page, reason=reason)
            for sid, (_, reason) in sorted(suggestions.items(), key=lambda item: item[1][0], reverse=True)[:5]
        ]
    return report
