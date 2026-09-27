from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from .model_api import ModelClient
from .schema import (
    EvidenceFinding, EvidenceQuote, EvidenceReport, EvidenceVerdict,
    Heading, PatentDraft, SourceDocument,
)


_MATH_BLOCK = re.compile(r"(\$\$.*?\$\$)", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[。！？；])\s*")
_NUMBER = re.compile(r"(?<![A-Za-z0-9-])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|[GMK])?(?![A-Za-z0-9])")
_VERDICTS: set[str] = {"supported", "partial", "unsupported", "uncertain"}


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    heading: Heading
    paragraph: int
    text: str
    source_ids: tuple[str, ...]

    @property
    def paragraph_key(self) -> str:
        return f"{self.heading}:{self.paragraph}"

    @property
    def has_math(self) -> bool:
        return "$" in self.text


def draft_fingerprint(draft: PatentDraft) -> str:
    payload = json.dumps(draft.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_blocks_fingerprint(source: SourceDocument) -> str:
    payload = json.dumps(
        [block.model_dump() for block in source.blocks],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_units(draft: PatentDraft) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for section in draft.sections:
        for paragraph_index, paragraph in enumerate(section.paragraphs, 1):
            for part in _MATH_BLOCK.split(paragraph.text):
                if not part.strip():
                    continue
                segments = [part] if part.startswith("$$") else _SENTENCE_END.split(part)
                for segment in segments:
                    text = segment.strip()
                    if text:
                        units.append(EvidenceUnit(
                            id=f"U{len(units) + 1:04d}", heading=section.heading,
                            paragraph=paragraph_index, text=text,
                            source_ids=tuple(paragraph.source_ids),
                        ))
    return units


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _numbers(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"(?<=\d)\s+(?=[%GMK](?![A-Za-z]))", "", normalized)
    return {match.group().replace(",", "") for match in _NUMBER.finditer(normalized)}


def _formula_finding(unit: EvidenceUnit, source: SourceDocument, ordinal: int) -> EvidenceFinding:
    blocks = source.block_map()
    citations = [
        EvidenceQuote(source_id=sid, page=blocks[sid].page, quote="", exact_match=False)
        for sid in unit.source_ids if sid in blocks
    ]
    return EvidenceFinding(
        id=f"C{ordinal:04d}", unit_id=unit.id, heading=unit.heading,
        paragraph=unit.paragraph, claim=unit.text, verdict="uncertain",
        reason="公式或含公式的陈述需对照原论文页面核对；PDF 提取文字不能可靠验证数学排版。",
        evidence=citations, checks=["formula_visual_review"],
    )


def _unchecked_finding(unit: EvidenceUnit, ordinal: int, reason: str) -> EvidenceFinding:
    return EvidenceFinding(
        id=f"C{ordinal:04d}", unit_id=unit.id, heading=unit.heading,
        paragraph=unit.paragraph, claim=unit.text, verdict="uncertain",
        reason=reason, checks=["not_fully_checked"],
    )


def _validate_claim(
    unit: EvidenceUnit, value: dict, source: SourceDocument, ordinal: int,
) -> EvidenceFinding:
    blocks = source.block_map()
    raw_claim = str(value.get("claim") or "").strip()
    claim = raw_claim or unit.text
    verdict = value.get("verdict") if value.get("verdict") in _VERDICTS else "uncertain"
    reason = str(value.get("reason") or "").strip()[:500] or "模型未提供判断理由。"
    checks: list[str] = []

    if not raw_claim or _normalize(raw_claim) not in _normalize(unit.text):
        claim = unit.text
        verdict = "uncertain"
        checks.append("claim_not_in_draft")
        reason = "模型返回的主张无法逐字定位到草稿原句；需要人工核对。"

    quotes: list[EvidenceQuote] = []
    raw_evidence = value.get("evidence")
    for raw in raw_evidence if isinstance(raw_evidence, list) else []:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("source_id") or "")
        if sid not in unit.source_ids or sid not in blocks:
            checks.append("citation_outside_paragraph")
            continue
        quote = str(raw.get("quote") or "").strip()
        exact = bool(quote and _normalize(quote) in _normalize(blocks[sid].text))
        quotes.append(EvidenceQuote(
            source_id=sid, page=blocks[sid].page,
            quote=quote, exact_match=exact,
        ))
        if not exact:
            checks.append("quote_not_found")

    if verdict == "supported":
        if not quotes:
            verdict = "uncertain"
            checks.append("missing_quote")
            reason = "模型称原文支持该主张，但未给出可核对的原文引句。"
        elif any(not quote.exact_match for quote in quotes):
            verdict = "uncertain"
            reason = "模型给出的引句无法在所引原文块中逐字找到。"
        else:
            claim_numbers = _numbers(claim)
            quote_numbers = _numbers(" ".join(q.quote for q in quotes))
            missing = sorted(claim_numbers - quote_numbers)
            if missing:
                verdict = "uncertain"
                checks.append("number_not_in_quote")
                reason = f"原文引句未包含主张中的数值：{', '.join(missing)}；需重新核对引用。"

    if any(flag in checks for flag in ("quote_not_found", "citation_outside_paragraph")):
        verdict = "uncertain"
        reason = "模型引句或引用编号无法在该段允许的原文位置核实。"
    elif checks and verdict == "supported":
        verdict = "uncertain"
        reason = "引用校验未通过，需人工核对原文。"

    return EvidenceFinding(
        id=f"C{ordinal:04d}", unit_id=unit.id, heading=unit.heading,
        paragraph=unit.paragraph, claim=claim, verdict=verdict,
        reason=reason, evidence=quotes, checks=list(dict.fromkeys(checks)),
    )


def review_paragraph(
    units: list[EvidenceUnit], source: SourceDocument, model: ModelClient,
    start_ordinal: int,
) -> list[EvidenceFinding]:
    findings: list[EvidenceFinding] = []
    formula_intro_ids = {
        unit.id for unit, following in zip(units, units[1:])
        if unit.text.rstrip().endswith(("：", ":")) and following.has_math
    }
    regular = [unit for unit in units if not unit.has_math and unit.id not in formula_intro_ids]
    for unit in units:
        if unit.has_math or unit.id in formula_intro_ids:
            findings.append(_formula_finding(unit, source, start_ordinal + len(findings)))
    if not regular:
        return findings

    blocks = source.block_map()
    cited = list(dict.fromkeys(sid for unit in regular for sid in unit.source_ids if sid in blocks))
    evidence = [
        {"source_id": sid, "page": blocks[sid].page, "text": blocks[sid].text}
        for sid in cited
    ]
    prompts = [{"unit_id": unit.id, "text": unit.text, "allowed_source_ids": unit.source_ids} for unit in regular]
    system = (
        "你是独立的技术证据核对员。对中文草稿中的每条技术事实，只按给定论文原文判断。"
        "同一段中有根据的部分和无根据的部分要分开。不得用常识、其他段落或未给出的论文内容补证据。"
        "允许忠实的中英翻译和技术同义表达；‘本发明’等专利文书措辞本身不是技术事实，不能因论文没有这些字而判不支持。"
        "只返回 JSON 对象。"
    )
    user = f"""逐个检查下列 unit。每个 unit 都必须出现在返回结果中；将一个 unit 内的不同技术主张拆开。
claim 必须逐字摘自对应 unit，不能改写。一个主张若跨多个引句才能成立，可列多个 evidence。
verdict 只能是 supported（引文直接支持全部内容）、partial（只支持一部分）、
unsupported（与原文矛盾或原文没有相关内容）、uncertain（语义或原文提取不清楚）。
支持时必须给出 source_id 和该块中连续、逐字相同的论文原文 quote，保持原文语言；quote 要包含主张的关键数值与限定条件。
不要只凭主题相似就判 supported。无证据时 evidence 返回空数组。reason 用一句中文说明具体差异。
JSON 格式：{{"items":[{{"unit_id":"U0001","claims":[{{"claim":"草稿中的原句片段","verdict":"supported","evidence":[{{"source_id":"B0001","quote":"exact source words"}}],"reason":"..."}}]}}]}}。
待核对：{json.dumps(prompts, ensure_ascii=False)}
原文证据：{json.dumps(evidence, ensure_ascii=False)}"""
    response = model.ask_json(system, user, max_tokens=6000)
    raw_items = response.get("items")
    if not isinstance(raw_items, list):
        raw_items = []
    by_unit: dict[str, list[dict]] = {}
    for item in raw_items:
        if not isinstance(item, dict) or not isinstance(item.get("claims"), list):
            continue
        key = str(item.get("unit_id") or "")
        if key in {unit.id for unit in regular}:
            by_unit.setdefault(key, []).extend(x for x in item["claims"] if isinstance(x, dict))

    for unit in regular:
        claims = by_unit.get(unit.id, [])
        if not claims:
            findings.append(_unchecked_finding(
                unit, start_ordinal + len(findings), "模型遗漏了这条原句，尚未完成证据核对。"
            ))
            continue
        seen: set[str] = set()
        for claim in claims:
            key = _normalize(str(claim.get("claim") or ""))
            if key in seen:
                continue
            seen.add(key)
            findings.append(_validate_claim(unit, claim, source, start_ordinal + len(findings)))
    return sorted(findings, key=lambda finding: (int(finding.unit_id[1:]), int(finding.id[1:])))


def build_evidence_report(
    source: SourceDocument, draft: PatentDraft, model: ModelClient,
    existing: EvidenceReport | None = None,
    checkpoint: Callable[[EvidenceReport], None] | None = None,
) -> EvidenceReport:
    fingerprint = draft_fingerprint(draft)
    blocks_fingerprint = source_blocks_fingerprint(source)
    if existing and (
        existing.source_sha256 != source.sha256
        or existing.source_blocks_sha256 != blocks_fingerprint
        or existing.draft_sha256 != fingerprint
    ):
        raise ValueError("已保存的证据核对与当前论文或草稿不匹配。")
    units = split_units(draft)
    grouped: dict[str, list[EvidenceUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.paragraph_key, []).append(unit)
    done = list(existing.completed_paragraphs) if existing else []
    report = EvidenceReport(
        source_sha256=source.sha256,
        source_blocks_sha256=blocks_fingerprint,
        draft_sha256=fingerprint,
        model=getattr(getattr(model, "config", None), "model", "test-model"),
        total_units=len(units),
        checked_units=sum(len(grouped[key]) for key in done if key in grouped),
        completed_paragraphs=done,
        findings=list(existing.findings) if existing else [],
        warnings=list(existing.warnings) if existing else [],
    )
    for key, paragraph_units in grouped.items():
        if key in report.completed_paragraphs:
            continue
        findings = review_paragraph(paragraph_units, source, model, len(report.findings) + 1)
        report.findings.extend(findings)
        report.completed_paragraphs.append(key)
        report.checked_units += len(paragraph_units)
        if checkpoint:
            checkpoint(report)
    return report
