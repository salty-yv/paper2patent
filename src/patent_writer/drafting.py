from __future__ import annotations

import re
from collections.abc import Callable

from .model_api import ModelClient
from .pdf_extract import expand_table_citations
from .schema import (
    ConceptGraph, DraftParagraph, DraftSection, PatentDraft, PatentPlan,
    PlanSection, SourceDocument,
)


_NUMBER = re.compile(r"(?<![A-Za-z0-9图第])\d+(?:\.\d+)?%?")
_FIGURE_REF = re.compile(r"(?:图|Figure|Fig\.?)\s*\d+", re.IGNORECASE)
_NUMBERED_LABEL = re.compile(r"(?:步骤|实施例|图|Figure|Fig\.?)\s*\d+", re.IGNORECASE)
_MATH = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
_RAW_MATH = re.compile(r"(?:\bmin_[A-Za-z]|\b[A-Za-z]+_[A-Za-z]|\^\{|Σ_\{|\\frac\b)")


def _section_prompt(
    source: SourceDocument, graph: ConceptGraph, section: PlanSection, previous: list[DraftSection]
) -> str:
    nodes = [graph.node_map()[node_id] for node_id in section.node_ids]
    # The invention summary restates the technical problems from Background.
    # Include their provenance even when the plan assigned those nodes there.
    reference_nodes = nodes + (
        [node for node in graph.nodes if node.category == "TechProblem" and node.id not in section.node_ids]
        if section.heading == "发明内容" else []
    )
    evidence_ids = expand_table_citations(
        source, list(dict.fromkeys(sid for node in reference_nodes for sid in node.source_ids))
    )
    source_map = source.block_map()
    excerpts = [
        {"id": sid, "page": source_map[sid].page, "text": source_map[sid].text[:1200]}
        for sid in evidence_ids if sid in source_map
    ]
    earlier = "\n".join(
        f"{s.heading}：{' '.join(p.text for p in s.paragraphs)[:900]}" for s in previous
    )[-2600:]
    requirements = {
        "技术领域": "简洁明确地写技术领域，避免写实验成绩。",
        "背景技术": "说明论文实际提及的现有技术及技术问题。来源不足时提出待确认项，不杜撰具体对比文件。",
        "发明内容": "依次写技术问题、解决问题的技术方案、有依据的有益效果。数值必须与原文一致。",
        "附图说明": "仅说明已选择并核对过的论文附图，不添加新图号。",
        "具体实施方式": "分段展开可实施的步骤、模块、参数和实施例；论文本身没有提供的细节列入待确认项。",
    }[section.heading]
    return f"""请写中国发明专利说明书草稿中的“{section.heading}”章节。{requirements}
保持中文技术表述清楚、可复核。不得创造不存在的试验、数值、先前技术、附图或替代实施例；
不要写“如权利要求……所述”。遇到必要但论文缺失的信息，在 questions 中列出，不要填入正文。
能从原文准确核对的数学公式，必须用 $...$ 包围 LaTeX 数学内容；较长公式可用 $$...$$ 独立成行。
不要把下标、上标、求和式写成未排版的普通字符。若原文提取不足以确认公式，改为文字说明并在 questions 中标明待核对的原论文页码。
每段必须有至少一个 node_id 和 source_id，且只可引用本次提供的编号。发明内容的技术问题应引用相应 TechProblem 概念及其原文，不要只引用贡献总结。
表格中的数值必须同时引用数据行及表题、列标题。一个段落包含不同主张时，应列出支持各主张的所有原文块。
返回 JSON：
{{"paragraphs":[{{"text":"完整中文段落","node_ids":["S1"],"source_ids":["B0001"]}}],"questions":["待发明人确认的问题"]}}。
章节概念：{[n.model_dump() for n in reference_nodes]}
原文证据：{excerpts}
之前章节摘要：{earlier}"""


def build_draft(
    source: SourceDocument, graph: ConceptGraph, plan: PatentPlan, model: ModelClient,
    existing: PatentDraft | None = None,
    checkpoint: Callable[[PatentDraft], None] | None = None,
) -> PatentDraft:
    sections: list[DraftSection] = list(existing.sections) if existing else []
    questions: list[str] = list(existing.questions) if existing else []
    warnings: list[str] = list(existing.warnings) if existing else []
    if [section.heading for section in sections] != [section.heading for section in plan.sections[:len(sections)]]:
        raise ValueError("已保存的章节与当前提纲不一致，无法从断点继续。")
    for section in plan.sections[len(sections):]:
        if not section.node_ids:
            questions.append(f"“{section.heading}”缺少论文证据，需要补充后撰写。")
            sections.append(DraftSection(heading=section.heading, paragraphs=[]))
            if checkpoint:
                checkpoint(PatentDraft(title=graph.patent_title, sections=sections, questions=questions, warnings=warnings))
            continue
        system = (
            "你是中文技术文书起草助手。根据给定概念和原文证据写专利说明书草稿，"
            "保留技术细节，不捏造事实。只返回 JSON 对象。"
        )
        data = model.ask_json(system, _section_prompt(source, graph, section, sections), max_tokens=4300)
        paragraphs: list[DraftParagraph] = []
        allowed_nodes = set(section.node_ids)
        if section.heading == "发明内容":
            allowed_nodes.update(node.id for node in graph.nodes if node.category == "TechProblem")
        allowed_sources = set(expand_table_citations(source, [
            sid for node_id in allowed_nodes for sid in graph.node_map()[node_id].source_ids
        ]))
        for raw in data.get("paragraphs", []):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            node_ids = [str(x) for x in raw.get("node_ids", []) if str(x) in allowed_nodes]
            source_ids = [str(x) for x in raw.get("source_ids", []) if str(x) in allowed_sources]
            if not text or not node_ids or not source_ids:
                warnings.append(f"“{section.heading}”有一段缺少有效概念或原文编号，已略过。")
                continue
            paragraphs.append(
                DraftParagraph(
                    text=text,
                    node_ids=list(dict.fromkeys(node_ids)),
                    source_ids=expand_table_citations(source, source_ids),
                )
            )
        if not paragraphs:
            raise ValueError(f"“{section.heading}”没有得到可追溯的有效段落；请重试该阶段。")
        questions.extend(str(q).strip() for q in data.get("questions", []) if str(q).strip())
        sections.append(DraftSection(heading=section.heading, paragraphs=paragraphs))
        if checkpoint:
            checkpoint(PatentDraft(title=graph.patent_title, sections=sections, questions=questions, warnings=warnings))
    draft = PatentDraft(
        title=graph.patent_title,
        sections=sections,
        questions=list(dict.fromkeys(questions)),
        warnings=warnings,
    )
    draft.warnings.extend(audit_draft(source, graph, plan, draft))
    source_map = source.block_map()
    for section in draft.sections:
        pages = sorted({
            source_map[sid].page
            for paragraph in section.paragraphs if _MATH.search(paragraph.text)
            for sid in paragraph.source_ids if sid in source_map
        })
        if pages:
            draft.questions.append(
                f"请对照原论文第 {'、'.join(map(str, pages))} 页核对“{section.heading}”中的数学公式及符号定义。"
            )
    draft.questions = list(dict.fromkeys(draft.questions))
    try:
        draft.warnings.extend(semantic_audit(source, draft, model))
    except Exception as exc:
        draft.warnings.append(f"语义证据核对暂未完成（{type(exc).__name__}），请人工复核正文与原文。")
    return draft


def semantic_audit(source: SourceDocument, draft: PatentDraft, model: ModelClient) -> list[str]:
    source_map = source.block_map()
    passages = []
    for section in draft.sections:
        for index, paragraph in enumerate(section.paragraphs, 1):
            passages.append({
                "heading": section.heading,
                "paragraph": index,
                "text": paragraph.text,
                "evidence": [
                    {"id": sid, "text": source_map[sid].text[:900]}
                    for sid in paragraph.source_ids if sid in source_map
                ],
            })
    if not passages:
        return []
    system = (
        "你是独立的技术事实核对员。只核对专利草稿的技术主张是否得到所引论文原文的支持，"
        "尤其检查算法步骤、数值、因果关系和效果。不要把一般写作风格当成问题。只输出 JSON。"
    )
    user = (
        "对每段检查正文与列出的证据。若没有实质性问题，返回 {\"issues\":[]}。"
        "若有不被证据支持的主张，返回 {\"issues\":[{\"heading\":\"发明内容\","
        "\"paragraph\":1,\"reason\":\"具体问题\"}]}。最多列出 20 项。"
        f"\n待核对内容：{passages}"
    )
    data = model.ask_json(system, user, max_tokens=2800)
    valid = {(p["heading"], p["paragraph"]) for p in passages}
    warnings = []
    for item in data.get("issues", []):
        if not isinstance(item, dict):
            continue
        key = (item.get("heading"), item.get("paragraph"))
        reason = str(item.get("reason") or "").strip()
        if key in valid and reason:
            warnings.append(f"语义核对：{key[0]}第 {key[1]} 段：{reason[:180]}")
    return warnings


def audit_draft(
    source: SourceDocument, graph: ConceptGraph, plan: PatentPlan, draft: PatentDraft
) -> list[str]:
    findings: list[str] = []
    all_source = " ".join(block.text for block in source.blocks)
    all_source_numbers = re.sub(r"(?<=\d),(?=\d)", "", all_source)
    source_map = source.block_map()
    covered_nodes: set[str] = set()
    for section in draft.sections:
        for index, paragraph in enumerate(section.paragraphs, 1):
            covered_nodes.update(paragraph.node_ids)
            cited_text = " ".join(source_map[sid].text for sid in paragraph.source_ids if sid in source_map)
            cited_numbers = re.sub(r"(?<=\d),(?=\d)", "", cited_text)
            plain_text = _MATH.sub("", paragraph.text)
            numeric_text = _NUMBERED_LABEL.sub("", plain_text)
            for number in set(_NUMBER.findall(numeric_text)):
                if number not in all_source_numbers:
                    findings.append(f"{section.heading}第 {index} 段的数值“{number}”在论文提取文本中未找到。")
                elif number not in cited_numbers:
                    findings.append(f"{section.heading}第 {index} 段的数值“{number}”不在所引原文块内。")
            if _RAW_MATH.search(plain_text):
                findings.append(f"{section.heading}第 {index} 段含未排版的公式写法，请核对并转为 Word 公式。")
            if _FIGURE_REF.search(paragraph.text) and "附图说明" not in [s.heading for s in plan.sections]:
                findings.append(f"{section.heading}第 {index} 段提到附图，但本次没有选用附图。")
            if not paragraph.source_ids:
                findings.append(f"{section.heading}第 {index} 段没有原文依据。")
    missing = [n.id for n in graph.nodes if n.id not in covered_nodes and n.category != "Figure"]
    if missing:
        findings.append(f"以下概念未在正文段落中体现：{', '.join(missing)}。")
    if not any(s.heading == "具体实施方式" and s.paragraphs for s in draft.sections):
        findings.append("具体实施方式缺少可追溯内容。")
    return list(dict.fromkeys(findings))
