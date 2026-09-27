from __future__ import annotations

import json
import hashlib
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .drafting import build_draft
from .evidence_candidates import attach_candidate_sources
from .evidence_review import build_evidence_report, draft_fingerprint, source_blocks_fingerprint
from .export import export_docx, export_evidence_report, export_review_files
from .graph import build_graph, check_graph
from .model_api import ModelClient, ModelConfig
from .pdf_extract import extract_pdf
from .planning import build_plan, check_plan
from .schema import ConceptGraph, EvidenceReport, Heading, PatentDraft, PatentPlan, SourceDocument


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def run_dir(run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("运行编号无效。")
    path = RUNS / run_id
    if not path.is_dir():
        raise FileNotFoundError("找不到这次运行。")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(path: Path, stage: str, state: str, message: str = "", usage: dict[str, int] | None = None) -> None:
    value = {
        "stage": stage,
        "state": state,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if usage is not None:
        value["usage"] = usage
    _write_json(path / "status.json", value)


def create_run(content: bytes, filename: str) -> tuple[str, SourceDocument]:
    if len(content) > 30 * 1024 * 1024:
        raise ValueError("PDF 超过 30 MB 的当前上传上限。")
    source = extract_pdf(content, Path(filename).name)
    run_id = uuid.uuid4().hex
    directory = RUNS / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "input.pdf").write_bytes(content)
    _write_json(directory / "source.json", source.model_dump())
    _status(directory, "parse", "done", "论文读取完成")
    return run_id, source


def load_source(run_id: str) -> SourceDocument:
    return SourceDocument.model_validate(_read_json(run_dir(run_id) / "source.json"))


def load_graph(run_id: str) -> ConceptGraph:
    return ConceptGraph.model_validate(_read_json(run_dir(run_id) / "graph.json"))


def load_plan(run_id: str) -> PatentPlan:
    return PatentPlan.model_validate(_read_json(run_dir(run_id) / "plan.json"))


def load_draft(run_id: str) -> PatentDraft:
    return PatentDraft.model_validate(_read_json(run_dir(run_id) / "draft.json"))


def save_draft(run_id: str, draft: PatentDraft) -> PatentDraft:
    directory = run_dir(run_id)
    original = load_draft(run_id)
    source_ids = set(load_source(run_id).block_map())
    node_ids = set(load_graph(run_id).node_map())
    if [section.heading for section in draft.sections] != [section.heading for section in original.sections]:
        raise ValueError("草稿章节不能在此步骤中调整。")
    if any(len(new.paragraphs) != len(old.paragraphs) for new, old in zip(draft.sections, original.sections)):
        raise ValueError("草稿段落数量不能在此步骤中调整。")
    for section in draft.sections:
        for paragraph in section.paragraphs:
            if not paragraph.text.strip() or not paragraph.source_ids or not paragraph.node_ids:
                raise ValueError("每段必须有正文、概念和原文依据。")
            if any(sid not in source_ids for sid in paragraph.source_ids):
                raise ValueError("草稿含无效原文块编号。")
            if any(nid not in node_ids for nid in paragraph.node_ids):
                raise ValueError("草稿含无效概念编号。")
    _archive_downstream(directory, ["draft.json", "evidence.json", "evidence_partial.json", "output"])
    _write_json(directory / "draft.json", draft.model_dump())
    output = directory / "output"
    output.mkdir(exist_ok=True)
    export_docx(draft, output / "说明书草稿.docx")
    export_review_files(draft, load_source(run_id), output)
    _status(directory, "draft", "done", "草稿已保存，请重新核对证据")
    return draft


def add_draft_citations(run_id: str, updates: list[tuple[Heading, int, list[str]]]) -> PatentDraft:
    """Append reviewer-selected source blocks and invalidate the prior audit."""
    directory = run_dir(run_id)
    source, draft = load_source(run_id), load_draft(run_id)
    valid_sources = set(source.block_map())
    sections = {section.heading: section for section in draft.sections}
    changed = False
    for heading, paragraph_number, source_ids in updates:
        section = sections.get(heading)
        if section is None or not 1 <= paragraph_number <= len(section.paragraphs):
            raise ValueError("引用对应的章节或段落不存在。")
        if not source_ids or any(sid not in valid_sources for sid in source_ids):
            raise ValueError("候选原文块编号无效。")
        paragraph = section.paragraphs[paragraph_number - 1]
        combined = list(dict.fromkeys([*paragraph.source_ids, *source_ids]))
        if combined != paragraph.source_ids:
            paragraph.source_ids = combined
            changed = True
    if not changed:
        return draft
    return save_draft(run_id, draft)


def load_evidence(run_id: str) -> EvidenceReport:
    report = EvidenceReport.model_validate(_read_json(run_dir(run_id) / "evidence.json"))
    source = load_source(run_id)
    draft = load_draft(run_id)
    if (
        report.source_sha256 != source.sha256
        or report.source_blocks_sha256 != source_blocks_fingerprint(source)
        or report.draft_sha256 != draft_fingerprint(draft)
    ):
        raise ValueError("证据核对结果与当前草稿不一致，请重新核对。")
    return report


def save_page_review(run_id: str, finding_id: str, decision: str, note: str = "") -> EvidenceReport:
    report = load_evidence(run_id)
    if decision not in ("matches", "differs"):
        raise ValueError("请选择页面核对结果。")
    if decision == "differs" and not note.strip():
        raise ValueError("发现差异时请写明具体差异。")
    finding = next((item for item in report.findings if item.id == finding_id), None)
    if finding is None or "formula_visual_review" not in finding.checks:
        raise ValueError("只能对需要页面核对的公式记录结果。")
    finding.page_review = decision
    finding.page_review_note = note.strip()[:500]
    finding.page_review_by = "user"
    directory = run_dir(run_id)
    _write_json(directory / "evidence.json", report.model_dump())
    export_evidence_report(report, directory / "output")
    _status(directory, "evidence", "done", "页面核对结果已保存")
    return report


def get_status(run_id: str) -> dict[str, Any]:
    return _read_json(run_dir(run_id) / "status.json")


def _archive_downstream(directory: Path, names: list[str]) -> None:
    existing = [directory / name for name in names if (directory / name).exists()]
    if not existing:
        return
    archive = directory / "history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))


def save_graph(run_id: str, graph: ConceptGraph) -> None:
    directory = run_dir(run_id)
    check_graph(graph, load_source(run_id))
    graph.warnings = [w for w in graph.warnings if not w.startswith("缺少 ")]
    missing = _missing_core(graph)
    graph.warnings.extend(f"缺少 {category} 技术概念，需要人工补充。" for category in missing)
    _archive_downstream(directory, ["graph.json", "plan.json", "plan_candidates.json", "draft.json", "draft_partial.json", "evidence.json", "evidence_partial.json", "output"])
    _write_json(directory / "graph.json", graph.model_dump())
    _status(directory, "graph", "done", "概念图已保存；后续阶段需重新生成")


def save_plan(run_id: str, plan: PatentPlan) -> None:
    directory = run_dir(run_id)
    check_plan(plan, load_graph(run_id))
    _archive_downstream(directory, ["plan.json", "plan_candidates.json", "draft.json", "draft_partial.json", "evidence.json", "evidence_partial.json", "output"])
    _write_json(directory / "plan.json", plan.model_dump())
    _status(directory, "plan", "done", "提纲已保存；正文需重新生成")


def stage_graph(run_id: str, config: ModelConfig, include_figures: bool = False) -> None:
    directory = run_dir(run_id)
    client = ModelClient(config)
    _status(directory, "graph", "running", "正在提取技术概念和关系")
    try:
        graph = build_graph(load_source(run_id), client, include_figures)
        check_graph(graph, load_source(run_id))
        _archive_downstream(directory, ["graph.json", "plan.json", "plan_candidates.json", "draft.json", "draft_partial.json", "evidence.json", "evidence_partial.json", "output"])
        _write_json(directory / "graph.json", graph.model_dump())
        _status(directory, "graph", "done", "概念图已生成，请审核", client.usage())
    except Exception as exc:
        _status(directory, "graph", "error", str(exc)[:400], client.usage())
        raise


def stage_plan(run_id: str, config: ModelConfig, candidate_count: int = 3) -> None:
    directory = run_dir(run_id)
    client = ModelClient(config)
    _status(directory, "plan", "running", "正在规划中文说明书章节")
    try:
        graph = load_graph(run_id)
        missing = _missing_core(graph)
        if missing:
            raise ValueError(f"概念图缺少 {', '.join(missing)}；请先补充有原文依据的内容。")
        plan, candidates = build_plan(
            graph, client, include_figures=any(n.category == "Figure" for n in graph.nodes),
            candidate_count=candidate_count,
        )
        check_plan(plan, graph)
        _archive_downstream(directory, ["plan.json", "plan_candidates.json", "draft.json", "draft_partial.json", "evidence.json", "evidence_partial.json", "output"])
        _write_json(directory / "plan.json", plan.model_dump())
        _write_json(directory / "plan_candidates.json", [c.model_dump() for c in candidates])
        _status(directory, "plan", "done", "提纲已生成，请审核", client.usage())
    except Exception as exc:
        _status(directory, "plan", "error", str(exc)[:400], client.usage())
        raise


def _missing_core(graph: ConceptGraph) -> list[str]:
    present = {node.category for node in graph.nodes}
    return [category for category in ("Field", "TechProblem", "Solution") if category not in present]


def stage_draft(run_id: str, config: ModelConfig) -> None:
    directory = run_dir(run_id)
    client = ModelClient(config)
    _status(directory, "draft", "running", "正在分章节起草与核查")
    try:
        source, graph, plan = load_source(run_id), load_graph(run_id), load_plan(run_id)
        plan_hash = hashlib.sha256(
            json.dumps(plan.model_dump(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        partial_path = directory / "draft_partial.json"
        existing = None
        if partial_path.is_file():
            partial = _read_json(partial_path)
            if partial.get("plan_hash") == plan_hash:
                existing = PatentDraft.model_validate(partial["draft"])
            else:
                _archive_downstream(directory, ["draft_partial.json"])

        def checkpoint(partial_draft: PatentDraft) -> None:
            _write_json(partial_path, {"plan_hash": plan_hash, "draft": partial_draft.model_dump()})
            _status(
                directory, "draft", "running",
                f"已完成 {len(partial_draft.sections)}/{len(plan.sections)} 个章节，正在继续起草",
                client.usage(),
            )

        draft = build_draft(source, graph, plan, client, existing=existing, checkpoint=checkpoint)
        _archive_downstream(directory, ["draft.json", "evidence.json", "evidence_partial.json", "output"])
        output = directory / "output"
        output.mkdir(exist_ok=True)
        _write_json(directory / "draft.json", draft.model_dump())
        export_docx(draft, output / "说明书草稿.docx")
        export_review_files(draft, source, output)
        _status(directory, "draft", "done", "草稿已生成，请查看待确认事项", client.usage())
    except Exception as exc:
        _status(directory, "draft", "error", str(exc)[:400], client.usage())
        raise


def stage_evidence(run_id: str, config: ModelConfig) -> None:
    directory = run_dir(run_id)
    client = ModelClient(config)
    _status(directory, "evidence", "running", "正在逐条核对草稿与论文原文")
    try:
        source, draft = load_source(run_id), load_draft(run_id)
        partial_path = directory / "evidence_partial.json"
        # A completed report is a finished audit. A new request starts a new
        # audit; only an interrupted partial report is resumed.
        previous_page_reviews: dict[tuple[str, int, str], tuple[str, str, str]] = {}
        if (directory / "evidence.json").is_file():
            try:
                previous = load_evidence(run_id)
                previous_page_reviews = {
                    (item.heading, item.paragraph, item.claim): (
                        item.page_review, item.page_review_note, item.page_review_by,
                    )
                    for item in previous.findings if item.page_review != "unreviewed"
                }
            except ValueError:
                pass
            _archive_downstream(directory, [
                "evidence.json", "evidence_partial.json",
                "output/逐条证据核对.csv", "output/逐条证据核对.json",
            ])
        existing = None
        if partial_path.is_file():
            partial = EvidenceReport.model_validate(_read_json(partial_path))
            if (
                partial.source_sha256 == source.sha256
                and partial.source_blocks_sha256 == source_blocks_fingerprint(source)
                and partial.draft_sha256 == draft_fingerprint(draft)
            ):
                existing = partial
            else:
                _archive_downstream(directory, ["evidence_partial.json"])

        def checkpoint(report: EvidenceReport) -> None:
            _write_json(partial_path, report.model_dump())
            _status(
                directory, "evidence", "running",
                f"已核对 {report.checked_units}/{report.total_units} 条原句",
                client.usage(),
            )

        report = build_evidence_report(source, draft, client, existing=existing, checkpoint=checkpoint)
        for item in report.findings:
            if "formula_visual_review" in item.checks:
                prior = previous_page_reviews.get((item.heading, item.paragraph, item.claim))
                if prior:
                    item.page_review, item.page_review_note, item.page_review_by = prior
        if (directory / "graph.json").is_file():
            attach_candidate_sources(report, source, draft, load_graph(run_id))
        _write_json(directory / "evidence.json", report.model_dump())
        export_evidence_report(report, directory / "output")
        _status(
            directory, "evidence", "done",
            f"证据核对完成：{len(report.findings)} 条主张",
            client.usage(),
        )
    except Exception as exc:
        _status(directory, "evidence", "error", str(exc)[:400], client.usage())
        raise
