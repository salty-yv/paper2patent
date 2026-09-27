from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from patent_writer import run_store
from patent_writer.evidence_candidates import attach_candidate_sources
from patent_writer.evidence_review import _numbers, build_evidence_report, draft_fingerprint, split_units
from patent_writer.model_api import ModelConfig
from patent_writer.pdf_extract import expand_table_citations
from patent_writer.schema import (
    ConceptGraph, ConceptNode, DraftParagraph, DraftSection, EvidenceFinding,
    EvidenceReport, PatentDraft, SourceBlock, SourceDocument,
)
from patent_writer.web import app


def source_sample() -> SourceDocument:
    return SourceDocument(
        filename="paper.pdf", sha256="source-hash", title="SyntheticStudy", page_count=2,
        blocks=[
            SourceBlock(
                id="B0001", page=1,
                text="The method reduces parameter count by 37.25% and FLOPs by 42.50%.",
                bbox=(1, 1, 3, 3),
            ),
            SourceBlock(
                id="B0002", page=2,
                text="We use channel grouping to preserve signal structure information.",
                bbox=(1, 1, 3, 3),
            ),
        ],
    )


def draft_sample() -> PatentDraft:
    return PatentDraft(
        title="测试草稿", sections=[DraftSection(
            heading="发明内容", paragraphs=[DraftParagraph(
                text="参数量减少37.25%。FLOPs减少60%。采用通道分组。",
                node_ids=["E1"], source_ids=["B0001", "B0002"],
            )],
        )],
    )


class ReviewModel:
    def __init__(self, response: dict):
        self.response = response
        self.calls = 0

    def ask_json(self, system: str, user: str, max_tokens: int = 4096) -> dict:
        self.calls += 1
        return self.response

    def usage(self) -> dict[str, int]:
        return {"calls": self.calls, "input_tokens": 0, "output_tokens": 0}


def test_claim_review_flags_unsupported_and_missing_units():
    source, draft = source_sample(), draft_sample()
    units = split_units(draft)
    assert len(units) == 3
    model = ReviewModel({"items": [
        {"unit_id": units[0].id, "claims": [{
            "claim": "参数量减少37.25%", "verdict": "supported",
            "evidence": [{"source_id": "B0001", "quote": "reduces parameter count by 37.25%"}],
            "reason": "原文一致",
        }]},
        {"unit_id": units[1].id, "claims": [{
            "claim": "FLOPs减少60%", "verdict": "unsupported",
            "evidence": [{"source_id": "B0001", "quote": "FLOPs by 42.50%"}],
            "reason": "论文为42.50%，草稿误写60%",
        }]},
    ]})
    report = build_evidence_report(source, draft, model)
    assert report.checked_units == report.total_units == 3
    assert [x.verdict for x in report.findings] == ["supported", "unsupported", "uncertain"]
    assert report.findings[0].evidence[0].exact_match
    assert report.findings[1].evidence[0].page == 1
    assert "遗漏" in report.findings[2].reason


def test_invalid_quote_and_number_fail_closed():
    source = source_sample()
    draft = PatentDraft(title="测试", sections=[DraftSection(
        heading="发明内容", paragraphs=[DraftParagraph(
            text="参数量减少37.25%。", node_ids=["E1"], source_ids=["B0001"]
        )]
    )])
    unit_id = split_units(draft)[0].id
    invalid_quote = ReviewModel({"items": [{"unit_id": unit_id, "claims": [{
        "claim": "参数量减少37.25%", "verdict": "supported",
        "evidence": [{"source_id": "B0001", "quote": "parameter count by 99%"}],
        "reason": "支持",
    }]}]})
    report = build_evidence_report(source, draft, invalid_quote)
    assert report.findings[0].verdict == "uncertain"
    assert "quote_not_found" in report.findings[0].checks

    number_missing = ReviewModel({"items": [{"unit_id": unit_id, "claims": [{
        "claim": "参数量减少37.25%", "verdict": "supported",
        "evidence": [{"source_id": "B0001", "quote": "The method reduces parameter count"}],
        "reason": "支持",
    }]}]})
    report = build_evidence_report(source, draft, number_missing)
    assert report.findings[0].verdict == "uncertain"
    assert "number_not_in_quote" in report.findings[0].checks


def test_formula_needs_visual_review_without_model_call():
    draft = PatentDraft(title="测试", sections=[DraftSection(
        heading="具体实施方式", paragraphs=[DraftParagraph(
            text="$$F_{SCR}=\\alpha F_{best}.$$", node_ids=["I1"], source_ids=["B0002"]
        )]
    )])
    model = ReviewModel({})
    report = build_evidence_report(source_sample(), draft, model)
    assert report.findings[0].verdict == "uncertain"
    assert "formula_visual_review" in report.findings[0].checks
    assert report.findings[0].evidence[0].page == 2
    assert model.calls == 0


def test_numeric_tokens_ignore_model_names_but_keep_units():
    assert _numbers("Dataset-1k、Model-A18；0.875G FLOPs，1,001张图像") == {"0.875G", "1001"}


def test_partial_report_resumes_without_rechecking_completed_paragraph():
    source = source_sample()
    draft = PatentDraft(title="测试", sections=[DraftSection(
        heading="发明内容", paragraphs=[
            DraftParagraph(text="参数量减少37.25%。", node_ids=["E1"], source_ids=["B0001"]),
            DraftParagraph(text="采用通道分组。", node_ids=["S1"], source_ids=["B0002"]),
        ],
    )])
    saved = []
    units = split_units(draft)

    class FlakyReview(ReviewModel):
        def ask_json(self, system: str, user: str, max_tokens: int = 4096) -> dict:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("connection lost")
            return {"items": [{"unit_id": units[0].id, "claims": [{
                "claim": "参数量减少37.25%", "verdict": "supported",
                "evidence": [{"source_id": "B0001", "quote": "reduces parameter count by 37.25%"}],
                "reason": "一致",
            }]}]}

    model = FlakyReview({})
    with pytest.raises(RuntimeError):
        build_evidence_report(source, draft, model, checkpoint=lambda report: saved.append(report.model_copy(deep=True)))
    assert saved[0].checked_units == 1

    resume = ReviewModel({"items": [{"unit_id": units[1].id, "claims": [{
        "claim": "采用通道分组", "verdict": "supported",
        "evidence": [{"source_id": "B0002", "quote": "We use channel grouping"}],
        "reason": "一致",
    }]}]})
    report = build_evidence_report(source, draft, resume, existing=saved[0])
    assert report.checked_units == report.total_units == 2
    assert len(report.findings) == 2
    assert resume.calls == 1


def test_stage_exports_review_and_invalidates_on_draft_change(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS", tmp_path)
    run_id = uuid.uuid4().hex
    directory = tmp_path / run_id
    directory.mkdir()
    source, draft = source_sample(), draft_sample()
    run_store._write_json(directory / "source.json", source.model_dump())
    run_store._write_json(directory / "draft.json", draft.model_dump())
    units = split_units(draft)
    model = ReviewModel({"items": [
        {"unit_id": unit.id, "claims": [{
            "claim": unit.text.strip("。"), "verdict": "unsupported",
            "evidence": [], "reason": "测试用的无依据结论",
        }]}
        for unit in units
    ]})
    monkeypatch.setattr(run_store, "ModelClient", lambda config: model)
    run_store.stage_evidence(run_id, ModelConfig(api_key="test-only-key"))
    report = run_store.load_evidence(run_id)
    assert report.checked_units == 3
    assert (directory / "output" / "逐条证据核对.csv").exists()
    with TestClient(app) as client:
        response = client.get(f"/api/runs/{run_id}/download/evidence_csv")
        assert response.status_code == 200
        assert client.get(f"/api/runs/{run_id}/data/evidence").status_code == 200
    draft.sections[0].paragraphs[0].text += "新增内容。"
    run_store._write_json(directory / "draft.json", draft.model_dump())
    with pytest.raises(ValueError, match="不一致"):
        run_store.load_evidence(run_id)
    assert report.draft_sha256 != draft_fingerprint(draft)


def test_candidate_search_recovers_problem_source_and_table_header():
    source = SourceDocument(
        filename="paper.pdf", sha256="hash", title="Test", page_count=2,
        blocks=[
            SourceBlock(id="B0001", page=1, text="Large models contain redundant parameters.", bbox=(1, 1, 2, 2)),
            SourceBlock(id="B0002", page=1, text="The proposed method uses pruning.", bbox=(1, 3, 2, 4)),
            SourceBlock(id="B0003", page=2, text="Table 1: Accuracy on Dataset A.", bbox=(1, 1, 2, 2), kind="table_caption"),
            SourceBlock(id="B0004", page=2, text="Dataset A IoU nIoU Pd Fa", bbox=(1, 3, 2, 4)),
            SourceBlock(id="B0005", page=2, text="Method 72.45 70.10 95.20 3.40", bbox=(1, 5, 2, 6)),
            SourceBlock(id="B0006", page=2, text="The result is IoU 72.45 and nIoU 70.10.", bbox=(1, 7, 2, 8)),
        ],
    )
    draft = PatentDraft(title="Test", sections=[DraftSection(heading="发明内容", paragraphs=[
        DraftParagraph(text="模型参数冗余。", node_ids=["N1"], source_ids=["B0002"]),
        DraftParagraph(text="数据集A的IoU为72.45。", node_ids=["M1"], source_ids=["B0005"]),
        DraftParagraph(text="IoU 72.45，nIoU 70.10。", node_ids=["N1"], source_ids=["B0002"]),
    ])])
    graph = ConceptGraph(nodes=[
        ConceptNode(id="P1", category="TechProblem", text="模型参数冗余", source_ids=["B0001"]),
        ConceptNode(id="N1", category="Novelty", text="剪枝方法", source_ids=["B0002"]),
        ConceptNode(id="M1", category="Embodiment", text="数据集A的IoU为72.45", source_ids=["B0005"]),
    ], edges=[])
    report = EvidenceReport(
        source_sha256=source.sha256, draft_sha256=draft_fingerprint(draft),
        model="test", total_units=3, checked_units=3,
        findings=[
            EvidenceFinding(id="C0001", unit_id="U0001", heading="发明内容", paragraph=1,
                            claim="模型参数冗余", verdict="unsupported", reason="当前引用不足"),
            EvidenceFinding(id="C0002", unit_id="U0002", heading="发明内容", paragraph=2,
                            claim="数据集A的IoU为72.45", verdict="partial", reason="缺少表头"),
            EvidenceFinding(id="C0003", unit_id="U0003", heading="发明内容", paragraph=3,
                            claim="IoU 72.45，nIoU 70.10", verdict="partial", reason="需要全文搜索"),
        ],
    )
    attach_candidate_sources(report, source, draft, graph)
    assert "B0001" in [item.source_id for item in report.findings[0].candidates]
    assert {"B0003", "B0004"} <= {item.source_id for item in report.findings[1].candidates}
    assert "B0006" in [item.source_id for item in report.findings[2].candidates]
    assert expand_table_citations(source, ["B0005"]) == ["B0005", "B0003", "B0004"]


def test_citation_patch_invalidates_old_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS", tmp_path)
    monkeypatch.setattr(run_store, "export_docx", lambda draft, path: path.write_bytes(b"docx"))
    run_id = uuid.uuid4().hex
    directory = tmp_path / run_id
    directory.mkdir()
    source, draft = source_sample(), draft_sample()
    draft.sections[0].paragraphs[0].source_ids = ["B0001"]
    run_store._write_json(directory / "source.json", source.model_dump())
    run_store._write_json(directory / "draft.json", draft.model_dump())
    run_store._write_json(directory / "graph.json", ConceptGraph(nodes=[
        ConceptNode(id="E1", category="Effects", text="测试", source_ids=["B0001"]),
    ], edges=[]).model_dump())
    run_store._write_json(directory / "evidence.json", {"old": True})
    with TestClient(app) as client:
        response = client.post(f"/api/runs/{run_id}/draft/citations", json={
            "updates": [{"heading": "发明内容", "paragraph": 1, "source_ids": ["B0002"]}]
        })
        assert response.status_code == 200
        assert response.json()["draft"]["sections"][0]["paragraphs"][0]["source_ids"] == ["B0001", "B0002"]
        assert client.get(f"/api/runs/{run_id}/data/evidence").status_code == 404
    assert (directory / "output" / "证据对照表.csv").exists()
    assert list((directory / "history").rglob("evidence.json"))


def test_formula_intro_requires_page_review_and_records_result(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS", tmp_path)
    run_id = uuid.uuid4().hex
    directory = tmp_path / run_id
    directory.mkdir()
    source = source_sample()
    draft = PatentDraft(title="测试", sections=[DraftSection(heading="具体实施方式", paragraphs=[
        DraftParagraph(text="插值系数如下：\n\n$$\\alpha=1.$$", node_ids=["I1"], source_ids=["B0001"]),
    ])])
    model = ReviewModel({})
    report = build_evidence_report(source, draft, model)
    assert model.calls == 0
    assert report.checked_units == 2
    assert all("formula_visual_review" in item.checks for item in report.findings)
    run_store._write_json(directory / "source.json", source.model_dump())
    run_store._write_json(directory / "draft.json", draft.model_dump())
    run_store._write_json(directory / "evidence.json", report.model_dump())
    (directory / "input.pdf").write_bytes(b"%PDF-1.4 sample")
    with TestClient(app) as client:
        response = client.put(f"/api/runs/{run_id}/evidence/page-review", json={
            "finding_id": report.findings[1].id, "decision": "matches",
        })
        assert response.status_code == 200
        pdf_response = client.get(f"/api/runs/{run_id}/source.pdf")
        assert pdf_response.status_code == 200
        assert pdf_response.content.startswith(b"%PDF")
    assert run_store.load_evidence(run_id).findings[1].page_review == "matches"
    assert "matches" in (directory / "output" / "逐条证据核对.csv").read_text(encoding="utf-8-sig")
