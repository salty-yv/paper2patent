from __future__ import annotations

import json
import httpx
import pymupdf
from fastapi.testclient import TestClient

from patent_writer import run_store
from patent_writer.drafting import audit_draft, build_draft
from patent_writer.graph import build_graph, merge_candidates
from patent_writer import model_api
from patent_writer.model_api import _extract_json
from patent_writer.model_api import ModelConfig
from patent_writer.pdf_extract import extract_pdf
from patent_writer.planning import build_plan, check_plan
from patent_writer.schema import ConceptEdge, ConceptNode, SourceBlock, SourceDocument
from patent_writer.web import app


def make_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((45, 70), "Channel grouping for sensor anomaly detection", fontsize=17)
    page.insert_text((45, 120), "Sensor anomalies are weak and ordinary filtering loses important channels.")
    page.insert_text((45, 145), "Our synthetic channel grouping reduces parameters by 37.25%.")
    page.insert_text((45, 170), "The method groups features and reconstructs important channels.")
    content = doc.tobytes()
    doc.close()
    return content


def source_fixture() -> SourceDocument:
    return SourceDocument(
        filename="sample.pdf", sha256="abc", title="Channel grouping", page_count=1,
        blocks=[
            SourceBlock(id="B0001", page=1, text="传感器异常检测属于信号处理领域，用于识别时序数据中幅值较弱且持续时间较短的异常。", bbox=(1, 1, 2, 2)),
            SourceBlock(id="B0002", page=1, text="普通通道筛选仅依据幅值判断重要性，可能丢失包含微弱异常信息的通道，造成识别性能下降。", bbox=(1, 3, 2, 4)),
            SourceBlock(id="B0003", page=1, text="提出多尺度通道分组方法，通过分组稀疏约束降低模型复杂度；在所述实验条件下，模型参数量减少37.25%。", bbox=(1, 5, 2, 6)),
            SourceBlock(id="B0004", page=1, text="将输入特征按尺度分组，依据分组系数判断通道重要性，并在筛选过程中重构仍包含异常信息的重要通道。", bbox=(1, 7, 2, 8)),
        ],
    )


class FakeModel:
    def usage(self) -> dict:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def ask_json(self, system: str, user: str, max_tokens: int = 4096) -> dict:
        if "请为中文专利说明书准备技术概念" in user:
            return {
                "patent_title": "一种传感器异常检测的通道分组方法",
                "items": [
                    {"category": "Field", "text": "传感器异常检测", "source_ids": ["B0001"]},
                    {"category": "PriorArt", "text": "普通通道筛选", "source_ids": ["B0002"]},
                    {"category": "TechProblem", "text": "筛选过程丢失微弱异常信息", "source_ids": ["B0002"]},
                    {"category": "Solution", "text": "多尺度通道分组", "source_ids": ["B0003"]},
                    {"category": "Effects", "text": "参数减少37.25%", "source_ids": ["B0003"]},
                    {"category": "Implementation", "text": "分组输入特征并重构通道", "source_ids": ["B0004"]},
                ],
            }
        if "重点寻找" in user:
            return {"edges": [{"source": "P1", "target": "S1", "relation": "solves"}]}
        if "将所有节点安排" in user:
            return {"sections": [
                {"heading": "技术领域", "node_ids": ["F1"]},
                {"heading": "背景技术", "node_ids": ["A1", "P1"]},
                {"heading": "发明内容", "node_ids": ["S1", "E1"]},
                {"heading": "具体实施方式", "node_ids": ["I1"]},
            ]}
        if "请写中国发明专利说明书草稿中的" in user:
            if "“技术领域”" in user:
                return {"paragraphs": [{"text": "本发明涉及传感器异常检测领域。", "node_ids": ["F1"], "source_ids": ["B0001"]}], "questions": []}
            if "“背景技术”" in user:
                return {"paragraphs": [{"text": "普通通道筛选易丢失微弱异常信息。", "node_ids": ["A1", "P1"], "source_ids": ["B0002"]}], "questions": []}
            if "“发明内容”" in user:
                return {"paragraphs": [{"text": "本发明采用多尺度通道分组，参数减少37.25%。", "node_ids": ["S1", "E1"], "source_ids": ["B0003"]}], "questions": []}
            return {"paragraphs": [{"text": "将输入特征按尺度分组，并重构重要通道。", "node_ids": ["I1"], "source_ids": ["B0004"]}], "questions": []}
        if "待核对内容" in user:
            return {"issues": []}
        raise AssertionError("Unexpected prompt")


def test_pdf_extraction_and_local_upload(tmp_path, monkeypatch):
    content = make_pdf()
    source = extract_pdf(content, "sample.pdf")
    assert source.page_count == 1
    assert "Channel grouping" in source.title
    assert "37.25%" in " ".join(b.text for b in source.blocks)
    monkeypatch.setattr(run_store, "RUNS", tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/runs", files={"file": ("sample.pdf", content, "application/pdf")})
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        assert client.get(f"/api/runs/{run_id}/data/source").json()["page_count"] == 1
        assert client.get(f"/api/runs/{run_id}/status").json()["state"] == "done"


def test_graph_plan_and_draft_preserve_sources():
    source = source_fixture()
    model = FakeModel()
    graph = build_graph(source, model)
    assert {n.category for n in graph.nodes} >= {"Field", "TechProblem", "Solution"}
    plan, candidates = build_plan(graph, model, candidate_count=3)
    check_plan(plan, graph)
    assert len(candidates) == 3
    draft = build_draft(source, graph, plan, model)
    assert len(draft.sections) == 4
    assert all(p.source_ids for s in draft.sections for p in s.paragraphs)
    assert not audit_draft(source, graph, plan, draft)


def test_conflicting_edges_and_cycle_are_resolved():
    nodes = [
        ConceptNode(id="A1", category="PriorArt", text="a", source_ids=["B0001"]),
        ConceptNode(id="P1", category="TechProblem", text="p", source_ids=["B0001"]),
        ConceptNode(id="S1", category="Solution", text="s", source_ids=["B0001"]),
    ]
    edges, warnings = merge_candidates(nodes, [
        [ConceptEdge(source="A1", target="P1", relation="causes")],
        [ConceptEdge(source="A1", target="P1", relation="causes"), ConceptEdge(source="P1", target="S1", relation="solves")],
        [ConceptEdge(source="A1", target="P1", relation="improves"), ConceptEdge(source="S1", target="A1", relation="validates")],
    ])
    assert any(e.source == "A1" and e.target == "P1" and e.relation == "causes" for e in edges)
    assert len(edges) == 2
    assert any("环" in warning for warning in warnings)


def test_model_json_parser_accepts_fences():
    assert _extract_json('```json\n{"ok":true}\n```') == {"ok": True}


def test_openai_compatible_adapter_without_external_network(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        assert request.headers["authorization"] == "Bearer test-only-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-flash"
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"items":[]}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        })

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(model_api.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    client = model_api.ModelClient(ModelConfig(api_key="test-only-key"))
    assert client.ask_json("system", "user") == {"items": []}
    assert client.usage() == {"calls": 1, "input_tokens": 10, "output_tokens": 4}


def test_full_local_stages_create_reviewable_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS", tmp_path)
    monkeypatch.setattr(run_store, "ModelClient", lambda config: FakeModel())
    run_id, _ = run_store.create_run(make_pdf(), "sample.pdf")
    config = ModelConfig(api_key="test-only-key")
    run_store.stage_graph(run_id, config)
    run_store.stage_plan(run_id, config)
    run_store.stage_draft(run_id, config)
    output = run_store.run_dir(run_id) / "output"
    assert (output / "说明书草稿.docx").is_file()
    assert (output / "证据对照表.csv").is_file()
    assert run_store.get_status(run_id)["state"] == "done"
    assert "test-only-key" not in (run_store.run_dir(run_id) / "status.json").read_text(encoding="utf-8")


def test_draft_resumes_after_section_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS", tmp_path)
    monkeypatch.setattr(run_store, "ModelClient", lambda config: FakeModel())
    run_id, _ = run_store.create_run(make_pdf(), "sample.pdf")
    config = ModelConfig(api_key="test-only-key")
    run_store.stage_graph(run_id, config)
    run_store.stage_plan(run_id, config)
    seen = {"field": 0, "failed": False}

    class FlakyModel(FakeModel):
        def ask_json(self, system: str, user: str, max_tokens: int = 4096) -> dict:
            if "请写中国发明专利说明书草稿中的“技术领域”" in user:
                seen["field"] += 1
            if "请写中国发明专利说明书草稿中的“背景技术”" in user and not seen["failed"]:
                seen["failed"] = True
                raise RuntimeError("temporary network failure")
            return super().ask_json(system, user, max_tokens)

    monkeypatch.setattr(run_store, "ModelClient", lambda config: FlakyModel())
    try:
        run_store.stage_draft(run_id, config)
    except RuntimeError:
        pass
    assert (run_store.run_dir(run_id) / "draft_partial.json").is_file()
    run_store.stage_draft(run_id, config)
    assert seen["field"] == 1
    assert (run_store.run_dir(run_id) / "output" / "说明书草稿.docx").is_file()
