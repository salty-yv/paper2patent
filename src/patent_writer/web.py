from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .model_api import ModelConfig
from .run_store import (
    _status, add_draft_citations, create_run, get_status, load_draft, load_evidence, load_graph, load_plan, load_source,
    run_dir, save_draft, save_graph, save_page_review, save_plan, stage_draft, stage_evidence, stage_graph, stage_plan,
)
from .schema import ConceptGraph, Heading, PatentDraft, PatentPlan


app = FastAPI(title="论文转中文专利说明书", docs_url=None, redoc_url=None)
_POOL = ThreadPoolExecutor(max_workers=1)
_JOBS: dict[str, Future] = {}
_LOCK = Lock()
_HTML = Path(__file__).resolve().parent / "static" / "index.html"


class StageRequest(BaseModel):
    provider: Literal["openai", "ollama"] = "openai"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    candidate_count: int = Field(default=3, ge=1, le=5)
    include_figures: bool = False


class CitationUpdate(BaseModel):
    heading: Heading
    paragraph: int = Field(ge=1)
    source_ids: list[str] = Field(min_length=1, max_length=20)


class CitationRequest(BaseModel):
    updates: list[CitationUpdate] = Field(min_length=1, max_length=100)


class PageReviewRequest(BaseModel):
    finding_id: str
    decision: Literal["matches", "differs"]
    note: str = Field(default="", max_length=500)


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(_HTML, media_type="text/html; charset=utf-8")


@app.post("/api/runs")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "请选择 PDF 文件。")
    content = await file.read(30 * 1024 * 1024 + 1)
    try:
        run_id, source = create_run(content, file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "run_id": run_id,
        "title": source.title,
        "pages": source.page_count,
        "blocks": len(source.blocks),
        "warnings": source.warnings,
    }


@app.get("/api/runs/{run_id}/status")
def status(run_id: str) -> dict:
    try:
        return get_status(run_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/runs/{run_id}/data/{kind}")
def data(run_id: str, kind: Literal["source", "graph", "plan", "draft", "evidence"]) -> dict:
    loaders = {"source": load_source, "graph": load_graph, "plan": load_plan, "draft": load_draft, "evidence": load_evidence}
    try:
        return loaders[kind](run_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(404, "该阶段尚未生成。") from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.put("/api/runs/{run_id}/graph")
def update_graph(run_id: str, graph: ConceptGraph) -> dict:
    try:
        save_graph(run_id, graph)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@app.put("/api/runs/{run_id}/plan")
def update_plan(run_id: str, plan: PatentPlan) -> dict:
    try:
        save_plan(run_id, plan)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@app.post("/api/runs/{run_id}/draft/citations")
def update_draft_citations(run_id: str, request: CitationRequest) -> dict:
    try:
        draft = add_draft_citations(
            run_id, [(item.heading, item.paragraph, item.source_ids) for item in request.updates]
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "draft": draft.model_dump()}


@app.put("/api/runs/{run_id}/draft")
def update_draft(run_id: str, draft: PatentDraft) -> dict:
    try:
        saved = save_draft(run_id, draft)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "draft": saved.model_dump()}


@app.put("/api/runs/{run_id}/evidence/page-review")
def update_page_review(run_id: str, request: PageReviewRequest) -> dict:
    try:
        report = save_page_review(run_id, request.finding_id, request.decision, request.note)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "evidence": report.model_dump()}


@app.get("/api/runs/{run_id}/source.pdf")
def source_pdf(run_id: str) -> FileResponse:
    try:
        path = run_dir(run_id) / "input.pdf"
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="application/pdf", filename="论文原文.pdf", content_disposition_type="inline")


def _do_stage(run_id: str, stage: str, request: StageRequest) -> None:
    config = ModelConfig(
        provider=request.provider, model=request.model,
        base_url=request.base_url, api_key=request.api_key,
    )
    try:
        if stage == "graph":
            stage_graph(run_id, config, include_figures=request.include_figures)
        elif stage == "plan":
            stage_plan(run_id, config, candidate_count=request.candidate_count)
        elif stage == "draft":
            stage_draft(run_id, config)
        else:
            stage_evidence(run_id, config)
    except Exception as exc:
        # stage_* normally writes its own error; this catches validation errors before it starts.
        current = get_status(run_id)
        if current.get("state") != "error":
            _status(run_dir(run_id), stage, "error", str(exc)[:400])


@app.post("/api/runs/{run_id}/start/{stage}")
def start_stage(run_id: str, stage: Literal["graph", "plan", "draft", "evidence"], request: StageRequest) -> dict:
    try:
        directory = run_dir(run_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        ModelConfig(
            provider=request.provider, model=request.model,
            base_url=request.base_url, api_key=request.api_key,
        ).validate()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if stage == "plan" and not (directory / "graph.json").exists():
        raise HTTPException(400, "请先生成并审核概念图。")
    if stage == "draft" and not (directory / "plan.json").exists():
        raise HTTPException(400, "请先生成并审核提纲。")
    if stage == "evidence" and not (directory / "draft.json").exists():
        raise HTTPException(400, "请先生成说明书草稿。")
    with _LOCK:
        current = _JOBS.get(run_id)
        if current and not current.done():
            raise HTTPException(409, "这一篇论文已有任务正在运行。")
        _status(directory, stage, "queued", "已排队，准备调用模型")
        _JOBS[run_id] = _POOL.submit(_do_stage, run_id, stage, request)
    return {"ok": True, "stage": stage}


@app.get("/api/runs/{run_id}/download/{kind}")
def download(run_id: str, kind: Literal["docx", "csv", "markdown", "review", "evidence_csv", "evidence_json"]) -> FileResponse:
    files = {
        "docx": "说明书草稿.docx",
        "csv": "证据对照表.csv",
        "markdown": "说明书草稿.md",
        "review": "待确认事项.json",
        "evidence_csv": "逐条证据核对.csv",
        "evidence_json": "逐条证据核对.json",
    }
    try:
        target = run_dir(run_id) / "output" / files[kind]
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    if not target.is_file():
        raise HTTPException(404, "草稿尚未生成。")
    if kind in ("evidence_csv", "evidence_json"):
        try:
            load_evidence(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, "证据核对结果已过期，请重新核对。") from exc
    return FileResponse(target, filename=target.name)
