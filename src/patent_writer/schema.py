from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Category = Literal[
    "Field", "TechProblem", "PriorArt", "Novelty", "Solution",
    "Implementation", "Effects", "Embodiment", "Figure",
]
Relation = Literal["solves", "implements", "causes", "improves", "validates"]
Heading = Literal["技术领域", "背景技术", "发明内容", "附图说明", "具体实施方式"]

CATEGORIES: tuple[Category, ...] = (
    "Field", "TechProblem", "PriorArt", "Novelty", "Solution",
    "Implementation", "Effects", "Embodiment", "Figure",
)
HEADINGS: tuple[Heading, ...] = (
    "技术领域", "背景技术", "发明内容", "附图说明", "具体实施方式",
)


class SourceBlock(BaseModel):
    id: str
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    kind: Literal["text", "figure_caption", "table_caption"] = "text"


class SourceDocument(BaseModel):
    filename: str
    sha256: str
    title: str
    page_count: int
    blocks: list[SourceBlock]
    warnings: list[str] = Field(default_factory=list)

    def block_map(self) -> dict[str, SourceBlock]:
        return {block.id: block for block in self.blocks}


class ConceptNode(BaseModel):
    id: str
    category: Category
    text: str
    source_ids: list[str] = Field(default_factory=list)


class ConceptEdge(BaseModel):
    source: str
    target: str
    relation: Relation
    votes: int = 1


class ConceptGraph(BaseModel):
    patent_title: str = "专利说明书草稿"
    nodes: list[ConceptNode]
    edges: list[ConceptEdge]
    warnings: list[str] = Field(default_factory=list)

    def node_map(self) -> dict[str, ConceptNode]:
        return {node.id: node for node in self.nodes}


class PlanSection(BaseModel):
    heading: Heading
    node_ids: list[str]


class PatentPlan(BaseModel):
    sections: list[PlanSection]
    score: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class DraftParagraph(BaseModel):
    text: str
    node_ids: list[str]
    source_ids: list[str]


class DraftSection(BaseModel):
    heading: Heading
    paragraphs: list[DraftParagraph]


class PatentDraft(BaseModel):
    title: str
    sections: list[DraftSection]
    questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


EvidenceVerdict = Literal["supported", "partial", "unsupported", "uncertain"]


class EvidenceQuote(BaseModel):
    source_id: str
    page: int
    quote: str
    exact_match: bool


class EvidenceCandidate(BaseModel):
    source_id: str
    page: int
    reason: str


class EvidenceFinding(BaseModel):
    id: str
    unit_id: str
    heading: Heading
    paragraph: int
    claim: str
    verdict: EvidenceVerdict
    reason: str
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    page_review: Literal["unreviewed", "matches", "differs"] = "unreviewed"
    page_review_note: str = ""
    page_review_by: Literal["", "user", "codex"] = ""


class EvidenceReport(BaseModel):
    source_sha256: str
    source_blocks_sha256: str = ""
    draft_sha256: str
    model: str
    total_units: int
    checked_units: int
    completed_paragraphs: list[str] = Field(default_factory=list)
    findings: list[EvidenceFinding]
    warnings: list[str] = Field(default_factory=list)
