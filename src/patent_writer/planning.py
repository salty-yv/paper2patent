from __future__ import annotations

from .model_api import ModelClient
from .schema import HEADINGS, ConceptGraph, ConceptNode, PatentPlan, PlanSection


_DEFAULT = {
    "Field": "技术领域",
    "PriorArt": "背景技术",
    "TechProblem": "背景技术",
    "Novelty": "发明内容",
    "Solution": "发明内容",
    "Effects": "发明内容",
    "Figure": "附图说明",
    "Implementation": "具体实施方式",
    "Embodiment": "具体实施方式",
}
_ALLOWED = {
    "Field": {"技术领域"},
    "PriorArt": {"背景技术"},
    "TechProblem": {"背景技术", "发明内容"},
    "Novelty": {"发明内容"},
    "Solution": {"发明内容", "具体实施方式"},
    "Effects": {"发明内容"},
    "Figure": {"附图说明", "具体实施方式"},
    "Implementation": {"具体实施方式", "发明内容"},
    "Embodiment": {"具体实施方式"},
}


def _sections(assignments: dict[str, str], ordered_ids: list[str], include_figures: bool) -> list[PlanSection]:
    headings = [h for h in HEADINGS if h != "附图说明" or include_figures]
    return [
        PlanSection(heading=heading, node_ids=[n for n in ordered_ids if assignments.get(n) == heading])
        for heading in headings
    ]


def _score(plan: PatentPlan, graph: ConceptGraph) -> float:
    nodes = graph.node_map()
    seen = [n for section in plan.sections for n in section.node_ids]
    if len(seen) != len(nodes) or set(seen) != set(nodes) or len(set(seen)) != len(seen):
        return -1.0
    fit = sum(
        section.heading in _ALLOWED[nodes[node_id].category]
        for section in plan.sections for node_id in section.node_ids
    ) / max(1, len(nodes))
    section_of = {n: s.heading for s in plan.sections for n in s.node_ids}
    internal = sum(section_of.get(e.source) == section_of.get(e.target) for e in graph.edges)
    connectivity = internal / max(1, len(graph.edges))
    # The paper's density gate is unsuitable for sparse Chinese section subgraphs;
    # use it as a ranking signal rather than claiming its English threshold is met.
    return round(0.85 * fit + 0.15 * connectivity, 4)


def _rule_plan(graph: ConceptGraph, include_figures: bool) -> PatentPlan:
    assignments = {n.id: _DEFAULT[n.category] for n in graph.nodes}
    ordered_ids = [n.id for n in graph.nodes]
    plan = PatentPlan(sections=_sections(assignments, ordered_ids, include_figures))
    plan.score = _score(plan, graph)
    return plan


def _model_plan(graph: ConceptGraph, model: ModelClient, include_figures: bool, variant: int) -> PatentPlan:
    nodes = [{"id": n.id, "category": n.category, "text": n.text[:320]} for n in graph.nodes]
    edges = [{"source": e.source, "target": e.target, "relation": e.relation} for e in graph.edges]
    headings = [h for h in HEADINGS if h != "附图说明" or include_figures]
    system = (
        "你是中国发明专利说明书的章节规划助手。必须保持技术事实和依赖顺序，"
        "每个节点只能分配一次。输出 JSON 对象，不生成说明书正文。"
    )
    user = f"""将所有节点安排进以下固定章节：{headings}。
技术领域只放 Field；背景技术放 PriorArt、TechProblem；发明内容包含技术问题、技术方案和有益效果；
具体实施方式解释可执行步骤和实施例；附图说明只对应实际选择的图。
候选方案 {variant}：优先考虑{'问题到方案的论证顺序' if variant % 2 else '步骤依赖与实施细节顺序'}。
返回格式：{{"sections":[{{"heading":"技术领域","node_ids":["F1"]}}, ...]}}。
每个节点 ID 恰好出现一次，章节顺序与给定列表一致，不能新增节点。
节点：{nodes}
关系：{edges}"""
    data = model.ask_json(system, user, max_tokens=2800)
    raw_sections = data.get("sections")
    if not isinstance(raw_sections, list):
        raise ValueError("模型提纲缺少 sections。")
    assignments: dict[str, str] = {}
    ordered_ids: list[str] = []
    node_ids = {n.id for n in graph.nodes}
    for raw in raw_sections:
        if not isinstance(raw, dict) or raw.get("heading") not in headings:
            continue
        heading = raw["heading"]
        for node_id in raw.get("node_ids", []):
            if node_id in node_ids and node_id not in assignments:
                assignments[node_id] = heading
                ordered_ids.append(node_id)
    for node in graph.nodes:
        if node.id not in assignments:
            assignments[node.id] = _DEFAULT[node.category]
            ordered_ids.append(node.id)
    plan = PatentPlan(sections=_sections(assignments, ordered_ids, include_figures))
    plan.score = _score(plan, graph)
    return plan


def build_plan(
    graph: ConceptGraph, model: ModelClient, include_figures: bool = False, candidate_count: int = 3
) -> tuple[PatentPlan, list[PatentPlan]]:
    candidates = [_rule_plan(graph, include_figures)]
    failures = 0
    for variant in range(1, max(1, min(candidate_count, 5))):
        try:
            candidates.append(_model_plan(graph, model, include_figures, variant))
        except Exception:
            failures += 1
    best = max(candidates, key=lambda plan: plan.score)
    if failures:
        best.warnings.append(f"{failures} 份模型提纲未通过格式检查，已选用可用方案。")
    for category in ("Field", "TechProblem", "Solution"):
        if not any(n.category == category for n in graph.nodes):
            best.warnings.append(f"缺少 {category} 概念，相关章节需要人工补充。")
    return best, candidates


def check_plan(plan: PatentPlan, graph: ConceptGraph) -> None:
    headings = [s.heading for s in plan.sections]
    if headings != [h for h in HEADINGS if h in headings] or len(set(headings)) != len(headings):
        raise ValueError("提纲的章节名称或顺序无效。")
    if not {"技术领域", "背景技术", "发明内容", "具体实施方式"} <= set(headings):
        raise ValueError("提纲缺少说明书的基础章节。")
    if any(n.category == "Figure" for n in graph.nodes) != ("附图说明" in headings):
        raise ValueError("附图说明与已选择的附图概念不一致。")
    listed = [n for section in plan.sections for n in section.node_ids]
    node_ids = set(graph.node_map())
    if set(listed) != node_ids or len(listed) != len(node_ids):
        raise ValueError("提纲必须包含每个概念节点且每个节点只出现一次。")
