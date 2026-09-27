from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .model_api import ModelClient
from .pdf_extract import source_context
from .schema import CATEGORIES, ConceptEdge, ConceptGraph, ConceptNode, SourceDocument


_PREFIX = {
    "Field": "F", "TechProblem": "P", "PriorArt": "A", "Novelty": "N",
    "Solution": "S", "Implementation": "I", "Effects": "E", "Embodiment": "M", "Figure": "G",
}
_RELATIONS = {"solves", "implements", "causes", "improves", "validates"}
_RULES = (
    ("PriorArt", "TechProblem", "causes"),
    ("TechProblem", "Solution", "solves"),
    ("Solution", "Implementation", "implements"),
    ("Implementation", "Effects", "causes"),
    ("Solution", "Novelty", "improves"),
    ("Implementation", "Embodiment", "implements"),
)


def _item_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("items"), list):
        return [x for x in data["items"] if isinstance(x, dict)]
    items = []
    for category in CATEGORIES:
        for value in data.get(category, []):
            if isinstance(value, dict):
                items.append({"category": category, **value})
    return items


def extract_concepts(source: SourceDocument, model: ModelClient, include_figures: bool = False) -> ConceptGraph:
    context = source_context(source)
    system = (
        "你是技术资料整理助手。只从给定科研论文提取事实，输出一个 JSON 对象。"
        "不得补写论文没有披露的技术步骤、参数、实验结果、现有技术或实施例。"
        "每条内容必须给出支持它的原文块编号 source_ids；无法找到依据时省略该条。"
    )
    user = f"""请为中文专利说明书准备技术概念。分类只用以下九种英文标识：
Field（技术领域）、TechProblem（现有方法的具体技术问题）、PriorArt（论文明确记载的现有方法）、
Novelty（论文明确提出的区别特征）、Solution（核心解决方案）、Implementation（可执行的算法或系统步骤）、
Effects（有数据或论证支持的效果）、Embodiment（论文确有支持的具体实现）、Figure（确有图题的附图）。
每条应是单独的技术事实，少用泛泛表述。特别保留公式、数值、模块名的原文。
返回格式：{{"patent_title":"简短中文技术名称","items":[{{"category":"Solution","text":"...","source_ids":["B0001"]}}]}}。
每类最多 6 条；所有 source_ids 必须来自下面的块编号。{'' if include_figures else '当前未选择复用论文附图，Figure 类返回空。'}

论文原文：
{context}"""
    data = model.ask_json(system, user, max_tokens=6500)
    source_ids = set(source.block_map())
    nodes: list[ConceptNode] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    for item in _item_list(data):
        category = item.get("category")
        if category not in CATEGORIES or (category == "Figure" and not include_figures):
            continue
        value = str(item.get("text") or "").strip()
        if len(value) < 4:
            continue
        cited = [str(x) for x in item.get("source_ids", []) if str(x) in source_ids]
        cited = list(dict.fromkeys(cited))
        if not cited:
            warnings.append(f"一条 {category} 概念缺少有效原文位置，已略过：{value[:35]}")
            continue
        counts[category] += 1
        if counts[category] > 6:
            continue
        nodes.append(
            ConceptNode(id=f"{_PREFIX[category]}{counts[category]}", category=category, text=value, source_ids=cited)
        )
    for mandatory in ("Field", "TechProblem", "Solution"):
        if not counts[mandatory]:
            warnings.append(f"缺少 {mandatory} 技术概念，需要人工补充。")
    title = str(data.get("patent_title") or "专利说明书草稿").strip()
    return ConceptGraph(patent_title=title[:80], nodes=nodes, edges=[], warnings=warnings)


def _rule_edges(nodes: list[ConceptNode]) -> list[ConceptEdge]:
    by_category: dict[str, list[ConceptNode]] = defaultdict(list)
    for node in nodes:
        by_category[node.category].append(node)
    edges: list[ConceptEdge] = []
    for source_type, target_type, relation in _RULES:
        targets = by_category[target_type]
        for source in by_category[source_type]:
            if not targets:
                continue
            best = max(targets, key=lambda target: len(set(source.source_ids) & set(target.source_ids)))
            edges.append(ConceptEdge(source=source.id, target=best.id, relation=relation))
    return edges


def _infer_edges(nodes: list[ConceptNode], model: ModelClient, variant: int) -> list[ConceptEdge]:
    compact = [
        {"id": n.id, "category": n.category, "text": n.text[:280], "source_ids": n.source_ids}
        for n in nodes
    ]
    system = (
        "你是技术关系抽取助手。只推断下列现有节点之间由论文支持的有向关系。"
        "节点 ID 不得新增，关系仅可用 solves、implements、causes、improves、validates。"
        "返回 JSON 对象，不输出解释。"
    )
    focus = "明确的技术因果链与实现依赖" if variant == 2 else "跨段落的改进关系与实验证实关系"
    user = (
        f"重点寻找{focus}。方向与原文的技术逻辑一致，不要为凑数量编关系。"
        "返回格式：{\"edges\":[{\"source\":\"P1\",\"target\":\"S1\",\"relation\":\"solves\"}]}。"
        f"\n节点：{compact}"
    )
    data = model.ask_json(system, user, max_tokens=2600)
    node_ids = {n.id for n in nodes}
    edges: list[ConceptEdge] = []
    for value in data.get("edges", []):
        if not isinstance(value, dict):
            continue
        a, b, rel = value.get("source"), value.get("target"), value.get("relation")
        if a in node_ids and b in node_ids and a != b and rel in _RELATIONS:
            edges.append(ConceptEdge(source=a, target=b, relation=rel))
    return edges


def _has_path(start: str, goal: str, edges: list[ConceptEdge]) -> bool:
    stack = [start]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == goal:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(e.target for e in edges if e.source == current)
    return False


def merge_candidates(nodes: list[ConceptNode], candidates: list[list[ConceptEdge]]) -> tuple[list[ConceptEdge], list[str]]:
    votes: Counter[tuple[str, str, str]] = Counter()
    rule_keys = {(e.source, e.target, e.relation) for e in candidates[0]} if candidates else set()
    for candidate in candidates:
        votes.update({(e.source, e.target, e.relation): 1 for e in candidate})
    by_pair: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (a, b, relation), count in votes.items():
        by_pair[(a, b)].append((relation, count))
    selected: list[ConceptEdge] = []
    warnings: list[str] = []
    for (a, b), relations in by_pair.items():
        relations.sort(key=lambda x: (-x[1], (a, b, x[0]) not in rule_keys, x[0]))
        relation, count = relations[0]
        if len(relations) > 1 and relations[0][1] == relations[1][1]:
            warnings.append(f"关系 {a}→{b} 存在同票冲突，已选择 {relation}；请复核。")
        selected.append(ConceptEdge(source=a, target=b, relation=relation, votes=count))
    result: list[ConceptEdge] = []
    for edge in sorted(selected, key=lambda e: (-e.votes, (e.source, e.target, e.relation) not in rule_keys)):
        if _has_path(edge.target, edge.source, result):
            warnings.append(f"已移除会形成环的关系 {edge.source}→{edge.target}。")
        else:
            result.append(edge)
    connected = {n for e in result for n in (e.source, e.target)}
    for node in nodes:
        if node.id not in connected:
            warnings.append(f"概念 {node.id} 与其他概念尚无关系，请复核。")
    return result, warnings


def build_graph(source: SourceDocument, model: ModelClient, include_figures: bool = False) -> ConceptGraph:
    graph = extract_concepts(source, model, include_figures)
    candidates = [_rule_edges(graph.nodes)]
    for variant in (2, 3):
        try:
            candidates.append(_infer_edges(graph.nodes, model, variant))
        except Exception as exc:
            graph.warnings.append(f"第 {variant} 份关系候选图生成失败：{type(exc).__name__}；保留已有候选图。")
    graph.edges, merge_warnings = merge_candidates(graph.nodes, candidates)
    graph.warnings.extend(merge_warnings)
    return graph


def check_graph(graph: ConceptGraph, source: SourceDocument) -> None:
    node_ids = [n.id for n in graph.nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("概念图存在重复节点 ID。")
    valid_sources = set(source.block_map())
    for node in graph.nodes:
        if not node.source_ids or not set(node.source_ids) <= valid_sources:
            raise ValueError(f"概念 {node.id} 的原文编号无效。")
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            raise ValueError("概念图存在指向未知节点的关系。")
