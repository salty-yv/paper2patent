"""Run a locally parsed paper through the configured API without saving its key."""

from __future__ import annotations

import argparse
import getpass
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patent_writer.model_api import ModelClient, ModelConfig, ModelError
from patent_writer.run_store import stage_draft, stage_evidence, stage_graph, stage_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--from-stage", choices=("graph", "plan", "draft", "evidence"), default="graph")
    parser.add_argument("--skip-check", action="store_true")
    arguments = parser.parse_args()
    key = getpass.getpass("模型接口密钥：")
    config = ModelConfig(
        provider="openai", model=arguments.model,
        base_url=arguments.base_url, api_key=key,
    )
    if not arguments.skip_check:
        print("正在检查接口连接", flush=True)
        ModelClient(config).ask_json("请只返回 JSON 对象。", "返回一个包含 ok=true 的 JSON 对象。", max_tokens=120)
    if arguments.from_stage == "graph":
        print("接口已连接，开始提取概念图", flush=True)
        stage_graph(arguments.run_id, config)
    if arguments.from_stage in ("graph", "plan"):
        print("开始规划提纲", flush=True)
        stage_plan(arguments.run_id, config, candidate_count=3)
    if arguments.from_stage != "evidence":
        print("开始起草正文", flush=True)
        stage_draft(arguments.run_id, config)
        print("草稿已完成", flush=True)
    print("开始逐条核对原文依据", flush=True)
    for attempt in range(5):
        try:
            stage_evidence(arguments.run_id, config)
            break
        except ModelError as exc:
            if "无法连接模型接口" not in str(exc) or attempt == 4:
                raise
            print(f"接口连接中断，稍后从已保存的段落继续（第 {attempt + 1} 次重连）", flush=True)
            time.sleep(min(5 * (attempt + 1), 20))
    print("证据核对已完成", flush=True)


if __name__ == "__main__":
    main()
