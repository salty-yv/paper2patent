from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import httpx


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["openai", "ollama"] = "openai"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("请填写模型名。")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("接口地址须以 https:// 或 http:// 开头。")
        if self.provider == "openai" and not self.api_key.strip():
            raise ValueError("请在本地页面填写接口密钥。")


def _extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ModelError("模型未返回 JSON，请重试该阶段或换用支持结构化输出的模型。") from None
        try:
            value = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelError(f"模型返回的 JSON 无法解析：{exc.msg}") from exc
    if not isinstance(value, dict):
        raise ModelError("模型返回的顶层格式应为 JSON 对象。")
    return value


class ModelClient:
    def __init__(self, config: ModelConfig):
        config.validate()
        self.config = config
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def ask_json(self, system: str, user: str, max_tokens: int = 4096) -> dict[str, Any]:
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "ollama":
            url = base + "/api/chat"
            headers: dict[str, str] = {}
            payload = {
                "model": self.config.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": max_tokens},
            }
        else:
            # Accept both service roots and OpenAI-style /v1 base URLs.
            url = base + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            payload = {
                "model": self.config.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if urlparse(base).hostname == "api.deepseek.com":
                # DeepSeek Flash defaults to high-effort thinking. On long JSON
                # extraction it can spend the output budget before final content.
                payload["thinking"] = {"type": "disabled"}
                payload["response_format"] = {"type": "json_object"}
        direct_first = self.config.provider == "openai" and urlparse(base).hostname == "api.deepseek.com"
        for attempt in range(5):
            try:
                # This Windows setup has a local HTTP proxy that occasionally
                # breaks TLS handshakes to DeepSeek. Try direct HTTPS first;
                # retain proxy fallback for networks that require it.
                trust_env = not direct_first or attempt in (1, 3)
                with httpx.Client(timeout=httpx.Timeout(240.0, connect=20.0), trust_env=trust_env) as client:
                    response = client.post(url, json=payload, headers=headers)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if response.is_error:
                    raise ModelError(
                        f"模型接口返回 HTTP {response.status_code}。请检查接口地址、模型名、密钥或额度。"
                    )
                data = response.json()
                finish_reason = ""
                if self.config.provider == "ollama":
                    content = data.get("message", {}).get("content", "")
                    self.input_tokens += int(data.get("prompt_eval_count") or 0)
                    self.output_tokens += int(data.get("eval_count") or 0)
                else:
                    choices = data.get("choices") or []
                    content = choices[0].get("message", {}).get("content", "") if choices else ""
                    finish_reason = choices[0].get("finish_reason", "") if choices else ""
                    usage = data.get("usage") or {}
                    self.input_tokens += int(usage.get("prompt_tokens") or 0)
                    self.output_tokens += int(usage.get("completion_tokens") or 0)
                self.calls += 1
                if not isinstance(content, str) or not content.strip():
                    if finish_reason == "length":
                        raise ModelError("模型已用完输出长度仍未返回正文；请重试该阶段或调整模型设置。")
                    raise ModelError("模型返回了空内容，请重试该阶段。")
                return _extract_json(content)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                detail = str(exc)
                if self.config.api_key:
                    detail = detail.replace(self.config.api_key, "[hidden]")
                detail = detail[:170]
                raise ModelError(f"无法连接模型接口：{type(exc).__name__}：{detail}") from exc
        raise ModelError("模型接口重试后仍未成功。")

    def usage(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
