from __future__ import annotations

import asyncio
import json
from typing import Any

from adept.models import DEFAULT_BASE_URLS, DeepSeekClient, ModelConfig, QwenClient


class _DummyResponse:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self._text = json.dumps(payload, ensure_ascii=False)

    async def __aenter__(self) -> "_DummyResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None

    async def text(self) -> str:
        return self._text


class _DummySession:
    def __init__(self, *, payload: dict[str, Any] | None = None) -> None:
        self.closed = False
        self.last_payload: dict[str, Any] | None = None
        self.payload = payload or {"choices": [{"message": {"content": "ok"}}]}

    def post(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> _DummyResponse:
        del endpoint, headers
        self.last_payload = dict(json or {})
        return _DummyResponse(
            status=200,
            payload=self.payload,
        )

    async def close(self) -> None:
        self.closed = True


def _qwen_config(*, request_kwargs: dict[str, Any] | None = None) -> ModelConfig:
    return ModelConfig(
        provider="qwen",
        model="qwen-plus",
        api_key="test-key",
        base_url=DEFAULT_BASE_URLS["qwen"],
        request_kwargs=request_kwargs or {},
    )


def _deepseek_config(
    *,
    model_name: str = "deepseek-chat",
    request_kwargs: dict[str, Any] | None = None,
) -> ModelConfig:
    return ModelConfig(
        provider="deepseek",
        model=model_name,
        api_key="test-key",
        base_url=DEFAULT_BASE_URLS["deepseek"],
        request_kwargs=request_kwargs or {},
    )


def test_qwen_non_streaming_forces_enable_thinking_false() -> None:
    async def _run() -> None:
        session = _DummySession()
        config = _qwen_config(request_kwargs={"enable_thinking": True})
        client = QwenClient(config=config, session=session)

        answer = await client.generate("请输出一句测试文本")

        assert answer == "ok"
        assert session.last_payload is not None
        assert session.last_payload["enable_thinking"] is False

    asyncio.run(_run())


def test_qwen_streaming_does_not_override_enable_thinking() -> None:
    async def _run() -> None:
        session = _DummySession()
        config = _qwen_config()
        client = QwenClient(config=config, session=session)

        answer = await client.generate(
            "请输出一句测试文本",
            stream=True,
            enable_thinking=True,
        )

        assert answer == "ok"
        assert session.last_payload is not None
        assert session.last_payload["stream"] is True
        assert session.last_payload["enable_thinking"] is True

    asyncio.run(_run())


def test_deepseek_defaults_to_non_streaming_request() -> None:
    async def _run() -> None:
        session = _DummySession()
        config = _deepseek_config()
        client = DeepSeekClient(config=config, session=session)

        answer = await client.generate("请输出一句测试文本")

        assert answer == "ok"
        assert session.last_payload is not None
        assert session.last_payload["stream"] is False
        assert session.last_payload["model"] == "deepseek-chat"

    asyncio.run(_run())


def test_deepseek_uses_default_model_when_model_is_empty() -> None:
    async def _run() -> None:
        session = _DummySession()
        config = _deepseek_config(model_name="")
        client = DeepSeekClient(config=config, session=session)

        answer = await client.generate("请输出一句测试文本")

        assert answer == "ok"
        assert session.last_payload is not None
        assert session.last_payload["model"] == "deepseek-chat"
        assert session.last_payload["stream"] is False

    asyncio.run(_run())


def test_deepseek_extracts_reasoning_content_when_content_missing() -> None:
    async def _run() -> None:
        session = _DummySession(
            payload={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": '{"score": 80, "reason": "结构清晰"}',
                        }
                    }
                ]
            }
        )
        config = _deepseek_config(model_name="deepseek-reasoner")
        client = DeepSeekClient(config=config, session=session)

        answer = await client.generate("请输出一句测试文本")

        assert "score" in answer

    asyncio.run(_run())


def test_deepseek_extracts_text_from_choice_text_fallback() -> None:
    async def _run() -> None:
        session = _DummySession(payload={"choices": [{"text": "fallback text"}]})
        config = _deepseek_config()
        client = DeepSeekClient(config=config, session=session)

        answer = await client.generate("请输出一句测试文本")

        assert answer == "fallback text"

    asyncio.run(_run())
