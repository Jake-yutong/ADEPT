from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import aiohttp  # pyright: ignore[reportMissingImports]

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ProviderName = Literal["deepseek", "qwen", "kimi", "openai", "custom"]


DEFAULT_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "openai": "https://api.openai.com/v1",
}


class ConfigError(ValueError):
    """配置读取或解析失败时抛出的异常。"""


class LLMAPIError(RuntimeError):
    """LLM 接口调用失败时抛出的异常。"""


@dataclass(slots=True)
class ModelConfig:
    """单个模型的配置。"""

    provider: ProviderName
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 90
    extra_headers: dict[str, str] = field(default_factory=dict)
    request_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeConfig:
    """运行时配置（并发、输出路径等）。"""

    concurrency: int = 5
    output_path: str = "outputs/adept_results.jsonl"


@dataclass(slots=True)
class AppConfig:
    """ADEPT 全局配置。"""

    teacher: ModelConfig
    student: ModelConfig
    judge: ModelConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


class ConfigLoader:
    """读取 config.yaml 与 .env，并构建统一配置对象。"""

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = "config.yaml",
        env_path: str | Path | None = ".env",
    ) -> AppConfig:
        """优先读取 YAML；不存在时回退到环境变量。"""

        if env_path is not None:
            cls._load_env_file(Path(env_path))

        if config_path is not None:
            yaml_path = Path(config_path)
            if yaml_path.exists():
                return cls._load_from_yaml(yaml_path)

        return cls._load_from_env()

    @staticmethod
    def _load_env_file(env_path: Path) -> None:
        """读取 .env 文件，并仅在变量缺失时注入到环境中。"""

        if not env_path.exists():
            return

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    @classmethod
    def _load_from_yaml(cls, yaml_path: Path) -> AppConfig:
        """从 YAML 配置文件读取 teacher/student/judge 三组模型配置。"""

        if yaml is None:
            raise ConfigError("读取 YAML 配置需要安装 PyYAML：pip install pyyaml")

        raw_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, Mapping):
            raise ConfigError("config.yaml 顶层结构必须为对象")

        model_root = raw_data.get("models") or raw_data.get("llm")
        if not isinstance(model_root, Mapping):
            raise ConfigError("config.yaml 必须包含 models（或 llm）节点")

        teacher_cfg = cls._model_from_mapping("teacher", model_root.get("teacher"))
        student_cfg = cls._model_from_mapping("student", model_root.get("student"))
        judge_cfg = cls._model_from_mapping("judge", model_root.get("judge"))

        runtime_mapping = raw_data.get("runtime", {})
        if not isinstance(runtime_mapping, Mapping):
            raise ConfigError("runtime 节点必须为对象")

        runtime_cfg = RuntimeConfig(
            concurrency=cls._as_int(runtime_mapping.get("concurrency"), default=5),
            output_path=str(runtime_mapping.get("output_path", "outputs/adept_results.jsonl")),
        )

        return AppConfig(
            teacher=teacher_cfg,
            student=student_cfg,
            judge=judge_cfg,
            runtime=runtime_cfg,
        )

    @classmethod
    def _load_from_env(cls) -> AppConfig:
        """从环境变量读取配置，变量名以 ADEPT_{ROLE}_* 约定。"""

        teacher_cfg = cls._model_from_env("teacher")
        student_cfg = cls._model_from_env("student")
        judge_cfg = cls._model_from_env("judge")

        runtime_cfg = RuntimeConfig(
            concurrency=cls._as_int(os.getenv("ADEPT_RUNTIME_CONCURRENCY"), default=5),
            output_path=os.getenv("ADEPT_RUNTIME_OUTPUT_PATH", "outputs/adept_results.jsonl"),
        )

        return AppConfig(
            teacher=teacher_cfg,
            student=student_cfg,
            judge=judge_cfg,
            runtime=runtime_cfg,
        )

    @classmethod
    def _model_from_mapping(cls, role: str, raw: Any) -> ModelConfig:
        """将 YAML 中某个角色的模型配置映射为 ModelConfig。"""

        if not isinstance(raw, Mapping):
            raise ConfigError(f"models.{role} 必须为对象")

        provider = str(raw.get("provider", "custom")).lower()
        model = str(raw.get("model", "")).strip()
        if not model:
            raise ConfigError(f"models.{role}.model 不能为空")

        base_url = str(raw.get("base_url") or DEFAULT_BASE_URLS.get(provider, "")).strip()
        if not base_url:
            raise ConfigError(f"models.{role}.base_url 未配置，且 provider={provider} 无默认地址")

        api_key = cls._resolve_api_key(
            role=role,
            provider=provider,
            inline_key=raw.get("api_key"),
            env_key=raw.get("api_key_env"),
        )

        return ModelConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=cls._as_float(raw.get("temperature"), default=0.2),
            max_tokens=cls._as_int(raw.get("max_tokens"), default=1024),
            timeout_seconds=cls._as_int(raw.get("timeout_seconds"), default=90),
            extra_headers=cls._as_str_dict(raw.get("extra_headers")),
            request_kwargs=dict(raw.get("request_kwargs", {})) if isinstance(raw.get("request_kwargs"), Mapping) else {},
        )

    @classmethod
    def _model_from_env(cls, role: str) -> ModelConfig:
        """从约定的环境变量读取角色模型配置。"""

        prefix = f"ADEPT_{role.upper()}_"
        provider = os.getenv(prefix + "PROVIDER", "deepseek").lower()
        model = os.getenv(prefix + "MODEL", cls._default_model(role, provider))
        base_url = os.getenv(prefix + "BASE_URL", DEFAULT_BASE_URLS.get(provider, "")).strip()

        if not model:
            raise ConfigError(f"缺少环境变量：{prefix}MODEL")
        if not base_url:
            raise ConfigError(f"缺少环境变量：{prefix}BASE_URL（provider={provider} 无默认地址）")

        api_key = cls._resolve_api_key(
            role=role,
            provider=provider,
            inline_key=os.getenv(prefix + "API_KEY"),
            env_key=None,
        )

        return ModelConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=cls._as_float(os.getenv(prefix + "TEMPERATURE"), default=0.2),
            max_tokens=cls._as_int(os.getenv(prefix + "MAX_TOKENS"), default=1024),
            timeout_seconds=cls._as_int(os.getenv(prefix + "TIMEOUT_SECONDS"), default=90),
        )

    @staticmethod
    def _default_model(role: str, provider: str) -> str:
        """为常见角色提供可用默认模型。"""

        defaults: dict[str, dict[str, str]] = {
            "teacher": {
                "deepseek": "deepseek-chat",
                "qwen": "qwen-max",
                "kimi": "moonshot-v1-8k",
                "openai": "gpt-4o-mini",
            },
            "student": {
                "deepseek": "deepseek-chat",
                "qwen": "qwen-plus",
                "kimi": "moonshot-v1-8k",
                "openai": "gpt-4o-mini",
            },
            "judge": {
                "deepseek": "deepseek-chat",
                "qwen": "qwen-max",
                "kimi": "moonshot-v1-32k",
                "openai": "gpt-4.1",
            },
        }
        return defaults.get(role, {}).get(provider, "")

    @staticmethod
    def _resolve_api_key(
        role: str,
        provider: str,
        inline_key: Any,
        env_key: Any,
    ) -> str:
        """解析 API Key：优先内联值，再查指定环境变量，最后查约定变量。"""

        if isinstance(inline_key, str) and inline_key.strip():
            return inline_key.strip()

        candidates: list[str] = []
        if isinstance(env_key, str) and env_key.strip():
            candidates.append(env_key.strip())

        candidates.extend(
            [
                f"ADEPT_{role.upper()}_API_KEY",
                f"{provider.upper()}_API_KEY",
                "OPENAI_API_KEY",
            ]
        )

        for key in candidates:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()

        raise ConfigError(
            f"无法解析 {role} 的 API Key，请设置环境变量 {candidates[0]}（或配置 api_key）"
        )

    @staticmethod
    def _as_int(raw: Any, default: int) -> int:
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"无法解析整数值: {raw}") from exc

    @staticmethod
    def _as_float(raw: Any, default: float) -> float:
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"无法解析浮点值: {raw}") from exc

    @staticmethod
    def _as_str_dict(raw: Any) -> dict[str, str]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ConfigError("extra_headers 必须为对象")
        return {str(k): str(v) for k, v in raw.items()}


class LLMClient(ABC):
    """LLM 客户端抽象基类。

    所有供应商客户端统一实现 generate() 异步方法，
    以便在 Orchestrator 中按统一接口调度。
    """

    def __init__(self, config: ModelConfig, session: aiohttp.ClientSession | None = None) -> None:
        self.config = config
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "LLMClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """释放当前客户端内部维护的 HTTP 会话。"""

        if self._session is not None and self._owns_session and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """懒加载 aiohttp 会话，避免提前占用连接资源。"""

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """根据输入 Prompt 生成文本输出。"""


class OpenAICompatibleClient(LLMClient):
    """兼容 OpenAI Chat Completions 协议的通用客户端实现。"""

    endpoint_path: str = "/chat/completions"

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
        }
        payload.update(self.config.request_kwargs)
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }

        session = await self._ensure_session()
        endpoint = f"{self.config.base_url.rstrip('/')}{self.endpoint_path}"

        async with session.post(endpoint, headers=headers, json=payload) as response:
            raw_text = await response.text()
            if response.status >= 400:
                raise LLMAPIError(
                    f"[{self.config.provider}] 请求失败：HTTP {response.status}，响应={raw_text[:500]}"
                )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMAPIError(
                f"[{self.config.provider}] 响应非 JSON，内容片段={raw_text[:200]}"
            ) from exc

        content = self._extract_content(data)
        if not content:
            raise LLMAPIError(f"[{self.config.provider}] 响应缺少可解析文本")
        return content

    @staticmethod
    def _extract_content(data: Mapping[str, Any]) -> str:
        """兼容不同供应商在 content 字段上的细微差异。"""

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            return ""

        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            return ""

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping):
                    part_text = part.get("text")
                    if isinstance(part_text, str):
                        text_parts.append(part_text)
                elif isinstance(part, str):
                    text_parts.append(part)
            return "".join(text_parts).strip()

        return ""


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek 客户端。"""

    default_model: str = "deepseek-chat"

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        # 对齐 DeepSeek 官方示例：非流式场景默认 stream=False。
        kwargs.setdefault("stream", False)

        # 若上层未提供 model，则回退到 DeepSeek 通用可用模型。
        if not self.config.model.strip() and not str(kwargs.get("model", "")).strip():
            kwargs["model"] = self.default_model

        return await super().generate(prompt, system_prompt=system_prompt, **kwargs)


class QwenClient(OpenAICompatibleClient):
    """Qwen 客户端（DashScope 兼容接口）。"""

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        # DashScope 兼容接口在非流式调用中要求显式关闭 thinking。
        # 这里统一兜底，避免默认参数导致 HTTP 400。
        stream_raw = kwargs.get("stream", self.config.request_kwargs.get("stream", False))
        if isinstance(stream_raw, bool):
            is_stream = stream_raw
        elif isinstance(stream_raw, str):
            is_stream = stream_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            is_stream = bool(stream_raw)

        if not is_stream:
            kwargs["enable_thinking"] = False

        return await super().generate(prompt, system_prompt=system_prompt, **kwargs)


class KimiClient(OpenAICompatibleClient):
    """Kimi 客户端（Moonshot 兼容接口）。"""


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI 客户端。"""


class LLMClientFactory:
    """按 provider 动态构建具体 LLM 客户端。"""

    _provider_to_client: dict[str, type[LLMClient]] = {
        "deepseek": DeepSeekClient,
        "qwen": QwenClient,
        "kimi": KimiClient,
        "openai": OpenAIClient,
        "custom": OpenAICompatibleClient,
    }

    @classmethod
    def create_client(
        cls,
        config: ModelConfig,
        session: aiohttp.ClientSession | None = None,
    ) -> LLMClient:
        provider = config.provider.lower()
        client_cls = cls._provider_to_client.get(provider)
        if client_cls is None:
            raise ConfigError(f"不支持的 provider: {provider}")
        return client_cls(config=config, session=session)
