from __future__ import annotations

import asyncio
import html
import json
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import streamlit as st  # pyright: ignore[reportMissingImports]

from adept import AppConfig, ModelConfig, RuntimeConfig, build_api_bundle
from adept.models import DEFAULT_BASE_URLS


# -----------------------------
# 页面级配置
# -----------------------------
# 这里先设置页面标题、布局等全局参数。
# 不设置 page_icon 可以避免使用任何 emoji，保持学术风格。
st.set_page_config(
    page_title="ADEPT 控制台",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# 数据结构定义
# -----------------------------
# 使用 dataclass 让问题样本结构更清晰，便于后续替换为真实后端数据。
@dataclass
class QuestionSample:
    source_material: str
    design_question: str
    teaching_guideline: str


@dataclass
class UIConfig:
    """将侧边栏输入统一封装，便于调用与维护。"""

    run_mode: str
    teacher_model_id: str
    student_model_id: str
    judge_model_id: str
    teacher_api_key: str
    student_api_key: str
    judge_api_key: str
    teacher_base_url: str
    student_base_url: str
    judge_base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: int


# -----------------------------
# 样式函数（学术风格 UI）
# -----------------------------
def apply_custom_style() -> None:
    """注入自定义 CSS，让页面更现代且有学术感（Apple 风格）。"""

    st.markdown(
        """
        <style>
        :root {
            --apple-bg: #fbfbfd;
            --apple-card: #ffffff;
            --apple-border: #d2d2d7;
            --apple-text: #1d1d1f;
            --apple-muted: #86868b;
            --apple-blue: #0071e3;
            --apple-blue-hover: #0077ed;
            --apple-radius-large: 18px;
            --apple-radius-base: 12px;
            --apple-shadow: 0 4px 6px rgba(0, 0, 0, 0.02), 0 10px 15px rgba(0, 0, 0, 0.03);
            --apple-shadow-hover: 0 8px 12px rgba(0, 0, 0, 0.04), 0 16px 24px rgba(0, 0, 0, 0.06);
        }

        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--apple-text);
            background: var(--apple-bg);
            -webkit-font-smoothing: antialiased;
        }

        section.main > div.block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
            color: var(--apple-text);
            font-weight: 600;
            letter-spacing: -0.015em;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--apple-border);
            background: #f5f5f7;
        }

        [data-testid="stSidebar"] * {
            color: var(--apple-text);
        }

        [data-testid="stWidgetLabel"] {
            color: var(--apple-text) !important;
            font-weight: 500;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {
            background: rgba(255, 255, 255, 0.8) !important;
            color: var(--apple-text) !important;
            border: 1px solid var(--apple-border) !important;
            border-radius: var(--apple-radius-base) !important;
            box-shadow: none !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within {
            border-color: var(--apple-blue) !important;
            box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.2) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: var(--apple-text) !important;
            -webkit-text-fill-color: var(--apple-text) !important;
        }

        [data-testid="stTextArea"] textarea:disabled {
            color: var(--apple-muted) !important;
            -webkit-text-fill-color: var(--apple-muted) !important;
            background: #f5f5f7 !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 1px dashed var(--apple-border) !important;
            border-radius: var(--apple-radius-large) !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            transition: border-color 0.2s ease;
        }
        
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--apple-blue) !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: var(--apple-text) !important;
            fill: var(--apple-text) !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--apple-border);
            border-radius: var(--apple-radius-large);
            box-shadow: var(--apple-shadow);
            background: var(--apple-card);
            overflow: hidden;
            transition: box-shadow 0.3s ease;
        }

        [data-testid="stExpander"]:hover {
            box-shadow: var(--apple-shadow-hover);
        }

        [data-testid="stExpander"] details summary p {
            color: var(--apple-text) !important;
            font-weight: 600;
            font-size: 1.05rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid var(--apple-border);
            border-radius: var(--apple-radius-large);
            padding: 1rem;
            background: var(--apple-card);
            box-shadow: var(--apple-shadow);
            text-align: center;
        }

        [data-testid="stMetricLabel"] {
            color: var(--apple-muted) !important;
            font-weight: 500;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.4rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--apple-text) !important;
            font-weight: 700;
            font-size: 2rem;
            letter-spacing: -0.02em;
        }

        [data-testid="stMarkdownContainer"] p {
            line-height: 1.6;
            color: var(--apple-text);
        }

        [data-testid="stButton"] > button {
            border: none !important;
            border-radius: 980px !important; /* Pill shape */
            background: var(--apple-blue) !important;
            color: #ffffff !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
            padding: 0.6rem 1.2rem !important;
            min-height: 2.8rem;
            transition: background-color 0.2s ease, transform 0.1s ease !important;
        }

        [data-testid="stButton"] > button:hover {
            background: var(--apple-blue-hover) !important;
        }
        
        [data-testid="stButton"] > button:active {
            transform: scale(0.98) !important;
        }

        .adept-intro {
            font-size: 1.05rem;
            color: var(--apple-text);
            border: none;
            background: var(--apple-card);
            border-radius: var(--apple-radius-large);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--apple-shadow);
            line-height: 1.5;
            text-align: center;
        }

        .adept-note {
            border: 1px solid var(--apple-border);
            border-radius: var(--apple-radius-large);
            background: #f5f5f7;
            color: var(--apple-text);
            padding: 1.2rem;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        .adept-scroll-panel {
            border: 1px solid var(--apple-border);
            border-radius: var(--apple-radius-base);
            background: #ffffff;
            padding: 0.7rem 0.8rem;
            height: 280px;
            overflow-y: auto;
            overflow-x: auto;
        }

        .adept-scroll-panel pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            color: var(--apple-text);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.55;
            font-size: 0.93rem;
        }

        .adept-caption {
            font-size: 0.9rem;
            color: var(--apple-muted);
            padding-left: 0.2rem;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }

        @media (max-width: 980px) {
            section.main > div.block-container {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 工具函数：读取上传文件
# -----------------------------
def load_rubric_text(uploaded_file: Any) -> str:
    """读取 rubric 文件内容；若未上传则返回默认 rubric。"""

    default_rubric = (
        "1) 问题理解（0-25）\n"
        "2) 设计策略（0-25）\n"
        "3) 论证质量（0-25）\n"
        "4) 表达清晰（0-25）"
    )

    if uploaded_file is None:
        return default_rubric

    raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
    return raw_text or default_rubric


def _parse_sample_object(raw: dict[str, Any]) -> QuestionSample | None:
    """把 JSON 对象解析为 QuestionSample；字段缺失时返回 None。"""

    source_material = str(raw.get("source_material", "")).strip()
    design_question = str(raw.get("design_question", "")).strip()
    teaching_guideline = str(
        raw.get(
            "teaching_guideline",
            "请围绕设计目标、约束条件、实现步骤和评估标准进行教学引导。",
        )
    ).strip()

    if not source_material or not design_question:
        return None

    return QuestionSample(
        source_material=source_material,
        design_question=design_question,
        teaching_guideline=teaching_guideline,
    )


def load_question_samples(uploaded_file: Any) -> list[QuestionSample]:
    """读取题目文件，支持 json/jsonl/txt；失败时回退到内置样例。"""

    # 内置样例用于“先看 UI 效果”的场景，保证用户无文件也可运行。
    fallback_samples = [
        QuestionSample(
            source_material=(
                "某高校计划改造旧图书馆一层公共区域，目标是提升小组协作学习效率，"
                "同时满足安静阅读、短时讨论与可持续材料使用。"
            ),
            design_question="请提出一个空间改造方案，并说明动线、分区和材料策略。",
            teaching_guideline="强调用户研究、空间层级、低碳材料与可评估指标。",
        )
    ]

    if uploaded_file is None:
        return fallback_samples

    text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
    if not text:
        return fallback_samples

    suffix = uploaded_file.name.lower().rsplit(".", maxsplit=1)[-1]
    samples: list[QuestionSample] = []

    try:
        if suffix == "jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    parsed = _parse_sample_object(data)
                    if parsed is not None:
                        samples.append(parsed)

        elif suffix == "json":
            data = json.loads(text)

            # 支持三种常见结构：
            # 1) 顶层就是单个 sample 对象
            # 2) 顶层是 sample 数组
            # 3) 顶层对象中含 samples 字段
            if isinstance(data, dict):
                if isinstance(data.get("samples"), list):
                    for item in data["samples"]:
                        if isinstance(item, dict):
                            parsed = _parse_sample_object(item)
                            if parsed is not None:
                                samples.append(parsed)
                else:
                    parsed = _parse_sample_object(data)
                    if parsed is not None:
                        samples.append(parsed)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        parsed = _parse_sample_object(item)
                        if parsed is not None:
                            samples.append(parsed)

        else:
            # 对 txt 做结构化解析：
            # 1) 优先按 QUESTION 块分割（适配 ADEPT 标准题目格式）
            # 2) 若无 QUESTION 标记则回退到单题模式
            samples = _parse_structured_txt(text)
    except Exception:
        # 解析失败直接回退，避免阻断 UI 演示。
        return fallback_samples

    return samples if samples else fallback_samples


def _parse_structured_txt(text: str) -> list[QuestionSample]:
    """解析 ADEPT 标准格式的 txt 文件，按 QUESTION 块拆分为独立样本。"""

    import re as _re

    # 提取全局说明作为答题格式约束
    format_hints: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "选择题" in stripped and "分" in stripped:
            format_hints.append(stripped)
        if "简答题" in stripped and "字" in stripped:
            format_hints.append(stripped)
    format_note = "；".join(format_hints) if format_hints else ""

    # 按 QUESTION N 标记切分
    question_blocks = _re.split(
        r"(?=^-{3,}\s*\n\s*QUESTION\s+\d+)",
        text,
        flags=_re.MULTILINE,
    )

    samples: list[QuestionSample] = []
    for block in question_blocks:
        block = block.strip()
        if not block:
            continue

        # 提取题号
        q_match = _re.search(r"QUESTION\s+(\d+)", block)
        if not q_match:
            continue

        q_num = q_match.group(1)

        # 提取难度与对应原文
        difficulty = ""
        d_match = _re.search(r"\[DIFFICULTY:\s*(.+?)\]", block)
        if d_match:
            difficulty = d_match.group(1).strip()

        source_ref = ""
        s_match = _re.search(r"对应原文[：:](.+)", block)
        if s_match:
            source_ref = s_match.group(1).strip()

        # 提取选择题和简答题
        choice_q = ""
        short_q = ""

        choice_match = _re.search(
            r"【选择题】\s*\n([\s\S]*?)(?=【简答题】|$)", block
        )
        if choice_match:
            choice_q = choice_match.group(1).strip()

        short_match = _re.search(
            r"【简答题】[^\n]*\n([\s\S]*?)(?=^-{3,}|$)",
            block,
            flags=_re.MULTILINE,
        )
        if short_match:
            short_q = short_match.group(1).strip()

        if not choice_q and not short_q:
            continue

        # 组装 design_question —— 将选择题和简答题合并，明确标注题型
        parts = [f"=== QUESTION {q_num} ({difficulty}) ==="]
        if choice_q:
            parts.append(f"【选择题 - 30分】\n{choice_q}")
        if short_q:
            parts.append(f"【简答题 - 70分】（100字以内）\n{short_q}")
        design_question = "\n\n".join(parts)

        # source_material = 对应原文描述 + 格式说明
        source_material = f"对应原文：{source_ref}" if source_ref else f"Question {q_num}"
        if format_note:
            source_material += f"\n\n答题格式说明：{format_note}"

        # teaching_guideline 从题目上下文推断
        block_start_in_text = text.find(block)
        preceding_text = text[:block_start_in_text] if block_start_in_text > 0 else ""
        # 找到最近的 PART 标题
        all_parts = list(_re.finditer(r"PART\s+[IVX]+:\s*(.+?)(?:\s*$|\s*=)", preceding_text, _re.MULTILINE))
        if all_parts:
            part_name = all_parts[-1].group(1).strip().rstrip("=").strip()
        else:
            part_name = "设计原理"
        teaching_guideline = (
            f"本题考察 {part_name} 相关理论。"
            f"难度：{difficulty}。"
            f"请围绕 {source_ref or part_name} 进行教学引导，"
            "重点辅导选择题的关键概念辨析和简答题的答题框架。"
        )

        samples.append(
            QuestionSample(
                source_material=source_material,
                design_question=design_question,
                teaching_guideline=teaching_guideline,
            )
        )

    # 若未解析出 QUESTION 块，回退到单题模式
    if not samples:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            samples.append(
                QuestionSample(
                    source_material="\n".join(lines[1:]),
                    design_question=lines[0],
                    teaching_guideline="请补充设计推理链路、关键步骤与可落地建议。",
                )
            )

    return samples


# -----------------------------
# Mock 模型函数（占位）
# -----------------------------
def mock_student_baseline(sample: QuestionSample, model_id: str) -> str:
    """模拟学生在无教师干预时的初始回答。"""

    time.sleep(0.7)
    return (
        f"[{model_id}] 基线回答：\n"
        "我会先给出空间功能分区，再描述关键动线。"
        "目前方案能够覆盖基本需求，但对评估指标和材料证据的说明较弱。"
    )


def mock_teacher_guidance(sample: QuestionSample, model_id: str) -> str:
    """模拟教师给出的教学干预知识。"""

    time.sleep(0.8)
    return (
        f"[{model_id}] 教学指导：\n"
        "1) 先建立用户画像与行为路径；\n"
        "2) 用“静区-协作区-过渡区”组织空间层级；\n"
        "3) 材料选择需说明耐久性、回收性与维护成本；\n"
        "4) 给出可量化评估指标，例如使用率、停留时长与噪声水平。"
    )


def mock_student_final(sample: QuestionSample, teacher_knowledge: str, model_id: str) -> str:
    """模拟学生吸收教师指导后的改进回答。"""

    time.sleep(0.9)
    return (
        f"[{model_id}] 最终回答：\n"
        "基于教师建议，我将空间划分为静读区、协作区和过渡讨论区，"
        "主环形动线连接入口与服务台，避免人流冲突。"
        "材料方面采用可再生木饰面与低 VOC 涂层，并提出维护周期。"
        "评估上设定三项指标：日均使用率提升 20%、噪声峰值下降 15%、"
        "协作区平均停留时长提升 10%。"
    )


def mock_rubric_judge(
    *,
    base_answer: str,
    improved_answer: str,
    rubric_text: str,
    judge_model_id: str,
) -> dict[str, Any]:
    """模拟裁判评分，并返回 base / knowledge / delta。"""

    # 这里用稳定的长度规则生成“可重复”的假分值，便于演示。
    # 正式接入时，只需要替换该函数中的打分逻辑即可。
    base_score = 58 + (len(base_answer) + len(judge_model_id)) % 16
    knowledge_bonus = 8 + (len(improved_answer) + len(rubric_text)) % 10
    knowledge_score = min(100, base_score + knowledge_bonus)

    return {
        "s_base": base_score,
        "s_knowledge": knowledge_score,
        "delta": knowledge_score - base_score,
        "reason": (
            "改进回答在策略结构、量化指标和可执行细节上更完整，"
            "与 Rubric 的“设计策略、论证质量、表达清晰”维度更匹配。"
        ),
    }


# -----------------------------
# 真实 API 调用函数
# -----------------------------
def infer_provider_by_model(model_id: str) -> str:
    """根据模型 ID 做启发式 provider 推断。"""

    model = model_id.strip().lower()

    if "deepseek" in model:
        return "deepseek"
    if "qwen" in model:
        return "qwen"
    if "moonshot" in model or "kimi" in model:
        return "kimi"
    if model.startswith(("gpt-", "o1", "o3", "o4")) or "openai" in model:
        return "openai"
    return "custom"


def infer_provider_by_base_url(base_url: str) -> str:
    """根据 Base URL 做启发式 provider 推断。"""

    url = base_url.strip().lower()
    if not url:
        return "custom"

    if "deepseek" in url:
        return "deepseek"
    if "dashscope.aliyuncs.com" in url or "qwen" in url:
        return "qwen"
    if "moonshot" in url or "kimi" in url:
        return "kimi"
    if "openai" in url:
        return "openai"
    return "custom"


def infer_default_model_for_role(*, role: str, provider: str) -> str:
    """按角色与 provider 返回默认模型；无法推断时返回空字符串。"""

    defaults: dict[str, dict[str, str]] = {
        "teacher": {
            "deepseek": "deepseek-chat",
            "qwen": "qwen-max",
            "kimi": "moonshot-v1-8k",
            "openai": "gpt-4.1",
        },
        "student": {
            "deepseek": "deepseek-chat",
            "qwen": "qwen-plus",
            "kimi": "moonshot-v1-8k",
            "openai": "gpt-4.1-mini",
        },
        "judge": {
            "deepseek": "deepseek-chat",
            "qwen": "qwen-max",
            "kimi": "moonshot-v1-32k",
            "openai": "gpt-4.1",
        },
    }
    return defaults.get(role, {}).get(provider, "")


def build_model_config_from_ui(
    *,
    role: str,
    model_id: str,
    api_key: str,
    base_url_input: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    allow_empty_model_id: bool = False,
) -> ModelConfig:
    """把界面参数转成后端模型配置。"""

    normalized_model = model_id.strip()
    normalized_base_url = base_url_input.strip()

    # 用户显式填写 Base URL 时，以 URL 推断结果优先，避免 provider 丢失。
    provider_from_base_url = infer_provider_by_base_url(normalized_base_url)
    provider_from_model = infer_provider_by_model(normalized_model) if normalized_model else "custom"

    if provider_from_base_url != "custom":
        provider = provider_from_base_url
    elif provider_from_model != "custom":
        provider = provider_from_model
    else:
        # 当无法自动识别 provider 时，默认回退到 OpenAI 兼容配置。
        provider = "openai"

    if not normalized_model:
        if not allow_empty_model_id:
            raise ValueError("Model ID 不能为空")
        inferred_default_model = infer_default_model_for_role(role=role, provider=provider)
        if not inferred_default_model:
            raise ValueError(
                f"{role.title()} Model ID 为空，且无法自动推断默认模型，请手动填写。"
            )
        normalized_model = inferred_default_model

    if normalized_base_url:
        base_url = normalized_base_url
    else:
        base_url = DEFAULT_BASE_URLS.get(provider, DEFAULT_BASE_URLS["openai"])

    return ModelConfig(
        provider=provider,  # type: ignore[arg-type]
        model=normalized_model,
        api_key=api_key.strip(),
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def analyze_role_config_overlap(
    role_configs: dict[str, ModelConfig],
) -> tuple[list[str], list[str]]:
    """分析角色配置是否复用 API Key，或路由到同一模型端点。"""

    same_key_pairs: list[str] = []
    same_endpoint_pairs: list[str] = []

    items = list(role_configs.items())
    for (left_name, left_cfg), (right_name, right_cfg) in combinations(items, 2):
        if left_cfg.api_key.strip() and left_cfg.api_key.strip() == right_cfg.api_key.strip():
            same_key_pairs.append(f"{left_name}/{right_name}")

        if (
            left_cfg.provider == right_cfg.provider
            and left_cfg.model.strip() == right_cfg.model.strip()
            and left_cfg.base_url.rstrip("/") == right_cfg.base_url.rstrip("/")
        ):
            same_endpoint_pairs.append(f"{left_name}/{right_name}")

    return same_key_pairs, same_endpoint_pairs


def run_coroutine(loop: asyncio.AbstractEventLoop, coro: Any) -> Any:
    """在固定事件循环中执行协程，避免跨 loop 使用 aiohttp session。"""

    return loop.run_until_complete(coro)


def render_scrollable_text(container: Any, text: str) -> None:
    """在固定高度可滚动容器中渲染文本输出。"""

    safe_text = html.escape(text)
    container.markdown(
        f'<div class="adept-scroll-panel"><pre>{safe_text}</pre></div>',
        unsafe_allow_html=True,
    )


def render_sample_markdown(sample: QuestionSample) -> str:
    """统一格式化当前样本展示内容。"""

    return "\n".join(
        [
            "### 当前样本",
            f"**Source Material**\n\n{sample.source_material}",
            f"**Design Question**\n\n{sample.design_question}",
        ]
    )


def render_sidebar() -> tuple[UIConfig, Any, Any]:
    """渲染侧边栏并返回配置与上传文件对象。"""

    with st.sidebar:
        st.title("ADEPT 控制台")
        st.markdown('<div class="adept-caption">统一配置评测参数与输入文件</div>', unsafe_allow_html=True)

        st.subheader("运行模式")

        run_mode = st.radio(
            label="运行模式",
            options=["Real API（真实）", "Mock（演示）"],
            index=0,
            help="真实模式会调用你在 ADEPT 后端里定义的 API 链路；演示模式使用假数据。",
        )

        st.subheader("API 与模型")

        teacher_model_id = st.text_input(
            label="Teacher Model ID",
            value="gpt-4.1",
            placeholder="手动输入 Teacher 模型 ID",
        ).strip()
        teacher_api_key = st.text_input(
            label="Teacher API Key",
            type="password",
            placeholder="输入 Teacher API Key",
        ).strip()
        teacher_base_url = st.text_input(
            label="Teacher Base URL（可选）",
            value="",
            placeholder="例如：https://api.openai.com/v1",
            help="留空时会按 Teacher Model ID 自动推断默认平台地址。",
        ).strip()

        st.divider()

        student_model_id = st.text_input(
            label="Student Model ID",
            value="gpt-4.1-mini",
            placeholder="手动输入 Student 模型 ID",
        ).strip()
        student_api_key = st.text_input(
            label="Student API Key",
            type="password",
            placeholder="输入 Student API Key",
        ).strip()
        student_base_url = st.text_input(
            label="Student Base URL（可选）",
            value="",
            placeholder="例如：https://api.deepseek.com",
            help="可与 Teacher/Judge 使用不同平台地址。",
        ).strip()

        st.divider()

        judge_model_options = [
            "deepseek-reasoner",
            "deepseek-chat",
            "qwen-max",
            "qwen-plus",
            "gpt-4.1",
            "gpt-4.1-mini",
            "moonshot-v1-32k",
            "moonshot-v1-8k",
            "自定义（手动输入）",
        ]
        judge_model_choice = st.selectbox(
            label="Judge Model",
            options=judge_model_options,
            index=0,
            help="建议从预置模型中选择，避免空值被错误回退到 OpenAI。",
        )
        if judge_model_choice == "自定义（手动输入）":
            judge_model_id = st.text_input(
                label="Judge Model ID（自定义）",
                value="",
                placeholder="手动输入 Judge 模型 ID",
            ).strip()
        else:
            judge_model_id = judge_model_choice
        judge_api_key = st.text_input(
            label="Judge API Key",
            type="password",
            placeholder="输入 Judge API Key",
        ).strip()
        judge_base_url = st.text_input(
            label="Judge Base URL（可选）",
            value="",
            placeholder="例如：https://dashscope.aliyuncs.com/compatible-mode/v1",
            help="可与 Teacher/Student 使用不同平台地址。",
        ).strip()

        with st.expander("高级推理参数", expanded=False):
            st.caption("该参数对三种角色统一生效，Judge 会默认使用 temperature=0。")
            temperature = st.slider(
                label="Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
            )
            max_tokens = st.number_input(
                label="Max Tokens",
                min_value=128,
                max_value=8192,
                value=2048,
                step=128,
            )
            timeout_seconds = st.number_input(
                label="Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=90,
                step=5,
            )

        st.divider()
        st.subheader("输入文件")

        question_file = st.file_uploader(
            label="上传问题文件",
            type=["json", "jsonl", "txt"],
            help="建议字段：source_material, design_question, teaching_guideline",
        )
        rubric_file = st.file_uploader(
            label="上传 Rubric 评分文件",
            type=["txt", "md", "json"],
        )

        st.caption("提示：真实模式下需分别配置 Teacher/Student/Judge 的 API。")

    ui_config = UIConfig(
        run_mode=run_mode,
        teacher_model_id=teacher_model_id,
        student_model_id=student_model_id,
        judge_model_id=judge_model_id,
        teacher_api_key=teacher_api_key,
        student_api_key=student_api_key,
        judge_api_key=judge_api_key,
        teacher_base_url=teacher_base_url,
        student_base_url=student_base_url,
        judge_base_url=judge_base_url,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout_seconds=int(timeout_seconds),
    )

    return ui_config, question_file, rubric_file


# -----------------------------
# 主界面渲染
# -----------------------------
def main() -> None:
    """渲染 ADEPT Streamlit 页面，并执行 mock 评测流程。"""

    apply_custom_style()

    # 使用 session_state 保存最近一次运行记录，避免页面交互导致结果消失。
    if "last_records" not in st.session_state:
        st.session_state["last_records"] = []
    if "last_render_snapshot" not in st.session_state:
        st.session_state["last_render_snapshot"] = None

    # ===== Sidebar: 配置区 =====
    ui_config, question_file, rubric_file = render_sidebar()

    # ===== Main Area: 主干预区 =====
    st.title("ADEPT: 教学智能体评估大屏")
    st.markdown(
        '<div class="adept-intro">该页面用于可视化展示 Teacher-Student-Judge 三角色评测流程。'
        "当前版本重点优化了阅读对比度与信息层级，支持真实 API 与 Mock 双模式。"
        "</div>",
        unsafe_allow_html=True,
    )

    samples = load_question_samples(question_file)
    rubric_text = load_rubric_text(rubric_file)
    current_mode = "Real API" if ui_config.run_mode.startswith("Real") else "Mock"

    preview_col1, preview_col2, preview_col3 = st.columns([1.0, 1.0, 2.2], gap="large")
    with preview_col1:
        st.metric("待评测题目数", value=len(samples))
    with preview_col2:
        st.metric("当前运行模式", current_mode)
    with preview_col3:
        st.text_area(
            label="Rubric 预览",
            value=rubric_text,
            height=130,
            disabled=True,
        )

    control_col1, control_col2 = st.columns([1.2, 2.8], gap="large")
    with control_col1:
        run_clicked = st.button(
            "开始评测 (Run Benchmark)",
            type="primary",
            width="stretch",
        )
    with control_col2:
        st.markdown(
            '<div class="adept-note">运行建议：先在侧边栏确认模型与文件配置；'
            "真实模式下请确保三个角色的 API Key / Base URL 可用。"
            "评测过程中可折叠区块查看重点信息，减少界面干扰。</div>",
            unsafe_allow_html=True,
        )

    # 预先创建占位组件，后续在运行时动态刷新。
    progress_bar = st.progress(0, text="等待开始评测")
    status_placeholder = st.empty()

    flow_left_col, flow_right_col = st.columns([2.2, 1.2], gap="large")

    with flow_left_col:
        with st.expander("区块 A: 当前题目", expanded=True):
            st.caption("展示当前样本的 Source Material 与 Design Question。")
            block_a_placeholder = st.empty()

        with st.expander("区块 B: 师生互动", expanded=True):
            st.caption("并排查看学生初答、教师指导和学生最终回答。")
            col_base, col_teacher, col_final = st.columns(3, gap="medium")
            col_base.markdown("##### S_base")
            col_teacher.markdown("##### Knowledge")
            col_final.markdown("##### S_knowledge")
            base_placeholder = col_base.empty()
            teacher_placeholder = col_teacher.empty()
            final_placeholder = col_final.empty()

    with flow_right_col:
        with st.expander("区块 C: LLM 裁判", expanded=True):
            st.caption("基于 Rubric 输出评分及增益（Delta Score）。")
            metric_col1, metric_col2, metric_col3 = st.columns(3, gap="small")
            metric_base_placeholder = metric_col1.empty()
            metric_knowledge_placeholder = metric_col2.empty()
            metric_delta_placeholder = metric_col3.empty()
            judge_reason_placeholder = st.empty()

    last_snapshot = st.session_state.get("last_render_snapshot")
    if isinstance(last_snapshot, dict):
        block_a_placeholder.markdown(str(last_snapshot.get("sample_markdown", "等待开始评测后展示当前题目内容。")))
        render_scrollable_text(base_placeholder, str(last_snapshot.get("base_answer", "等待生成学生基线回答。")))
        render_scrollable_text(teacher_placeholder, str(last_snapshot.get("teacher_knowledge", "等待生成教师教学指导。")))
        render_scrollable_text(final_placeholder, str(last_snapshot.get("improved_answer", "等待生成学生最终回答。")))

        judge_result_snapshot = last_snapshot.get("judge_result")
        if isinstance(judge_result_snapshot, dict):
            metric_base_placeholder.metric("S_base", int(judge_result_snapshot.get("s_base", 0)))
            metric_knowledge_placeholder.metric("S_knowledge", int(judge_result_snapshot.get("s_knowledge", 0)))
            metric_delta_placeholder.metric("Delta Score", int(judge_result_snapshot.get("delta", 0)))
            judge_reason_placeholder.markdown(
                f"**Rubric 评分说明**\n\n{judge_result_snapshot.get('reason', '评分接口未提供理由')}"
            )
        else:
            judge_reason_placeholder.markdown("Rubric 评分说明将在评测后显示。")
    else:
        block_a_placeholder.markdown("等待开始评测后展示当前题目内容。")
        render_scrollable_text(base_placeholder, "等待生成学生基线回答。")
        render_scrollable_text(teacher_placeholder, "等待生成教师教学指导。")
        render_scrollable_text(final_placeholder, "等待生成学生最终回答。")
        judge_reason_placeholder.markdown("Rubric 评分说明将在评测后显示。")

    # 若用户点击运行，则执行 mock 评测。
    if run_clicked:
        # 仅在真实模式要求必须提供 API Key 与三角色 Model ID。
        if ui_config.run_mode.startswith("Real"):
            missing_fields: list[str] = []
            if not ui_config.teacher_model_id.strip():
                missing_fields.append("Teacher Model ID")
            if not ui_config.teacher_api_key.strip():
                missing_fields.append("Teacher API Key")
            if not ui_config.student_model_id.strip():
                missing_fields.append("Student Model ID")
            if not ui_config.student_api_key.strip():
                missing_fields.append("Student API Key")
            if not ui_config.judge_model_id.strip():
                missing_fields.append("Judge Model ID")
            if not ui_config.judge_api_key.strip():
                missing_fields.append("Judge API Key")

            if missing_fields:
                st.warning("请先补全真实模式配置：" + "、".join(missing_fields))
                return

        records: list[dict[str, Any]] = []
        total_steps = max(1, len(samples) * 5)
        current_step = 0
        api_bundle = None
        loop: asyncio.AbstractEventLoop | None = None

        with st.spinner("正在执行 ADEPT 评测流程，请稍候..."):
            try:
                if ui_config.run_mode.startswith("Real"):
                    teacher_cfg = build_model_config_from_ui(
                        role="teacher",
                        model_id=ui_config.teacher_model_id,
                        api_key=ui_config.teacher_api_key,
                        base_url_input=ui_config.teacher_base_url,
                        temperature=ui_config.temperature,
                        max_tokens=ui_config.max_tokens,
                        timeout_seconds=ui_config.timeout_seconds,
                    )
                    student_cfg = build_model_config_from_ui(
                        role="student",
                        model_id=ui_config.student_model_id,
                        api_key=ui_config.student_api_key,
                        base_url_input=ui_config.student_base_url,
                        temperature=ui_config.temperature,
                        max_tokens=ui_config.max_tokens,
                        timeout_seconds=ui_config.timeout_seconds,
                    )
                    judge_cfg = build_model_config_from_ui(
                        role="judge",
                        model_id=ui_config.judge_model_id,
                        api_key=ui_config.judge_api_key,
                        base_url_input=ui_config.judge_base_url,
                        temperature=0.0,
                        max_tokens=ui_config.max_tokens,
                        timeout_seconds=ui_config.timeout_seconds,
                    )

                    role_configs = {
                        "Teacher": teacher_cfg,
                        "Student": student_cfg,
                        "Judge": judge_cfg,
                    }
                    same_key_pairs, same_endpoint_pairs = analyze_role_config_overlap(role_configs)
                    if same_key_pairs:
                        st.info(
                            "检测到复用同一 API Key 的角色："
                            + "、".join(same_key_pairs)
                            + "。这不会被系统自动判定为同一模型输出；"
                            "是否同模型取决于 Provider / Model ID / Base URL。"
                        )
                    if same_endpoint_pairs:
                        st.warning(
                            "检测到以下角色指向同一模型端点："
                            + "、".join(same_endpoint_pairs)
                            + "。这会导致输出风格高度接近，建议至少区分模型或平台地址。"
                        )

                    app_config = AppConfig(
                        teacher=teacher_cfg,
                        student=student_cfg,
                        judge=judge_cfg,
                        runtime=RuntimeConfig(concurrency=1),
                    )

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    api_bundle = build_api_bundle(app_config, rubric=rubric_text)

                for idx, sample in enumerate(samples, start=1):
                    status_placeholder.info(f"正在处理第 {idx}/{len(samples)} 题")

                    # Step 1: 展示当前题目（区块 A）
                    sample_markdown = render_sample_markdown(sample)
                    block_a_placeholder.markdown(sample_markdown)
                    current_step += 1
                    progress_bar.progress(int(current_step / total_steps * 100), text="加载当前题目")

                    if ui_config.run_mode.startswith("Real"):
                        assert loop is not None
                        assert api_bundle is not None

                        # Step 2: 真实 API 生成 Student 基线回答
                        baseline_result = run_coroutine(
                            loop,
                            api_bundle.student_api.answer_baseline(
                                source_material=sample.source_material,
                                design_question=sample.design_question,
                            ),
                        )
                        base_answer = str(baseline_result.answer)
                        render_scrollable_text(base_placeholder, base_answer)
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="Real API：生成基线回答")

                        # Step 3: 真实 API 生成 Teacher 教学指导
                        teacher_result = run_coroutine(
                            loop,
                            api_bundle.teacher_api.teach(
                                source_material=sample.source_material,
                                teaching_guideline=sample.teaching_guideline,
                                design_question=sample.design_question,
                            ),
                        )
                        teacher_knowledge = str(teacher_result.answer)
                        render_scrollable_text(teacher_placeholder, teacher_knowledge)
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="Real API：生成教学指导")

                        # Step 4: 真实 API 生成 Student 最终回答
                        final_result = run_coroutine(
                            loop,
                            api_bundle.student_api.answer_intervention(
                                source_material=sample.source_material,
                                teacher_output=teacher_knowledge,
                                design_question=sample.design_question,
                            ),
                        )
                        improved_answer = str(final_result.answer)
                        render_scrollable_text(final_placeholder, improved_answer)
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="Real API：生成最终回答")

                        # Step 5: 真实 API Rubric 评分
                        base_judge = run_coroutine(
                            loop,
                            api_bundle.rubric_api.evaluate(
                                source_material=sample.source_material,
                                design_question=sample.design_question,
                                student_answer=base_answer,
                            ),
                        )
                        knowledge_judge = run_coroutine(
                            loop,
                            api_bundle.rubric_api.evaluate(
                                source_material=sample.source_material,
                                design_question=sample.design_question,
                                student_answer=improved_answer,
                            ),
                        )

                        s_base = int(base_judge.get("score", 0))
                        s_knowledge = int(knowledge_judge.get("score", 0))
                        judge_result = {
                            "s_base": s_base,
                            "s_knowledge": s_knowledge,
                            "delta": s_knowledge - s_base,
                            "reason": str(knowledge_judge.get("reason", "评分接口未提供理由")),
                        }
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="Real API：完成裁判评分")

                    else:
                        # Step 2: 生成 Student 基线回答（区块 B - Column 1）
                        base_answer = mock_student_baseline(sample, ui_config.student_model_id)
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="生成基线回答")

                        # Step 3: 生成 Teacher 教学指导（区块 B - Column 2）
                        teacher_knowledge = mock_teacher_guidance(sample, ui_config.teacher_model_id)
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="生成教学指导")

                        # Step 4: 生成 Student 最终回答（区块 B - Column 3）
                        improved_answer = mock_student_final(sample, teacher_knowledge, ui_config.student_model_id)
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="生成最终回答")

                        # Step 5: 裁判评分（区块 C）
                        judge_result = mock_rubric_judge(
                            base_answer=base_answer,
                            improved_answer=improved_answer,
                            rubric_text=rubric_text,
                            judge_model_id=ui_config.judge_model_id or "judge-auto",
                        )
                        current_step += 1
                        progress_bar.progress(int(current_step / total_steps * 100), text="完成裁判评分")

                        render_scrollable_text(base_placeholder, base_answer)
                        render_scrollable_text(teacher_placeholder, teacher_knowledge)
                        render_scrollable_text(final_placeholder, improved_answer)

                    # 区块 C: 评分展示
                    metric_base_placeholder.metric("S_base", judge_result["s_base"])
                    metric_knowledge_placeholder.metric("S_knowledge", judge_result["s_knowledge"])
                    metric_delta_placeholder.metric("Delta Score", judge_result["delta"])
                    judge_reason_placeholder.markdown(
                        f"**Rubric 评分说明**\n\n{judge_result['reason']}"
                    )

                    st.session_state["last_render_snapshot"] = {
                        "sample_markdown": sample_markdown,
                        "base_answer": base_answer,
                        "teacher_knowledge": teacher_knowledge,
                        "improved_answer": improved_answer,
                        "judge_result": judge_result,
                    }

                    records.append(
                        {
                            "sample_index": idx,
                            "s_base": judge_result["s_base"],
                            "s_knowledge": judge_result["s_knowledge"],
                            "delta_score": judge_result["delta"],
                            "run_mode": ui_config.run_mode,
                        }
                    )

            finally:
                if api_bundle is not None and loop is not None:
                    run_coroutine(loop, api_bundle.close())
                if loop is not None:
                    asyncio.set_event_loop(None)
                    loop.close()

        progress_bar.progress(100, text="评测完成")
        status_placeholder.success("评测流程执行完毕。")
        if records:
            st.session_state["last_records"] = records
        else:
            status_placeholder.warning("本次运行未生成有效记录，已保留上次评测结果。")

    # 无论是否刚刚运行，只要 session_state 中有结果，就显示汇总区。
    if st.session_state["last_records"]:
        results = st.session_state["last_records"]
        avg_delta = sum(row["delta_score"] for row in results) / len(results)
        positive_ratio = (
            sum(1 for row in results if row["delta_score"] > 0) / len(results)
            if results
            else 0.0
        )

        st.subheader("评测结果汇总")
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        sum_col1.metric("样本数量", len(results))
        sum_col2.metric("平均 Delta", f"{avg_delta:.2f}")
        sum_col3.metric("提升占比", f"{positive_ratio:.0%}")

        try:
            st.dataframe(results, width="stretch", hide_index=True)
        except Exception:
            st.dataframe(results, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
