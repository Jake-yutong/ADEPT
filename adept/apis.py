from __future__ import annotations

import ast
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .models import AppConfig, LLMClient, LLMClientFactory
from .orchestrator import (
    ADEPTOrchestrator,
    DefaultPromptTemplateEngine,
    JsonlLogger,
    PromptTemplateEngine,
)


DEFAULT_RUBRIC = """
1) 问题理解（0-25）：是否准确识别题目目标与约束。
2) 设计策略（0-25）：方案是否有结构、步骤与可执行性。
3) 论证质量（0-25）：理由是否充分、逻辑是否一致。
4) 表达清晰（0-25）：语言组织是否清晰、便于教学评估。
""".strip()


@dataclass(slots=True)
class APICallResult:
    """teacher/student API 的统一返回结构。"""

    prompt: str
    answer: str


class TeacherAPI:
    """Teacher API：根据素材与教学大纲生成教学辅导。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_engine: PromptTemplateEngine | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.prompt_engine = prompt_engine or DefaultPromptTemplateEngine()
        self.system_prompt = system_prompt

    async def teach(self, *, source_material: str, teaching_guideline: str) -> APICallResult:
        prompt = self.prompt_engine.render_teacher_prompt(
            source_material=source_material,
            teaching_guideline=teaching_guideline,
        )
        answer = await self.client.generate(prompt, system_prompt=self.system_prompt)
        return APICallResult(prompt=prompt, answer=answer)


class StudentAPI:
    """Student API：提供 baseline 与 intervention 两种作答能力。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_engine: PromptTemplateEngine | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.prompt_engine = prompt_engine or DefaultPromptTemplateEngine()
        self.system_prompt = system_prompt

    async def answer_baseline(self, *, source_material: str, design_question: str) -> APICallResult:
        prompt = self.prompt_engine.render_student_baseline_prompt(
            source_material=source_material,
            design_question=design_question,
        )
        answer = await self.client.generate(prompt, system_prompt=self.system_prompt)
        return APICallResult(prompt=prompt, answer=answer)

    async def answer_intervention(
        self,
        *,
        source_material: str,
        teacher_output: str,
        design_question: str,
    ) -> APICallResult:
        prompt = self.prompt_engine.render_student_intervention_prompt(
            source_material=source_material,
            teacher_output=teacher_output,
            design_question=design_question,
        )
        answer = await self.client.generate(prompt, system_prompt=self.system_prompt)
        return APICallResult(prompt=prompt, answer=answer)


class RubricScoringAPI:
    """Rubric 评分 API：基于 rubric 输出结构化 score/reason。"""

    _json_fence_pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

    def __init__(
        self,
        client: LLMClient,
        *,
        rubric: str | None = None,
        min_score: int = 0,
        max_score: int = 100,
        system_prompt: str | None = "你是严格的设计教育评分专家，请仅返回合法 JSON。",
    ) -> None:
        if min_score > max_score:
            raise ValueError("min_score 不能大于 max_score")

        self.client = client
        self.rubric = (rubric or DEFAULT_RUBRIC).strip()
        self.min_score = min_score
        self.max_score = max_score
        self.system_prompt = system_prompt

    async def evaluate(
        self,
        *,
        source_material: str,
        design_question: str,
        student_answer: str,
        reference_answer: str | None = None,
    ) -> dict[str, Any]:
        prompt = self._render_prompt(
            source_material=source_material,
            design_question=design_question,
            student_answer=student_answer,
            reference_answer=reference_answer,
        )

        raw_text = await self.client.generate(prompt, system_prompt=self.system_prompt)
        try:
            parsed_json = self._extract_json_object(raw_text)
        except ValueError:
            # 首轮返回不合规时再补一次强约束请求，降低失败率。
            repair_prompt = (
                prompt
                + "\n\n请严格按要求只返回 JSON 对象，不要包含解释、代码块或其他文本。"
            )
            raw_text = await self.client.generate(repair_prompt, system_prompt=self.system_prompt)
            parsed_json = self._extract_json_object(raw_text)

        score = self._parse_score(parsed_json.get("score"))

        reason_raw = parsed_json.get("reason")
        reason = str(reason_raw or "").strip() or "评分接口未提供理由"

        return {
            "score": score,
            "reason": reason,
            "raw": {
                "response_text": raw_text,
                "parsed_json": parsed_json,
            },
        }

    def _parse_score(self, score_raw: Any) -> int:
        try:
            score = int(score_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Rubric 评分结果中的 score 非法: {score_raw}") from exc

        if score < self.min_score or score > self.max_score:
            raise ValueError(
                f"Rubric 评分结果超出范围: {score}，期望区间 [{self.min_score}, {self.max_score}]"
            )
        return score

    @classmethod
    def _extract_json_object(cls, raw_text: str) -> dict[str, Any]:
        candidates: list[str] = []
        stripped = raw_text.strip()
        if stripped:
            candidates.append(stripped)

        fence_match = cls._json_fence_pattern.search(raw_text)
        if fence_match:
            candidates.append(fence_match.group(1).strip())

        balanced_object = cls._find_balanced_object(raw_text)
        if balanced_object:
            candidates.append(balanced_object)

        seen: set[str] = set()
        unique_candidates = [item for item in candidates if not (item in seen or seen.add(item))]

        for candidate in unique_candidates:
            parsed_mapping = cls._parse_mapping_candidate(candidate)
            if parsed_mapping is not None:
                return parsed_mapping

        fallback = cls._extract_score_reason_fallback(raw_text)
        if fallback is not None:
            return fallback

        raise ValueError("Rubric 评分 API 返回内容无法解析为 JSON 对象")

    @staticmethod
    def _parse_mapping_candidate(candidate: str) -> dict[str, Any] | None:
        """优先按 JSON 解析，失败后尝试解析 Python 字面量字典。"""

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                return None

        if isinstance(data, Mapping):
            return dict(data)
        return None

    @staticmethod
    def _extract_score_reason_fallback(raw_text: str) -> dict[str, Any] | None:
        """当返回非标准 JSON 时，启发式提取 score/reason。"""

        score_patterns = [
            r'"score"\s*[:：]\s*(-?\d+)',
            r"\bscore\b\s*[:：]\s*(-?\d+)",
            r"分数\s*[:：]\s*(-?\d+)",
            r"得分\s*[:：]\s*(-?\d+)",
        ]
        score: int | None = None
        for pattern in score_patterns:
            matched = re.search(pattern, raw_text, re.IGNORECASE)
            if matched:
                score = int(matched.group(1))
                break

        if score is None:
            return None

        reason_patterns = [
            r'"reason"\s*[:：]\s*"([^"]+)"',
            r'"reason"\s*[:：]\s*\'([^\']+)\'',
            r"\breason\b\s*[:：]\s*([^\n\r]+)",
            r"理由\s*[:：]\s*([^\n\r]+)",
            r"评语\s*[:：]\s*([^\n\r]+)",
        ]
        reason = ""
        for pattern in reason_patterns:
            matched = re.search(pattern, raw_text, re.IGNORECASE)
            if matched:
                reason = matched.group(1).strip().strip('"\'`')
                break

        if not reason:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            filtered = [
                line
                for line in lines
                if not re.search(r"score|分数|得分|```|\{|\}", line, re.IGNORECASE)
            ]
            if filtered:
                reason = filtered[0][:200]

        if not reason:
            reason = "模型返回了非标准 JSON，已启发式解析。"

        return {"score": score, "reason": reason}

    @staticmethod
    def _find_balanced_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False

        for idx in range(start, len(text)):
            char = text[idx]
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        return None

    def _render_prompt(
        self,
        *,
        source_material: str,
        design_question: str,
        student_answer: str,
        reference_answer: str | None,
    ) -> str:
        reference_text = reference_answer if reference_answer else "无"

        return (
            "请你基于 Rubric 对学生答案评分。\n\n"
            "【Rubric】\n"
            f"{self.rubric}\n\n"
            "【设计素材】\n"
            f"{source_material}\n\n"
            "【设计问题】\n"
            f"{design_question}\n\n"
            "【学生答案】\n"
            f"{student_answer}\n\n"
            "【参考答案】\n"
            f"{reference_text}\n\n"
            "输出要求：\n"
            f"1) score 必须是 {self.min_score}-{self.max_score} 的整数\n"
            "2) reason 必须为简洁中文说明\n"
            "3) 只返回 JSON，不要附加任何其他文字\n"
            f'JSON 格式: {{"score": {self.min_score}, "reason": "..."}}'
        )


@dataclass(slots=True)
class APIBundle:
    """ADEPT 三类 API 与底层 client 的打包对象。"""

    teacher_api: TeacherAPI
    student_api: StudentAPI
    rubric_api: RubricScoringAPI
    teacher_client: LLMClient
    student_client: LLMClient
    judge_client: LLMClient

    async def close(self) -> None:
        await asyncio.gather(
            self.teacher_client.close(),
            self.student_client.close(),
            self.judge_client.close(),
        )


def build_api_bundle(
    config: AppConfig,
    *,
    prompt_engine: PromptTemplateEngine | None = None,
    rubric: str | None = None,
) -> APIBundle:
    """基于 AppConfig 构建 teacher/student/rubric 三类 API。"""

    teacher_client = LLMClientFactory.create_client(config.teacher)
    student_client = LLMClientFactory.create_client(config.student)
    judge_client = LLMClientFactory.create_client(config.judge)

    teacher_api = TeacherAPI(teacher_client, prompt_engine=prompt_engine)
    student_api = StudentAPI(student_client, prompt_engine=prompt_engine)
    rubric_api = RubricScoringAPI(judge_client, rubric=rubric)

    return APIBundle(
        teacher_api=teacher_api,
        student_api=student_api,
        rubric_api=rubric_api,
        teacher_client=teacher_client,
        student_client=student_client,
        judge_client=judge_client,
    )


def build_orchestrator_with_apis(
    config: AppConfig,
    *,
    logger: JsonlLogger | None = None,
    prompt_engine: PromptTemplateEngine | None = None,
    rubric: str | None = None,
) -> tuple[ADEPTOrchestrator, APIBundle]:
    """构建已接入 teacher/student/rubric API 的 orchestrator。"""

    api_bundle = build_api_bundle(config, prompt_engine=prompt_engine, rubric=rubric)
    orchestrator_logger = logger or JsonlLogger(config.runtime.output_path)

    orchestrator = ADEPTOrchestrator(
        teacher_client=api_bundle.teacher_client,
        student_client=api_bundle.student_client,
        evaluator=api_bundle.rubric_api,
        logger=orchestrator_logger,
        prompt_engine=prompt_engine,
        max_concurrency=config.runtime.concurrency,
        teacher_api=api_bundle.teacher_api,
        student_api=api_bundle.student_api,
    )

    return orchestrator, api_bundle
