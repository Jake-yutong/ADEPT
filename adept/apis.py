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

    async def teach(self, *, source_material: str, teaching_guideline: str, design_question: str = "") -> APICallResult:
        prompt = self.prompt_engine.render_teacher_prompt(
            source_material=source_material,
            teaching_guideline=teaching_guideline,
            design_question=design_question,
        )
        answer = await self.client.generate(prompt, system_prompt=self.system_prompt)
        return APICallResult(prompt=prompt, answer=answer)


class StudentAPI:
    """Student API：提供 baseline 与 intervention 两种作答能力。"""

    _subquestion_pattern = re.compile(
        r"(?:^|\n)\s*(?:第\s*\d+\s*[题问小]|[（(]?\d+[）)\.、:\:]|[一二三四五六七八九十]+[、\.\s]|Q\d+)"
    )
    _choice_pattern = re.compile(
        r"选择题|单选|多选|判断题"
        r"|(?:^|[\s\n])[A-Da-d][\.\.、\)）\s]"
        r"|选项[：:]"
        r"|以下.*(?:正确|错误|合适|不正确|不合适)的是",
        re.MULTILINE,
    )

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
        answer = await self._maybe_rewrite_for_coverage(
            source_material=source_material,
            design_question=design_question,
            current_answer=answer,
            teacher_output=None,
        )
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
        answer = await self._maybe_rewrite_for_coverage(
            source_material=source_material,
            design_question=design_question,
            current_answer=answer,
            teacher_output=teacher_output,
        )
        return APICallResult(prompt=prompt, answer=answer)

    @classmethod
    def _needs_coverage_pass(cls, design_question: str) -> bool:
        text = design_question.strip()
        if not text:
            return False

        numbered_count = len(cls._subquestion_pattern.findall(text))
        question_mark_count = text.count("?") + text.count("？")
        has_choice = bool(cls._choice_pattern.search(text))
        # 降低门槛：只要有子问题编号、多个问号、或任何选择题元素即触发
        return numbered_count >= 1 or question_mark_count >= 2 or has_choice

    @staticmethod
    def _build_coverage_rewrite_prompt(
        *,
        source_material: str,
        design_question: str,
        current_answer: str,
        teacher_output: str | None,
    ) -> str:
        sections = [
            "检查当前答案是否完整覆盖了题目中的选择题和简答题。",
            "",
            "【设计素材】",
            source_material,
            "",
            "【题目】",
            design_question,
        ]

        if teacher_output is not None:
            sections.extend(["", "【导师辅导内容】", teacher_output])

        sections.extend(
            [
                "",
                "【当前答案】",
                current_answer,
                "",
                "修订要求：",
                "1) 若当前答案已包含选择题答案和简答题答案，原样返回；",
                "2) 若选择题未作答，补上选项字母；",
                "3) 若简答题未作答或不完整，补全答案（100字以内）；",
                "4) 回答格式：选择题：X  简答题：（答案）",
                "5) 只返回最终答案，不要解释检查过程。",
            ]
        )

        return "\n".join(sections)

    async def _maybe_rewrite_for_coverage(
        self,
        *,
        source_material: str,
        design_question: str,
        current_answer: str,
        teacher_output: str | None,
    ) -> str:
        if not self._needs_coverage_pass(design_question):
            return current_answer

        rewrite_prompt = self._build_coverage_rewrite_prompt(
            source_material=source_material,
            design_question=design_question,
            current_answer=current_answer,
            teacher_output=teacher_output,
        )

        revised_answer = await self.client.generate(rewrite_prompt, system_prompt=self.system_prompt)
        revised_answer = revised_answer.strip()
        return revised_answer or current_answer


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
            try:
                parsed_json = self._extract_json_object(raw_text)
            except ValueError:
                # 第二轮仍失败，用极简 prompt 做最后尝试
                minimal_prompt = (
                    "请对以下学生答案打分，只返回一个JSON对象，格式为 "
                    '{"score": 整数, "reason": "理由"}，'
                    f"分数范围 {self.min_score}-{self.max_score}。\n\n"
                    f"学生答案：{student_answer[:500]}\n\n"
                    "只返回JSON，不要任何其他文字。"
                )
                raw_text = await self.client.generate(
                    minimal_prompt, system_prompt="只返回合法JSON。"
                )
                try:
                    parsed_json = self._extract_json_object(raw_text)
                except ValueError:
                    # 所有尝试均失败，返回降级结果而非崩溃
                    return {
                        "score": 0,
                        "reason": "评分接口多次返回非法格式，无法解析，已降级为 0 分。",
                        "raw": {
                            "response_text": raw_text,
                            "parsed_json": None,
                            "parse_error": True,
                        },
                    }

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
    def _looks_like_placeholder_reason(reason: str) -> bool:
        normalized = reason.strip().strip('"\'`').replace(" ", "")
        return normalized in {"", "...", "…", "待补充", "请填写理由", "理由"}

    @classmethod
    def _parse_mapping_candidate(cls, candidate: str) -> dict[str, Any] | None:
        """优先按 JSON 解析，失败后尝试解析 Python 字面量字典。"""

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                return None

        if not isinstance(data, Mapping):
            return None

        mapping = {str(k): v for k, v in data.items()}

        score_value: Any | None = None
        for key in ("score", "分数", "得分"):
            if key in mapping:
                score_value = mapping[key]
                break

        if score_value is None:
            return None

        reason_value: Any = ""
        for key in ("reason", "理由", "评语", "analysis", "comment"):
            if key in mapping:
                reason_value = mapping[key]
                break

        score_text = str(score_value).strip()
        reason_text = str(reason_value).strip()

        if not score_text:
            return None
        if score_text.startswith("<") and score_text.endswith(">"):
            return None

        # 避免把提示词模板中的占位 JSON 误判为评分结果。
        if cls._looks_like_placeholder_reason(reason_text) and score_text in {"0", "0.0"}:
            return None

        return {
            "score": score_value,
            "reason": reason_text,
        }

    @staticmethod
    def _is_template_context(raw_text: str, start_idx: int, end_idx: int) -> bool:
        left = raw_text[max(0, start_idx - 24) : start_idx]
        right = raw_text[end_idx : min(len(raw_text), end_idx + 24)]
        compact_context = (left + right).lower().replace(" ", "")
        if "json格式" in compact_context:
            return True
        return False

    @staticmethod
    def _extract_score_reason_fallback(raw_text: str) -> dict[str, Any] | None:
        """当返回非标准 JSON 时，启发式提取 score/reason。"""

        score_patterns = [
            r'"score"\s*[:：]\s*(-?\d+)(?!\s*[-~到至]\s*\d+)',
            r"\bscore\b\s*[:：]\s*(-?\d+)(?!\s*[-~到至]\s*\d+)",
            r"分数\s*[:：]\s*(-?\d+)(?!\s*[-~到至]\s*\d+)",
            r"得分\s*[:：]\s*(-?\d+)(?!\s*[-~到至]\s*\d+)",
        ]
        score_candidates: list[tuple[int, int]] = []
        for pattern in score_patterns:
            for matched in re.finditer(pattern, raw_text, re.IGNORECASE):
                if RubricScoringAPI._is_template_context(raw_text, matched.start(), matched.end()):
                    continue
                score_candidates.append((matched.start(), int(matched.group(1))))

        if not score_candidates:
            return None

        score_candidates.sort(key=lambda item: item[0])
        score = score_candidates[-1][1]

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

        if RubricScoringAPI._looks_like_placeholder_reason(reason):
            return None

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

        # 尝试提取与当前题目匹配的局部 rubric
        per_question_rubric = self._extract_matching_rubric(design_question)

        return (
            "请你作为严格的设计教育评分专家，基于 Rubric 对学生答案进行逐项评分。\n\n"
            "【评分标准 Rubric】\n"
            f"{per_question_rubric}\n\n"
            "【设计素材】\n"
            f"{source_material}\n\n"
            "【题目】\n"
            f"{design_question}\n\n"
            "【学生答案】\n"
            f"{student_answer}\n\n"
            "【参考答案】\n"
            f"{reference_text}\n\n"
            "评分指南：\n"
            "1) 选择题（30分）：答对得30分，答错得0分，未作答得0分；\n"
            "2) 简答题（70分）：按 Rubric 中各维度逐项评分后累加；\n"
            "3) 总分 = 选择题得分 + 简答题得分；\n"
            "4) 必须检查：学生是否回答了选择题？是否回答了简答题？漏答则该部分0分；\n"
            "5) 简答题超过100字酌情扣5-15分；\n\n"
            "输出要求：\n"
            f"score 必须是 {self.min_score}-{self.max_score} 的整数，"
            "reason 必须为简洁中文说明（含各部分得分明细）。\n"
            "只返回 JSON，不要附加任何其他文字、代码块标记或解释。\n"
            "JSON 仅包含两个键：score（整数）和 reason（字符串）。\n\n"
            '输出示例：{"score": 75, "reason": "选择题30分（答对B）；简答题45分：概念准确性25/35，案例相关性15/25，表达质量5/10。"}'
        )

    def _extract_matching_rubric(self, design_question: str) -> str:
        """从完整 rubric 中提取与当前题目匹配的局部评分标准。"""

        # 从 design_question 中提取题号
        q_match = re.search(r"QUESTION\s+(\d+)", design_question)
        if not q_match:
            return self.rubric

        q_num = q_match.group(1)

        # 在 rubric 中查找对应的 RUBRIC N 或 QUESTION N 段落
        patterns = [
            rf"(?:RUBRIC\s+{q_num}|QUESTION\s+{q_num})\b",
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, self.rubric, re.IGNORECASE))
            if not matches:
                continue

            start = matches[0].start()
            # 向前找到段落开头（分隔线）
            line_start = self.rubric.rfind("\n", 0, start)
            if line_start < 0:
                line_start = 0
            section_start = self.rubric.rfind("---", 0, line_start)
            if section_start < 0:
                section_start = line_start

            # 向后找到下一个 RUBRIC 段落或文件末尾
            next_rubric = re.search(
                rf"(?:^-{{3,}}\s*\n\s*RUBRIC\s+(?!{q_num}\b))",
                self.rubric[matches[0].end():],
                re.MULTILINE,
            )
            if next_rubric:
                section_end = matches[0].end() + next_rubric.start()
            else:
                # 也尝试找下一个 PART 分隔
                next_part = re.search(
                    r"^={3,}",
                    self.rubric[matches[0].end():],
                    re.MULTILINE,
                )
                if next_part:
                    section_end = matches[0].end() + next_part.start()
                else:
                    section_end = len(self.rubric)

            extracted = self.rubric[section_start:section_end].strip()
            if len(extracted) > 50:
                return extracted

        return self.rubric


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
