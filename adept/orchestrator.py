from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, NotRequired, Protocol, Sequence, TypedDict

from .models import LLMClient


class DatasetRecord(TypedDict):
    """单条数据样本结构定义。"""

    source_material: str
    teaching_guideline: str
    design_question: str
    reference_answer: NotRequired[str]
    sample_id: NotRequired[str]
    meta: NotRequired[dict[str, Any]]


@dataclass(slots=True)
class JudgeResult:
    """评估器返回的结构化评分结果。"""

    score: int
    reason: str
    raw: dict[str, Any] | None = None


class EvaluatorProtocol(Protocol):
    """Evaluator 协议。

    具体实现可在后续 evaluator.py 中接入 LLM-as-a-Judge。
    """

    async def evaluate(
        self,
        *,
        source_material: str,
        design_question: str,
        student_answer: str,
        reference_answer: str | None = None,
    ) -> JudgeResult | Mapping[str, Any]:
        ...


class PromptTemplateEngine(Protocol):
    """Prompt 模板引擎协议。"""

    def render_teacher_prompt(self, source_material: str, teaching_guideline: str) -> str:
        ...

    def render_student_baseline_prompt(self, source_material: str, design_question: str) -> str:
        ...

    def render_student_intervention_prompt(
        self,
        source_material: str,
        teacher_output: str,
        design_question: str,
    ) -> str:
        ...


class DefaultPromptTemplateEngine:
    """默认 Prompt 模板实现。

    该实现基于字符串模板，便于后续替换为 Jinja2 引擎。
    """

    def render_teacher_prompt(self, source_material: str, teaching_guideline: str) -> str:
        return (
            "你是一位设计教育导师，请依据以下素材和教学大纲，为学生生成可执行的学习辅导。\n\n"
            "【设计素材】\n"
            f"{source_material}\n\n"
            "【教学大纲】\n"
            f"{teaching_guideline}\n\n"
            "请输出：\n"
            "1) 关键概念拆解\n"
            "2) 解题步骤建议\n"
            "3) 常见误区提醒\n"
            "4) 作答质量检查清单\n"
        )

    def render_student_baseline_prompt(self, source_material: str, design_question: str) -> str:
        return (
            "请你作为设计专业学生，仅根据提供的设计素材回答问题。\n\n"
            "【设计素材】\n"
            f"{source_material}\n\n"
            "【设计问题】\n"
            f"{design_question}\n\n"
            "要求：结构清晰，给出可落地的设计思路与理由。"
        )

    def render_student_intervention_prompt(
        self,
        source_material: str,
        teacher_output: str,
        design_question: str,
    ) -> str:
        return (
            "请你作为设计专业学生，先学习导师辅导，再回答同一个设计问题。\n\n"
            "【设计素材】\n"
            f"{source_material}\n\n"
            "【导师辅导内容】\n"
            f"{teacher_output}\n\n"
            "【设计问题】\n"
            f"{design_question}\n\n"
            "要求：显式体现你如何应用导师辅导提升答案质量。"
        )


class JsonlLogger:
    """并发安全的 JSON Lines 记录器。"""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, payload: Mapping[str, Any]) -> None:
        """将单条记录追加写入 .jsonl。"""

        line = json.dumps(payload, ensure_ascii=False)
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass(slots=True)
class SampleRunResult:
    """单样本运行结果。"""

    index: int
    sample_id: str
    s_base: int | None
    s_knowledge: int | None
    delta_score: int | None
    status: str
    error: str | None


class ADEPTOrchestrator:
    """ADEPT 主流程编排器。

    流程顺序：
    1) Baseline（学生直接回答）
    2) Teaching（教师生成辅导）
    3) Intervention（学生带辅导再次回答）
    4) Evaluation（Judge 打分并计算 Delta）
    """

    def __init__(
        self,
        *,
        teacher_client: LLMClient,
        student_client: LLMClient,
        evaluator: EvaluatorProtocol,
        logger: JsonlLogger,
        prompt_engine: PromptTemplateEngine | None = None,
        max_concurrency: int = 5,
    ) -> None:
        self.teacher_client = teacher_client
        self.student_client = student_client
        self.evaluator = evaluator
        self.logger = logger
        self.prompt_engine = prompt_engine or DefaultPromptTemplateEngine()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, dataset: Sequence[DatasetRecord]) -> dict[str, Any]:
        """并发执行整个数据集，并返回汇总信息。"""

        tasks = [
            asyncio.create_task(self._run_one_sample(index=i, item=item))
            for i, item in enumerate(dataset)
        ]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r.delta_score is not None]
        deltas = [r.delta_score for r in successful if r.delta_score is not None]

        summary = {
            "total_samples": len(dataset),
            "successful_samples": len(successful),
            "failed_samples": len(dataset) - len(successful),
            "avg_delta_score": round(mean(deltas), 2) if deltas else None,
            "max_delta_score": max(deltas) if deltas else None,
            "min_delta_score": min(deltas) if deltas else None,
        }

        return {
            "summary": summary,
            "results": [asdict(item) for item in results],
        }

    async def _run_one_sample(self, index: int, item: DatasetRecord) -> SampleRunResult:
        """在并发信号量保护下执行单条样本。"""

        async with self._semaphore:
            return await self._execute_sample(index=index, item=item)

    async def _execute_sample(self, index: int, item: DatasetRecord) -> SampleRunResult:
        """执行单条样本的完整四阶段流程，并实时写入日志。"""

        sample_id = str(item.get("sample_id", f"sample-{index:04d}"))
        started_at = datetime.now(timezone.utc).isoformat()

        baseline_prompt = ""
        teacher_prompt = ""
        intervention_prompt = ""
        baseline_answer = ""
        teacher_output = ""
        intervention_answer = ""
        s_base: int | None = None
        s_knowledge: int | None = None
        base_reason = ""
        knowledge_reason = ""

        try:
            source_material = self._required_text(item, "source_material")
            teaching_guideline = self._required_text(item, "teaching_guideline")
            design_question = self._required_text(item, "design_question")
            reference_answer = item.get("reference_answer")

            baseline_prompt = self.prompt_engine.render_student_baseline_prompt(
                source_material=source_material,
                design_question=design_question,
            )
            baseline_answer = await self.student_client.generate(baseline_prompt)

            baseline_judge = await self.evaluator.evaluate(
                source_material=source_material,
                design_question=design_question,
                student_answer=baseline_answer,
                reference_answer=reference_answer,
            )
            baseline_result = self._normalize_judge_result(baseline_judge)
            s_base = baseline_result.score
            base_reason = baseline_result.reason

            teacher_prompt = self.prompt_engine.render_teacher_prompt(
                source_material=source_material,
                teaching_guideline=teaching_guideline,
            )
            teacher_output = await self.teacher_client.generate(teacher_prompt)

            intervention_prompt = self.prompt_engine.render_student_intervention_prompt(
                source_material=source_material,
                teacher_output=teacher_output,
                design_question=design_question,
            )
            intervention_answer = await self.student_client.generate(intervention_prompt)

            intervention_judge = await self.evaluator.evaluate(
                source_material=source_material,
                design_question=design_question,
                student_answer=intervention_answer,
                reference_answer=reference_answer,
            )
            intervention_result = self._normalize_judge_result(intervention_judge)
            s_knowledge = intervention_result.score
            knowledge_reason = intervention_result.reason

            delta_score = s_knowledge - s_base

            await self.logger.write(
                {
                    "timestamp": started_at,
                    "index": index,
                    "sample_id": sample_id,
                    "status": "success",
                    "input": {
                        "source_material": source_material,
                        "teaching_guideline": teaching_guideline,
                        "design_question": design_question,
                        "reference_answer": reference_answer,
                        "meta": item.get("meta"),
                    },
                    "prompts": {
                        "student_baseline_prompt": baseline_prompt,
                        "teacher_prompt": teacher_prompt,
                        "student_intervention_prompt": intervention_prompt,
                    },
                    "outputs": {
                        "student_baseline_answer": baseline_answer,
                        "teacher_output": teacher_output,
                        "student_intervention_answer": intervention_answer,
                    },
                    "scores": {
                        "s_base": s_base,
                        "s_knowledge": s_knowledge,
                        "delta_score": delta_score,
                        "s_base_reason": base_reason,
                        "s_knowledge_reason": knowledge_reason,
                    },
                }
            )

            return SampleRunResult(
                index=index,
                sample_id=sample_id,
                s_base=s_base,
                s_knowledge=s_knowledge,
                delta_score=delta_score,
                status="success",
                error=None,
            )

        except Exception as exc:
            error_message = str(exc)
            await self.logger.write(
                {
                    "timestamp": started_at,
                    "index": index,
                    "sample_id": sample_id,
                    "status": "failed",
                    "error": error_message,
                    "prompts": {
                        "student_baseline_prompt": baseline_prompt,
                        "teacher_prompt": teacher_prompt,
                        "student_intervention_prompt": intervention_prompt,
                    },
                    "outputs": {
                        "student_baseline_answer": baseline_answer,
                        "teacher_output": teacher_output,
                        "student_intervention_answer": intervention_answer,
                    },
                    "scores": {
                        "s_base": s_base,
                        "s_knowledge": s_knowledge,
                    },
                }
            )

            return SampleRunResult(
                index=index,
                sample_id=sample_id,
                s_base=s_base,
                s_knowledge=s_knowledge,
                delta_score=None,
                status="failed",
                error=error_message,
            )

    @staticmethod
    def _required_text(item: Mapping[str, Any], key: str) -> str:
        """保证关键字段存在且为非空字符串。"""

        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"数据字段缺失或为空: {key}")
        return value.strip()

    @staticmethod
    def _normalize_judge_result(result: JudgeResult | Mapping[str, Any]) -> JudgeResult:
        """兼容 Evaluator 返回 dataclass 或 dict 两种形式。"""

        if isinstance(result, JudgeResult):
            return result

        score_raw = result.get("score")
        reason_raw = result.get("reason")

        try:
            score = int(score_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Judge score 非法: {score_raw}") from exc

        if not isinstance(reason_raw, str):
            reason_raw = str(reason_raw or "")

        raw_copy = dict(result)
        return JudgeResult(score=score, reason=reason_raw.strip(), raw=raw_copy)
