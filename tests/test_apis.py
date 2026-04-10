from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from adept.apis import RubricScoringAPI, StudentAPI, TeacherAPI, build_api_bundle
from adept.models import AppConfig, LLMClient, ModelConfig, RuntimeConfig
from adept.orchestrator import ADEPTOrchestrator, JsonlLogger


def _dummy_model_config(model_name: str = "dummy-model") -> ModelConfig:
    return ModelConfig(
        provider="custom",
        model=model_name,
        api_key="test-key",
        base_url="https://example.test/v1",
    )


class QueueLLMClient(LLMClient):
    """按顺序返回预置响应的测试 client。"""

    def __init__(self, responses: list[str], *, model_name: str = "dummy-model") -> None:
        super().__init__(config=_dummy_model_config(model_name=model_name))
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        if not self._responses:
            raise AssertionError("QueueLLMClient 没有剩余响应")
        return self._responses.pop(0)


def test_teacher_api_generates_prompt_and_answer() -> None:
    async def _run() -> None:
        client = QueueLLMClient(["这是教师辅导内容"])
        api = TeacherAPI(client)

        result = await api.teach(
            source_material="包豪斯课程简介",
            teaching_guideline="强调结构与功能统一",
        )

        assert "教学大纲" in result.prompt
        assert "包豪斯课程简介" in result.prompt
        assert result.answer == "这是教师辅导内容"

    asyncio.run(_run())


def test_student_api_supports_baseline_and_intervention() -> None:
    async def _run() -> None:
        client = QueueLLMClient(["baseline 答案", "intervention 答案"])
        api = StudentAPI(client)

        baseline = await api.answer_baseline(
            source_material="色彩构成基础",
            design_question="如何构建品牌主色体系？",
        )
        intervention = await api.answer_intervention(
            source_material="色彩构成基础",
            teacher_output="先定义语义色，再确定层级映射",
            design_question="如何构建品牌主色体系？",
        )

        assert "仅根据提供的设计素材" in baseline.prompt
        assert "必须回答" in baseline.prompt
        assert baseline.answer == "baseline 答案"
        assert "导师" in intervention.prompt
        assert "必须回答" in intervention.prompt
        assert intervention.answer == "intervention 答案"

    asyncio.run(_run())


def test_rubric_scoring_api_parses_plain_json() -> None:
    async def _run() -> None:
        client = QueueLLMClient(['{"score": 82, "reason": "结构完整，论证较清晰"}'])
        api = RubricScoringAPI(client)

        result = await api.evaluate(
            source_material="移动端导航设计案例",
            design_question="请给出信息架构优化建议",
            student_answer="我会按任务频次组织导航分组",
            reference_answer="先做任务流分析，再迭代 IA",
        )

        assert result["score"] == 82
        assert "结构完整" in result["reason"]

    asyncio.run(_run())


def test_rubric_scoring_api_parses_fenced_json() -> None:
    async def _run() -> None:
        client = QueueLLMClient(
            [
                "评语如下：\n```json\n"
                "{\"score\": 90, \"reason\": \"策略明确且表达清晰\"}\n"
                "```"
            ]
        )
        api = RubricScoringAPI(client)

        result = await api.evaluate(
            source_material="服务蓝图案例",
            design_question="如何优化触点一致性？",
            student_answer="按前台触点与后台支撑进行联动设计",
        )

        assert result["score"] == 90
        assert result["reason"] == "策略明确且表达清晰"

    asyncio.run(_run())


def test_rubric_scoring_api_parses_python_literal_dict() -> None:
    async def _run() -> None:
        client = QueueLLMClient(["{'score': 81, 'reason': '结构可执行，步骤清楚'}"])
        api = RubricScoringAPI(client)

        result = await api.evaluate(
            source_material="校园导视系统改造",
            design_question="如何降低用户迷路率？",
            student_answer="按高频路径重构导视层级并补充转角提示",
        )

        assert result["score"] == 81
        assert "结构可执行" in result["reason"]

    asyncio.run(_run())


def test_rubric_scoring_api_parses_non_json_text_fallback() -> None:
    async def _run() -> None:
        client = QueueLLMClient([
            "评分结果如下：\nscore: 77\nreason: 方案与约束对应较好，但评估指标还可补充。"
        ])
        api = RubricScoringAPI(client)

        result = await api.evaluate(
            source_material="图书馆空间更新",
            design_question="如何平衡安静阅读与小组协作？",
            student_answer="通过分区和动线控制减少相互干扰",
        )

        assert result["score"] == 77
        assert "约束对应" in result["reason"]

    asyncio.run(_run())


def test_rubric_scoring_api_retries_when_first_response_invalid() -> None:
    async def _run() -> None:
        client = QueueLLMClient([
            "我认为答案整体不错。",
            '{"score": 74, "reason": "第二次输出合法 JSON"}',
        ])
        api = RubricScoringAPI(client)

        result = await api.evaluate(
            source_material="公共服务触点优化",
            design_question="如何提升触点一致性？",
            student_answer="统一命名体系并梳理关键任务流",
        )

        assert result["score"] == 74
        assert "第二次" in result["reason"]
        assert len(client.calls) == 2

    asyncio.run(_run())


def test_rubric_scoring_api_clamps_out_of_range_score() -> None:
    async def _run() -> None:
        # 越界分数应被 clamp 到 [min_score, max_score]，而不是抛出异常
        client = QueueLLMClient(['{"score": 999, "reason": "超出范围的分数"}'])
        api = RubricScoringAPI(client, min_score=0, max_score=100)

        result = await api.evaluate(
            source_material="字体排印案例",
            design_question="如何提升层级可读性？",
            student_answer="通过字号与字重建立视觉层级",
        )
        assert result["score"] == 100, f"期望 clamp 到 100，实际得到 {result['score']}"

    asyncio.run(_run())


def test_rubric_scoring_api_clamps_negative_score() -> None:
    async def _run() -> None:
        # 负数分数应 clamp 到 min_score=0，不抛异常
        client = QueueLLMClient(['{"score": -60, "reason": "判负分"}'])
        api = RubricScoringAPI(client, min_score=0, max_score=100)

        result = await api.evaluate(
            source_material="交互原型案例",
            design_question="如何降低操作出错率？",
            student_answer="增加确认步骤和撤销功能",
        )
        assert result["score"] == 0

    asyncio.run(_run())


def test_rubric_scoring_api_parses_fraction_score() -> None:
    async def _run() -> None:
        # judge 返回 "70/100" 形式的分数，应正确解析为 70
        client = QueueLLMClient(['{"score": "70/100", "reason": "部分要点覆盖"}'])
        api = RubricScoringAPI(client, min_score=0, max_score=100)

        result = await api.evaluate(
            source_material="空间导视案例",
            design_question="如何重构导视层级？",
            student_answer="按使用频次重新排列标识层级",
        )
        assert result["score"] == 70

    asyncio.run(_run())


def test_rubric_scoring_api_parses_percentage_score() -> None:
    async def _run() -> None:
        # judge 返回 "70%" 形式的分数，应正确解析为 70
        client = QueueLLMClient(['{"score": "70%", "reason": "达到及格水平"}'])
        api = RubricScoringAPI(client, min_score=0, max_score=100)

        result = await api.evaluate(
            source_material="品牌色彩系统案例",
            design_question="如何定义品牌主色？",
            student_answer="先定义语义色，再建立层级映射",
        )
        assert result["score"] == 70

    asyncio.run(_run())


def test_rubric_scoring_api_parses_score_with_suffix() -> None:
    async def _run() -> None:
        # judge 返回 "75分" / "75 points" 形式，应取前缀数字 75
        client = QueueLLMClient(['{"score": "75分", "reason": "方案较完整"}'])
        api = RubricScoringAPI(client, min_score=0, max_score=100)

        result = await api.evaluate(
            source_material="服务设计案例",
            design_question="如何简化服务流程？",
            student_answer="删除低价值步骤，合并冗余触点",
        )
        assert result["score"] == 75

    asyncio.run(_run())


def test_rubric_scoring_api_custom_score_range() -> None:
    async def _run() -> None:
        # 使用非标准满分（如 150 分制）的 rubric 场景
        client = QueueLLMClient(['{"score": 120, "reason": "优秀答案"}'])
        api = RubricScoringAPI(client, min_score=0, max_score=150)

        result = await api.evaluate(
            source_material="高分制评测案例",
            design_question="综合设计策略题",
            student_answer="完整回答了所有维度",
        )
        assert result["score"] == 120

    asyncio.run(_run())


def test_rubric_scoring_api_custom_range_clamps_overflow() -> None:
    async def _run() -> None:
        # 自定义分值范围下，越界仍应被 clamp，不抛异常
        client = QueueLLMClient(['{"score": 200, "reason": "超出 150 的分数"}'])
        api = RubricScoringAPI(client, min_score=0, max_score=150)

        result = await api.evaluate(
            source_material="高分制评测案例",
            design_question="综合设计策略题",
            student_answer="答案内容",
        )
        assert result["score"] == 150

    asyncio.run(_run())


def test_rubric_scoring_api_null_score_degrades_gracefully() -> None:
    async def _run() -> None:
        # judge 返回 score=null，三次重试全部失败后应降级为 0 分而非崩溃
        client = QueueLLMClient([
            '{"score": null, "reason": "无法评分"}',
            '{"score": null, "reason": "再次无法评分"}',
            '{"score": null, "reason": "第三次无法评分"}',
        ])
        api = RubricScoringAPI(client, min_score=0, max_score=100)

        result = await api.evaluate(
            source_material="异常场景测试",
            design_question="测试题目",
            student_answer="测试答案",
        )
        # score 无法解析时应降级为 0，不抛异常
        assert result["score"] == 0
        assert "无法解析" in result["reason"] or result["score"] == 0

    asyncio.run(_run())


def test_rubric_scoring_api_fallback_float_score_in_freetext() -> None:
    async def _run() -> None:
        # judge 以非 JSON 格式返回浮点分数，启发式解析应正确提取
        client = QueueLLMClient([
            "综合评估：\nscore: 77.5\n理由: 答案覆盖了主要知识点但缺乏深度。"
        ])
        api = RubricScoringAPI(client)

        result = await api.evaluate(
            source_material="公共设施可达性研究",
            design_question="如何改善社区公共空间的无障碍通道？",
            student_answer="增加坡道和触觉导盲铺装",
        )
        assert result["score"] == 77

    asyncio.run(_run())


def test_orchestrator_runs_with_teacher_student_and_rubric_apis(tmp_path: Path) -> None:
    async def _run() -> None:
        teacher_client = QueueLLMClient(["导师建议：先确认约束，再展开方案"])
        student_client = QueueLLMClient(["baseline 方案", "intervention 方案"])
        judge_client = QueueLLMClient(
            [
                '{"score": 58, "reason": "baseline 细节不足"}',
                '{"score": 86, "reason": "intervention 可执行性提升明显"}',
            ]
        )

        teacher_api = TeacherAPI(teacher_client)
        student_api = StudentAPI(student_client)
        rubric_api = RubricScoringAPI(judge_client)

        output_path = tmp_path / "adept_results.jsonl"
        logger = JsonlLogger(output_path)

        orchestrator = ADEPTOrchestrator(
            teacher_client=teacher_client,
            student_client=student_client,
            evaluator=rubric_api,
            logger=logger,
            max_concurrency=1,
            teacher_api=teacher_api,
            student_api=student_api,
        )

        dataset = [
            {
                "sample_id": "sample-001",
                "source_material": "社区导视系统现状分析",
                "teaching_guideline": "强调用户路径与可达性",
                "design_question": "如何重构导视层级并降低迷路率？",
            }
        ]

        result = await orchestrator.run(dataset)

        assert result["summary"]["total_samples"] == 1
        assert result["summary"]["successful_samples"] == 1
        assert result["summary"]["avg_delta_score"] == 28
        assert teacher_client.calls and student_client.calls
        assert len(judge_client.calls) == 2

        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        payload = json.loads(lines[0])
        assert payload["status"] == "success"
        assert payload["scores"]["delta_score"] == 28
        assert "教学大纲" in payload["prompts"]["teacher_prompt"]

    asyncio.run(_run())


def test_build_api_bundle_from_config() -> None:
    config = AppConfig(
        teacher=_dummy_model_config("teacher-model"),
        student=_dummy_model_config("student-model"),
        judge=_dummy_model_config("judge-model"),
        runtime=RuntimeConfig(concurrency=2, output_path="outputs/test.jsonl"),
    )

    bundle = build_api_bundle(config)

    assert bundle.teacher_api.client is bundle.teacher_client
    assert bundle.student_api.client is bundle.student_client
    assert bundle.rubric_api.client is bundle.judge_client

    asyncio.run(bundle.close())
