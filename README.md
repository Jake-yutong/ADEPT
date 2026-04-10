# ADEPT

**AI Design Education Performance Test** — 用于评估 LLM 在设计教育场景中教学效果的自动化 Benchmark 框架。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      streamlit_app.py                           │
│              Streamlit Web UI（控制台 / 结果可视化）               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ build_api_bundle()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        adept/apis.py                            │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐  │
│  │ TeacherAPI  │   │ StudentAPI  │   │  RubricScoringAPI    │  │
│  │             │   │             │   │                      │  │
│  │ teach()     │   │ answer_     │   │ evaluate()           │  │
│  │             │   │ baseline()  │   │ _parse_score()       │  │
│  │             │   │ answer_     │   │ _extract_json_       │  │
│  │             │   │ intervention│   │  object()            │  │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬───────────┘  │
│         └────────────┬────┘                     │              │
│                      │ APIBundle                │              │
└──────────────────────┼──────────────────────────┼──────────────┘
                       │                          │
                       ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     adept/orchestrator.py                       │
│                                                                 │
│  ADEPTOrchestrator.run(dataset)                                 │
│    ├─ 1. Student baseline answer                                │
│    ├─ 2. Judge scores baseline  →  s_base                       │
│    ├─ 3. Teacher generates guidance                             │
│    ├─ 4. Student intervention answer (with teacher output)      │
│    └─ 5. Judge scores intervention  →  s_knowledge             │
│                                                                 │
│  Δ = s_knowledge − s_base                                       │
│  JsonlLogger  →  outputs/adept_results.jsonl                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ LLMClient.generate()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       adept/models.py                           │
│                                                                 │
│  LLMClientFactory                                               │
│    ├─ DeepSeekClient   (provider: deepseek)                     │
│    ├─ QwenClient       (provider: qwen)                         │
│    ├─ KimiClient       (provider: kimi)                         │
│    ├─ OpenAIClient     (provider: openai)                       │
│    └─ OpenAICompatibleClient  (provider: custom)                │
│                                                                 │
│  ConfigLoader  ←  config.yaml  /  .env  /  环境变量             │
│  AppConfig = { teacher, student, judge, runtime }               │
└─────────────────────────────────────────────────────────────────┘
```

### 核心评测流程（Delta Score）

```
输入题目集 (.txt / .json / .jsonl)
        │
        ▼
[StudentAPI] 无辅导作答   →  [Judge]  →  s_base
        │
[TeacherAPI] 生成教学辅导
        │
[StudentAPI] 带辅导作答   →  [Judge]  →  s_knowledge
        │
        ▼
  Δ = s_knowledge − s_base
（衡量大模型辅导对学习效果的提升）
```

### 模块说明

| 模块 | 文件 | 职责 |
|---|---|---|
| **TeacherAPI** | `adept/apis.py` | 基于素材与教学大纲生成辅导内容，不直接给出答案 |
| **StudentAPI** | `adept/apis.py` | 生成 baseline（无辅导）与 intervention（有辅导）两种作答；内置多问题覆盖二次检查 |
| **RubricScoringAPI** | `adept/apis.py` | 基于 Rubric 对学生答案打分，返回 `{score, reason}`；支持多种分数格式（整数 / 浮点 / 分数式 / 百分比 / 带单位），越界自动 clamp |
| **ADEPTOrchestrator** | `adept/orchestrator.py` | 并发调度五阶段评测流程，写入 JSONL 日志 |
| **JsonlLogger** | `adept/orchestrator.py` | 异步并发安全日志写入 |
| **DefaultPromptTemplateEngine** | `adept/orchestrator.py` | 内置 Prompt 模板（可替换为 Jinja2 引擎） |
| **LLMClientFactory** | `adept/models.py` | 按 provider 实例化对应 LLM 客户端 |
| **ConfigLoader** | `adept/models.py` | 按优先级加载：`config.yaml` → `.env` → 环境变量 |

### 支持的 LLM Provider

| Provider | 默认 Base URL |
|---|---|
| `deepseek` | `https://api.deepseek.com` |
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `kimi` | `https://api.moonshot.cn/v1` |
| `openai` | `https://api.openai.com/v1` |
| `custom` | 自定义（OpenAI 兼容协议） |

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

通过 `config.yaml` 或 `.env` 文件配置三个角色（teacher / student / judge）的模型：

```yaml
# config.yaml 示例
models:
  teacher:
    provider: deepseek
    model: deepseek-chat
    api_key_env: ADEPT_TEACHER_API_KEY
  student:
    provider: qwen
    model: qwen-plus
    api_key_env: ADEPT_STUDENT_API_KEY
  judge:
    provider: deepseek
    model: deepseek-reasoner
    api_key_env: ADEPT_JUDGE_API_KEY

runtime:
  concurrency: 3
  output_path: outputs/adept_results.jsonl
```

或通过环境变量：

```bash
export ADEPT_TEACHER_API_KEY=sk-xxx
export ADEPT_STUDENT_API_KEY=sk-yyy
export ADEPT_JUDGE_API_KEY=sk-zzz
```

### Python API

```python
from adept import ConfigLoader, build_orchestrator_with_apis

config = ConfigLoader.load(config_path="config.yaml", env_path=".env")
orchestrator, api_bundle = build_orchestrator_with_apis(config)
```

---

## Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

启动后访问 `http://localhost:8501`。

UI 支持两种运行模式（侧边栏切换）：

- **Real API（真实）**：调用真实 LLM，在侧边栏配置 Teacher / Student / Judge 的 Model ID、API Key 和 Base URL（支持三角色独立配置）
- **Mock（演示）**：使用预置假数据，无需 API Key，用于验证界面交互

支持上传自定义题目文件（`.txt` / `.json` / `.jsonl`）和 Rubric 评分文件，替换内置 Affordance 题目集。

---

## 运行测试

```bash
pytest
```

测试配置默认输出详细进度（`-vv -s -rA`）。失败即止：

```bash
pytest --maxfail=1
```

---

## 输入文件格式

### 题目文件（推荐 `.txt` ADEPT 标准格式）

```
--------------------------------------------------------------------------------
QUESTION 1 [DIFFICULTY: LEVEL 1]
对应原文：案例描述

【选择题】
题目内容...
A. 选项A
B. 选项B

【简答题】（100字以内）
题目内容...
```

也支持 JSON/JSONL，字段：`source_material`、`design_question`、`teaching_guideline`。

### Rubric 文件

纯文本描述评分标准，支持任意评分维度和满分值。`RubricScoringAPI` 会自动按题号提取对应片段。

---

## 输出

评测结果写入 `outputs/adept_results.jsonl`，每行一个样本：

```json
{
  "index": 0,
  "sample_id": "sample-0000",
  "status": "success",
  "scores": { "s_base": 60, "s_knowledge": 85, "delta_score": 25 },
  "prompts": { "teacher_prompt": "...", "baseline_prompt": "...", "intervention_prompt": "..." },
  "answers": { "baseline": "...", "teacher_output": "...", "intervention": "..." }
}
```

---

## GitHub 部署

可通过 [Streamlit Community Cloud](https://streamlit.io/cloud) 一键托管：

1. 推送仓库到 GitHub
2. 在 Streamlit Community Cloud 新建 App
3. 入口文件选择 `streamlit_app.py`

