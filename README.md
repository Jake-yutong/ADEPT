# ADEPT
The AI Design Education Performance Test (ADEPT) is an automated benchmarking framework engineered to evaluate the pedagogical effectiveness of Large Language Models (LLMs) in design education.

## API Integration

ADEPT now includes three role-oriented APIs:

- `TeacherAPI`: generate teaching guidance from source material and guideline.
- `StudentAPI`: generate baseline and intervention answers.
- `RubricScoringAPI`: score student answers with rubric and return structured JSON (`score`, `reason`).

You can build an orchestrator with all APIs wired in one step:

```python
from adept import ConfigLoader, build_orchestrator_with_apis

config = ConfigLoader.load(config_path="config.yaml", env_path=".env")
orchestrator, api_bundle = build_orchestrator_with_apis(config)
```

## Run Tests With Visible Progress

Pytest configuration is set to verbose mode by default (`-vv -s -rA`), so you can watch each test execution in real time.

```bash
pytest
```

If you want to stop early on first failure:

```bash
pytest --maxfail=1
```

## Streamlit Web UI

This repository now includes a Streamlit single-page UI for ADEPT at `streamlit_app.py`.

Install Streamlit:

```bash
pip install streamlit
```

Run locally:

```bash
streamlit run streamlit_app.py
```

After startup, open:

- Local: `http://localhost:8501`

In VS Code / Codespaces, if port forwarding is enabled for 8501, you can open the forwarded URL directly from the Ports panel.

The current UI version uses mock functions (`time.sleep` + fake outputs) so you can validate UX and interaction flow before wiring real model APIs.

### Real API Mode

The Streamlit page supports two modes in sidebar:

- `Real API（真实）`: call ADEPT `TeacherAPI`, `StudentAPI`, and `RubricScoringAPI`.
- `Mock（演示）`: run with fake outputs for UI testing.

For `Real API（真实）`, provide:

1. Teacher / Student / Judge model IDs (manual input)
2. Teacher / Student / Judge API keys (independent)
3. Optional Teacher / Student / Judge Base URLs (independent, useful for cross-platform APIs)

## GitHub Hosting Option

You can host this UI directly from GitHub via Streamlit Community Cloud:

1. Push repository to GitHub.
2. Create an app on Streamlit Community Cloud.
3. Select this repository and set the entrypoint file to `streamlit_app.py`.
