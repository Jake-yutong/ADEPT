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
