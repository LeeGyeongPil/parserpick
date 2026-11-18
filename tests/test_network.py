"""Real Hugging Face downloads. Deselected by default: run with `uv run pytest -m network`."""

import pytest

from parserpick.recommend import recommend
from parserpick.source import load_source

pytestmark = pytest.mark.network

CASES = [
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_xml"),
    ("zai-org/GLM-4.5-Air", "glm45"),
    ("Qwen/Qwen3-0.6B", "hermes"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31"),
]


@pytest.mark.parametrize(("repo", "family"), CASES)
def test_real_templates(repo, family):
    assert recommend(load_source(f"hf:{repo}")).family == family
