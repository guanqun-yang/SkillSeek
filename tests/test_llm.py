import asyncio
import os
from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel, ValidationError

from conftest import requires_env
from llm import LLMConfig, agenerate, fit_to_context, parse_structured, request_kwargs, split_reasoning


class Answer(BaseModel):
    answer: int


def config(**overrides) -> LLMConfig:
    base = dict(model="openrouter/openai/gpt-oss-120b", max_context_length=131072, max_output_tokens=2000,
                reasoning=True, reasoning_strength="low", temperature=0.0, top_p=1.0, top_k=1)
    return LLMConfig(**{**base, **overrides})


def message(text):
    return NS(type="message", content=[NS(type="output_text", text=text)])


def test_split_reasoning_items():
    response = NS(output=[
        NS(type="reasoning", summary=[NS(text="17*23 = 391")], content=None),
        message('{"answer": 391}'),
    ])
    output, reasoning = split_reasoning(response)
    assert output == '{"answer": 391}'
    assert reasoning == "17*23 = 391"


def test_split_think_tags():
    output, reasoning = split_reasoning(NS(output=[message('<think>multiply first</think>\n\n{"answer": 391}')]))
    assert output == '{"answer": 391}'
    assert reasoning == "multiply first"
    assert "<think>" not in output and "</think>" not in output


def test_split_unopened_think():
    output, reasoning = split_reasoning(NS(output=[message('multiply first</think>{"answer": 391}')]))
    assert output == '{"answer": 391}'
    assert reasoning == "multiply first"


def test_split_plain_answer():
    output, reasoning = split_reasoning(NS(output=[message('\n\n{"answer": 391}\n')]))
    assert (output, reasoning) == ('{"answer": 391}', "")


@pytest.mark.parametrize("text", ['{"answer": 391}', '```json\n{"answer": 391}\n```', '```\n{"answer": 391}\n```'])
def test_parse_structured(text):
    assert parse_structured(text, Answer).answer == 391


@pytest.mark.parametrize("text", ['{"answer": "many"}', "The answer is 391.", '{"result": 391}'])
def test_parse_structured_invalid(text):
    with pytest.raises(ValidationError):
        parse_structured(text, Answer)


def test_request_kwargs():
    kwargs = request_kwargs(config(api_base="http://localhost:8000/v1"))
    assert kwargs["temperature"] == 0.0 and kwargs["top_p"] == 1.0 and kwargs["top_k"] == 1
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["max_output_tokens"] == 2000
    assert kwargs["api_base"] == "http://localhost:8000/v1"
    assert request_kwargs(config(reasoning=False))["reasoning"] == {"effort": "none"}


def test_request_kwargs_reasoning():
    assert request_kwargs(config(model="openai/gpt-5-mini"))["reasoning"] == {"effort": "low", "summary": "auto"}
    assert request_kwargs(config(model="gemini/gemini-3.8-flash"))["reasoning"] == {"effort": "low"}
    minimax = request_kwargs(config(model="minimax/MiniMax-M2"))
    assert "reasoning" not in minimax
    assert (minimax["temperature"], minimax["top_p"], minimax["top_k"]) == (0.0, 1.0, 1)


def test_fit_to_context():
    cfg = config(max_context_length=600, max_output_tokens=100)
    body = fit_to_context(cfg, "Summarize:\n{body}\nJSON:", "word " * 5000)
    assert 0 < len(body) < len("word " * 5000)
    assert fit_to_context(config(), "Summarize:\n{body}", "short body") == "short body"
    with pytest.raises(ValueError):
        fit_to_context(config(max_context_length=120, max_output_tokens=100), "x " * 200 + "{body}", "body")


HARD, EASY = (4817, 2963), (17, 23)
LIVE = [
    pytest.param("openrouter/openai/gpt-oss-120b", {}, HARD, True, marks=requires_env("OPENROUTER_API_KEY"), id="openrouter-gpt-oss"),
    pytest.param("openrouter/anthropic/claude-haiku-4.5", {"reasoning": False}, EASY, False, marks=requires_env("OPENROUTER_API_KEY"), id="openrouter-anthropic"),
    pytest.param("openrouter/moonshotai/kimi-k2", {"reasoning": False}, EASY, False, marks=requires_env("OPENROUTER_API_KEY"), id="openrouter-kimi"),
    pytest.param("openrouter/z-ai/glm-4.6", {}, HARD, True, marks=requires_env("OPENROUTER_API_KEY"), id="openrouter-zai"),
    pytest.param("openai/gpt-5-mini", {"temperature": 1.0}, HARD, True, marks=requires_env("OPENAI_API_KEY"), id="openai"),
    pytest.param("gemini/gemini-3.8-flash", {}, HARD, True, marks=requires_env("GEMINI_API_KEY"), id="gemini"),
    pytest.param("minimax/MiniMax-M2", {"temperature": 1.0, "top_p": 0.95, "top_k": 40}, HARD, True, marks=requires_env("MINIMAX_API_KEY"), id="minimax"),
    pytest.param("anthropic/claude-haiku-4-5", {"reasoning": False}, EASY, False, marks=requires_env("ANTHROPIC_API_KEY"), id="anthropic"),
    pytest.param("moonshot/kimi-k2-0905-preview", {"reasoning": False}, EASY, False, marks=requires_env("MOONSHOT_API_KEY"), id="kimi"),
    pytest.param("zai/glm-4.6", {}, HARD, True, marks=requires_env("ZAI_API_KEY"), id="zai"),
    pytest.param("hosted_vllm", {}, EASY, True, marks=requires_env("VLLM_API_BASE"), id="vllm"),
]


@pytest.mark.parametrize("model,overrides,operands,expects_reasoning", LIVE)
def test_live_provider(model, overrides, operands, expects_reasoning):
    if model == "hosted_vllm":
        model = f"hosted_vllm/{os.environ.get('VLLM_MODEL', 'openai/gpt-oss-120b')}"
        overrides = {**overrides, "api_base": os.environ["VLLM_API_BASE"]}
    cfg = config(model=model, **overrides)
    a, b = operands
    prompt = f'What is {a}*{b}? Reply with the JSON object {{"answer": <integer>}} and nothing else.'
    result = asyncio.run(agenerate(cfg, prompt))
    answer = parse_structured(result.output, Answer).answer
    if operands == EASY:
        assert answer == a * b
    assert "<think>" not in result.output and "</think>" not in result.output
    assert result.output_tokens and result.output_tokens > 0
    if expects_reasoning or result.reasoning_tokens:
        assert result.reasoning.strip()
        assert result.reasoning.strip() not in result.output


def test_split_no_message():
    with pytest.raises(ValueError, match="no message item"):
        split_reasoning(NS(output=[NS(type="reasoning", summary=[], content=[NS(text="use python")]), NS(type="mcp_call")]))
