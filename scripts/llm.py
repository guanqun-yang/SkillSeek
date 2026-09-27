"""Provider-agnostic LLM calls through LiteLLM's Responses API."""

import re
from typing import Literal, TypeVar

import litellm
from pydantic import BaseModel, Field

litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.DOTALL)


class LLMConfig(BaseModel):
    model: str
    api_base: str | None = None
    max_context_length: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    reasoning: bool
    reasoning_strength: Literal["minimal", "low", "medium", "high"]
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=1)


class LLMResult(BaseModel):
    output: str
    reasoning: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    request: dict


def count_tokens(config: LLMConfig, text: str) -> int:
    return litellm.token_counter(model=config.model, text=text)


def fit_to_context(config: LLMConfig, template: str, body: str) -> str:
    budget = config.max_context_length - config.max_output_tokens
    while body and count_tokens(config, template.format(body=body)) > budget:
        body = body[: int(len(body) * 0.8)]
    if count_tokens(config, template.format(body=body)) > budget:
        raise ValueError(f"prompt does not fit in max_context_length={config.max_context_length}")
    return body


def request_kwargs(config: LLMConfig) -> dict:
    kwargs = {
        "model": config.model,
        "max_output_tokens": config.max_output_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }
    model, provider, _, _ = litellm.get_llm_provider(config.model)
    if "reasoning_effort" in (litellm.get_supported_openai_params(model=model, custom_llm_provider=provider) or []):
        kwargs["reasoning"] = {"effort": config.reasoning_strength if config.reasoning else "none"}
        if provider == "openai" and config.reasoning:
            kwargs["reasoning"]["summary"] = "auto"
    if config.api_base:
        kwargs["api_base"] = config.api_base
    return kwargs


def split_reasoning(response) -> tuple[str, str]:
    kinds = [getattr(item, "type", None) for item in response.output]
    if "message" not in kinds:
        raise ValueError(f"response has no message item: {kinds}")
    output_parts, reasoning_parts = [], []
    for item in response.output:
        kind = getattr(item, "type", None)
        if kind == "reasoning":
            for part in list(getattr(item, "summary", None) or []) + list(getattr(item, "content", None) or []):
                if getattr(part, "text", None):
                    reasoning_parts.append(part.text)
        elif kind == "message":
            for part in item.content:
                if getattr(part, "text", None):
                    output_parts.append(part.text)
    output = "".join(output_parts)
    reasoning_parts.extend(m.strip() for m in _THINK.findall(output))
    output = _THINK.sub("", output)
    if "</think>" in output:
        head, output = output.rsplit("</think>", 1)
        reasoning_parts.append(head.replace("<think>", "").strip())
    return output.strip(), "\n".join(p for p in reasoning_parts if p)


async def agenerate(config: LLMConfig, prompt: str) -> LLMResult:
    kwargs = request_kwargs(config)
    response = await litellm.aresponses(input=prompt, **kwargs)
    output, reasoning = split_reasoning(response)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "output_tokens_details", None)
    return LLMResult(
        output=output,
        reasoning=reasoning,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        request=kwargs,
    )


def parse_structured(text: str, schema: type[T]) -> T:
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    return schema.model_validate_json(text)
