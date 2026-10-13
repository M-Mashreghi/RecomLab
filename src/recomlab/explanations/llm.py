"""Structured LLM explanation generator with safe fallbacks.

The model receives only the condition-scoped evidence payload and must answer
with JSON matching :data:`OUTPUT_SCHEMA`: explanation text, cited evidence ids,
an abstain flag and an uncertainty level. Responses are validated (schema and
"every cited id exists in the payload"); only transient provider errors and
schema failures are retried, a bounded number of times. Any remaining failure
falls back to the deterministic template (or no explanation) -- never to an
LLM-generated ranking, and never blocking recommendation delivery.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from recomlab.config.settings import SecretStr
from recomlab.explanations.conditions import (
    Explanation,
    ExplanationCondition,
    TemplateExplainer,
)
from recomlab.explanations.evidence import EvidencePayload

logger = logging.getLogger("recomlab.explanations.llm")

PROMPT_VERSION = "expl-prompt-v1"
UNCERTAINTY = ("low", "medium", "high")
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
        "uncertainty": {"type": "string", "enum": list(UNCERTAINTY)},
    },
    "required": ["explanation", "cited_evidence_ids", "abstain", "uncertainty"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = (
    "You write one short explanation (at most two sentences) of why a recommender system "
    "suggested an item. Use ONLY the evidence entries provided. Do not state user "
    "preferences, item facts, or reasons that are not in the evidence. Cite the evidence_id "
    "of every entry you rely on. If the evidence is insufficient, set abstain to true and "
    "leave explanation empty. Never change, re-rank or suggest other items."
)


class ProviderError(RuntimeError):
    def __init__(self, message: str, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


class SchemaError(ValueError):
    pass


class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str: ...


class MockProvider:
    """Deterministic provider for tests: returns queued strings or raises queued errors."""

    name = "mock"

    def __init__(
        self, responses: Sequence[str | Exception] | Callable[[str], str], model: str = "mock-1"
    ) -> None:
        self._responses = responses if callable(responses) else list(responses)
        self.model = model
        self.calls: list[str] = []

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        self.calls.append(prompt)
        if callable(self._responses):
            return self._responses(prompt)
        if not self._responses:
            raise ProviderError("mock exhausted", transient=False)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class AnthropicProvider:  # pragma: no cover - live API, never called in CI
    """Optional adapter for the Claude Messages API (``pip install anthropic``).

    Uses structured outputs (``output_config.format``) and low effort for
    low-variance, cheap explanations. Sampling parameters such as temperature
    are not sent: current Claude models reject them.
    """

    name = "anthropic"

    def __init__(self, model: str, api_key: SecretStr, timeout_seconds: float = 20.0) -> None:
        try:
            import anthropic
        except ImportError as err:
            raise ProviderError("anthropic SDK not installed", transient=False) from err
        self._anthropic = anthropic
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key.get_secret_value() or None, timeout=timeout_seconds, max_retries=0
        )

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        a = self._anthropic
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except (a.RateLimitError, a.APIConnectionError, a.APITimeoutError) as err:
            raise ProviderError(str(err), transient=True) from err
        except a.APIStatusError as err:
            raise ProviderError(str(err), transient=err.status_code >= 500) from err
        if response.stop_reason == "refusal":
            raise ProviderError("model declined the request", transient=False)
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text:
            raise ProviderError(f"empty response (stop_reason={response.stop_reason})", False)
        return text


@dataclass(frozen=True)
class ParsedOutput:
    text: str
    cited: tuple[str, ...]
    abstain: bool
    uncertainty: str


def parse_output(raw: str, payload: EvidencePayload) -> ParsedOutput:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise SchemaError(f"not JSON: {err}") from err
    if not isinstance(data, dict) or set(data) != set(OUTPUT_SCHEMA["required"]):
        raise SchemaError("wrong keys")
    text, cited, abstain, unc = (
        data["explanation"],
        data["cited_evidence_ids"],
        data["abstain"],
        data["uncertainty"],
    )
    if not isinstance(text, str) or not isinstance(abstain, bool) or unc not in UNCERTAINTY:
        raise SchemaError("wrong field types")
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        raise SchemaError("cited_evidence_ids must be a list of strings")
    unknown = sorted(set(cited) - payload.ids)
    if unknown:
        raise SchemaError(f"cites evidence not in payload: {unknown}")
    if not abstain and (not text.strip() or not cited):
        raise SchemaError("non-abstaining explanation must have text and citations")
    return ParsedOutput(text.strip(), tuple(dict.fromkeys(cited)), abstain, unc)


def build_prompt(payload: EvidencePayload, condition: ExplanationCondition) -> str:
    evidence = [
        {"evidence_id": e.evidence_id, "kind": e.kind, "data": dict(e.data)}
        for e in payload.evidence
    ]
    body = {
        "condition": condition.condition_id,
        "recommended_item_id": payload.item_id,
        "evidence": evidence,
        "unavailable": [dict(u) for u in payload.unavailable],
    }
    return "Explain this recommendation.\n" + json.dumps(body, sort_keys=True, default=str)


class StructuredLLMExplainer:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_attempts: int = 3,
        fallback: str = "template",  # template | none
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        if fallback not in ("template", "none"):
            raise ValueError("fallback must be 'template' or 'none'")
        self.provider = provider
        self.max_attempts = max(1, max_attempts)
        self.fallback = fallback
        self.prompt_version = prompt_version
        self.template = TemplateExplainer()

    def _fallback(self, payload: EvidencePayload, condition_id: str, reason: str) -> Explanation:
        logger.warning(
            "llm explanation fallback",
            extra={"condition": condition_id, "reason": reason, "prompt": self.prompt_version},
        )
        if self.fallback == "none":
            return Explanation(
                condition_id,
                None,
                (),
                "none",
                abstained=True,
                fallback_from=condition_id,
                prompt_version=self.prompt_version,
                model=self.provider.model,
            )
        out = self.template.explain(payload)
        return Explanation(
            out.condition_id,
            out.text,
            out.cited_evidence,
            out.generator,
            out.abstained,
            fallback_from=condition_id,
            prompt_version=self.prompt_version,
            model=self.provider.model,
        )

    def explain(self, payload: EvidencePayload, condition: ExplanationCondition) -> Explanation:
        prompt = build_prompt(payload, condition)
        last_error = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.provider.generate(SYSTEM_PROMPT, prompt, OUTPUT_SCHEMA)
                parsed = parse_output(raw, payload)
            except ProviderError as err:
                last_error = f"provider: {err}"
                if not err.transient:
                    break
                continue
            except SchemaError as err:
                last_error = f"schema: {err}"
                continue
            except Exception as err:  # noqa: BLE001 - explanation must never break serving
                last_error = f"unexpected: {type(err).__name__}: {err}"
                break
            logger.info(
                "llm explanation generated",
                extra={
                    "condition": condition.condition_id,
                    "prompt_version": self.prompt_version,
                    "model": self.provider.model,
                    "attempt": attempt,
                },
            )
            return Explanation(
                condition.condition_id,
                None if parsed.abstain else parsed.text,
                parsed.cited,
                f"llm:{self.provider.name}",
                abstained=parsed.abstain,
                prompt_version=self.prompt_version,
                model=self.provider.model,
            )
        return self._fallback(payload, condition.condition_id, last_error)
