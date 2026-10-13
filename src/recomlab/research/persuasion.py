"""Explanation wording / persuasiveness stress test (research mode only).

Wording variants change only the *framing* of an existing explanation; they
keep its cited evidence and never touch ranking. They can be applied only
inside an explicit :class:`ResearchMode`, so production/demo explanations can
never be silently modified. The stress test reports **persuasion** (effect of
wording on acceptance) and **correctness** (fidelity of the worded text)
separately.

Free-text responses are sanitized (emails, URLs, phone numbers, long digit
runs removed; length bounded) before any optional LLM-assisted coding; the
default coder is a deterministic keyword coder with no LLM.
"""

from __future__ import annotations

import csv
import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recomlab.domain.schemas import Item
from recomlab.explanations.conditions import Explanation
from recomlab.explanations.evidence import EvidencePayload
from recomlab.explanations.fidelity import validate_explanation
from recomlab.research.analysis import compare_groups, holm_adjust
from recomlab.research.study import StudyConfig, StudyGovernanceError


class ResearchModeError(RuntimeError):
    pass


@dataclass(frozen=True)
class WordingVariant:
    variant_id: str
    style: str
    prefix: str = ""
    suffix: str = ""

    def render(self, text: str) -> str:
        return f"{self.prefix}{text}{self.suffix}".strip()


VARIANTS: Mapping[str, WordingVariant] = {
    v.variant_id: v
    for v in (
        WordingVariant("neutral-v1", "neutral"),
        WordingVariant("enthusiastic-v1", "enthusiastic", "You will love this! ", ""),
        WordingVariant("authority-v1", "authority", "Our experts are confident: ", ""),
        WordingVariant("social_proof-v1", "social_proof", "", " Everyone is watching it."),
    )
}


class ResearchMode:
    """Explicit research context; wording variants exist only inside it."""

    def __init__(self, study: StudyConfig, enabled: bool = False) -> None:
        if not enabled:
            raise ResearchModeError("research mode must be enabled explicitly")
        self.study = study
        self.active = True

    def apply_wording(self, explanation: Explanation, variant_id: str) -> Explanation:
        if not self.active:
            raise ResearchModeError("research mode is closed")
        if explanation.text is None:
            return explanation
        variant = VARIANTS[variant_id]
        return dataclasses.replace(
            explanation,
            text=variant.render(explanation.text),
            generator=f"{explanation.generator}+wording:{variant_id}",
        )

    def close(self) -> None:
        self.active = False


def apply_wording(
    explanation: Explanation, variant_id: str, mode: ResearchMode | None
) -> Explanation:
    """Guard used by callers: refuses to alter explanations outside research mode."""
    if mode is None:
        raise ResearchModeError("wording variants require an explicit research mode")
    return mode.apply_wording(explanation, variant_id)


# --- free text --------------------------------------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_DIGITS = re.compile(r"\d{5,}")


def sanitize_free_text(text: str, max_chars: int = 500) -> tuple[str, list[str]]:
    flags = []
    out = text
    for name, pattern, token in (
        ("email", _EMAIL, "[EMAIL]"),
        ("url", _URL, "[URL]"),
        ("phone", _PHONE, "[PHONE]"),
        ("digits", _DIGITS, "[NUMBER]"),
    ):
        if pattern.search(out):
            flags.append(name)
            out = pattern.sub(token, out)
    out = " ".join(out.split())
    if len(out) > max_chars:
        out = out[:max_chars] + "…"
        flags.append("truncated")
    return out, flags


CODEBOOK = {
    "trust": ("trust", "reliable", "believe"),
    "relevance": ("relevant", "match", "like what"),
    "manipulation": ("pushy", "manipulat", "salesy", "fake"),
    "transparency": ("understand", "why", "explain"),
}


def keyword_coder(text: str) -> list[str]:
    low = text.lower()
    return sorted(code for code, words in CODEBOOK.items() if any(w in low for w in words))


def code_free_text(
    responses: Sequence[Mapping[str, Any]],
    coder: Callable[[str], list[str]] = keyword_coder,
) -> list[dict[str, Any]]:
    """Sanitize first, then code. ``coder`` only ever receives sanitized text."""
    out = []
    for r in responses:
        clean, flags = sanitize_free_text(str(r.get("text", "")))
        out.append(
            {
                "participant_id": r["participant_id"],
                "variant": r.get("variant"),
                "text": clean,
                "sanitization_flags": ";".join(flags),
                "codes": ";".join(coder(clean)),
            }
        )
    return out


def export_coding(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["participant_id", "variant", "text", "sanitization_flags", "codes"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


# --- stress test --------------------------------------------------------------
def stress_test(
    responses: Sequence[Mapping[str, Any]],
    worded: Mapping[str, tuple[Explanation, EvidencePayload]],
    catalog: Sequence[Item],
    *,
    baseline_variant: str = "neutral-v1",
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Persuasion (acceptance by variant) and correctness (fidelity) reported separately.

    ``responses``: ``{"variant", "acceptance", "simulated"}`` rows.
    ``worded``: variant id -> (worded explanation, evidence payload) used for fidelity.
    """
    variants = sorted({str(r["variant"]) for r in responses})
    if baseline_variant not in variants:
        raise ValueError("baseline variant has no responses")
    acc = {v: [float(r["acceptance"]) for r in responses if r["variant"] == v] for v in variants}
    persuasion: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for v in variants:
        if v == baseline_variant:
            continue
        diff, lo, hi, _, _, p, d = compare_groups(
            acc[v], acc[baseline_variant], n_resamples, seed, v
        )
        pvalues.append(p)
        persuasion.append(
            {
                "variant": v,
                "acceptance_diff": diff,
                "ci_low": lo,
                "ci_high": hi,
                "p": p,
                "cohens_d": d,
            }
        )
    adjusted = holm_adjust(pvalues)
    for row, padj in zip(persuasion, adjusted, strict=True):
        row["p_holm"] = padj
    correctness = {}
    for v, (exp, payload) in sorted(worded.items()):
        res = validate_explanation(exp, payload, catalog)
        correctness[v] = {
            "accepted": res.accepted,
            "unsupported_rate": res.unsupported_rate,
            "unsupported_claims": [c.value for c in res.unsupported],
        }
    return {
        "simulated": all(bool(r.get("simulated")) for r in responses),
        "persuasion": persuasion,
        "correctness": correctness,
        "interpretation": (
            "Persuasion measures whether wording changes acceptance; correctness measures "
            "whether the worded explanation stays faithful to recommender evidence. A variant "
            "can raise acceptance while being unfaithful."
        ),
    }


def research_report(
    study: StudyConfig,
    analysis: Mapping[str, Any],
    stress: Mapping[str, Any] | None = None,
    fidelity_summary: Mapping[str, Any] | None = None,
) -> str:
    simulated = analysis["data_status"].startswith("SIMULATED")
    if not simulated and study.data_mode != "real":
        raise StudyGovernanceError("real-data report requires a real-mode study config")
    lines = [f"# Research Report: {study.study_id} (v{study.version})", ""]
    if simulated:
        lines += [
            "> **SIMULATED DATA — software validation only. These results are not evidence "
            "about real people.**",
            "",
        ]
    lines += ["## Research Questions & Hypotheses", ""]
    lines += [f"- RQ: {q}" for q in study.research_questions]
    lines += [
        f"- {h.hypothesis_id}: {h.statement} ({h.outcome}: {h.treatment} vs {h.control}, "
        f"{h.direction})"
        for h in study.hypotheses
    ]
    lines += [
        "",
        "## Methods",
        "",
        f"- Conditions: {', '.join(study.conditions)}",
        f"- Randomization: permuted blocks of {study.block_size}, seed {study.seed}",
        f"- Primary outcome: {study.primary_outcome}; secondary: "
        f"{', '.join(study.secondary_outcomes) or 'none'}",
        f"- Exclusion rules: {dict(study.exclusion_rules)}",
        f"- Analysis: Welch t-tests, bootstrap 95% CIs, {analysis['multiple_comparison_policy']}",
        f"- Ethics status: {study.ethics.status}"
        + (f" ({study.ethics.reference})" if study.ethics.reference else ""),
        "",
        "## Results",
        "",
        f"Enrolled {analysis['n_enrolled']}, analyzed {analysis['n_analyzed']}.",
        "",
        "| Test | Outcome | Δ mean | 95% CI | p (Holm) | d |",
        "|---|---|---|---|---|---|",
    ]
    for c in analysis["comparisons"]:
        d = "n/a" if c["cohens_d"] is None else f"{c['cohens_d']:.2f}"
        lines.append(
            f"| {c['hypothesis_id']}{'' if c['pre_specified'] else ' (exploratory)'} | "
            f"{c['outcome']} | {c['mean_diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
            f"| {c['p_holm']:.3g} | {d} |"
        )
    lines += ["", "## Explanation Fidelity (system-side, separate from perception)", ""]
    if fidelity_summary:
        lines += [f"- {k}: {v}" for k, v in sorted(fidelity_summary.items())]
    else:
        lines.append("- Not provided.")
    lines += ["", "## Persuasion vs Correctness", ""]
    if stress:
        for row in stress["persuasion"]:
            corr = stress["correctness"].get(row["variant"], {})
            lines.append(
                f"- {row['variant']}: acceptance Δ {row['acceptance_diff']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] (persuasion); fidelity "
                f"{'pass' if corr.get('accepted') else 'FAIL'} (correctness)"
            )
        lines.append(f"- {stress['interpretation']}")
    else:
        lines.append("- Stress test not run.")
    lines += [
        "",
        "## Limitations",
        "",
        "- Perception outcomes are separate from offline recommendation relevance.",
        "- Wording variants were applied only in research mode and never changed rankings.",
        "- Free text is sanitized before any coding; LLM-assisted coding is off by default.",
        f"- Data status: {analysis['data_status']}.",
        "",
    ]
    return "\n".join(lines)
