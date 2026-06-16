"""Tokenizer fertility checks for the pre-R3 multilingual freeze.

The R3 tokenizer is an irreversible training artifact, so the freeze needs a
small, repeatable gate before any expensive run starts. Fertility is measured on
fixed English, Turkish, and Old Turkic-script samples:

* English/Turkish use UTF-8 bytes per token. Higher is better.
* Old Turkic runes use tokens per rune. Lower is better.

The defaults encode the targets documented in ``docs/turkic-languages.md``:
Turkish >= 3.5 bytes/token and Old Turkic <= 1.5 tokens/rune. English regression
is checked only when a baseline tokenizer is provided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from orkhon.data.old_turkic import contains_old_turkic
from orkhon.tokenizer.tokenizer import load_tokenizer

DEFAULT_ENGLISH: tuple[str, ...] = (
    "Orkhon is a compact, auditable language model stack built from scratch.",
    "The model learns from web text, code, evaluation data, and careful post-training.",
)

DEFAULT_TURKISH: tuple[str, ...] = (
    "Orkhon, Türkçe yüzey biçimini ve eklemeli morfolojiyi verimli öğrenmelidir.",
    "Bilge Kağan yazıtları Türk dili tarihi için temel kaynaklardan biridir.",
)

DEFAULT_OLD_TURKIC: tuple[str, ...] = (
    "𐰋𐰃𐰠𐰏𐰀 𐰴𐰍𐰣 𐱅𐰇𐰼𐰰",
    "𐰚𐰇𐰠 𐱅𐰃𐰏𐰤 𐰋𐰃𐱅𐰃𐰏",
)


@dataclass(frozen=True)
class FertilityMetrics:
    """Comparable tokenizer fertility measurements."""

    tokenizer: str
    english_bytes_per_token: float
    turkish_bytes_per_token: float
    old_turkic_tokens_per_rune: float
    english_tokens: int
    turkish_tokens: int
    old_turkic_tokens: int
    old_turkic_runes: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FertilityGate:
    """Gate result for one candidate tokenizer, optionally against a baseline."""

    candidate: FertilityMetrics
    baseline: FertilityMetrics | None
    passed: bool
    checks: dict[str, bool]
    targets: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "passed": self.passed,
            "checks": self.checks,
            "targets": self.targets,
        }


def _non_empty(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple(line.strip() for line in lines if line.strip())


def read_sample_file(path: str | Path) -> tuple[str, ...]:
    """Read a UTF-8 sample file, one non-empty line per sample."""
    return _non_empty(Path(path).read_text(encoding="utf-8").splitlines())


def _bytes_per_token(tokenizer, texts: Iterable[str]) -> tuple[float, int]:
    text = "\n".join(_non_empty(texts))
    token_count = len(tokenizer.encode(text))
    byte_count = len(text.encode("utf-8"))
    return byte_count / max(token_count, 1), token_count


def _tokens_per_rune(tokenizer, texts: Iterable[str]) -> tuple[float, int, int]:
    text = "\n".join(_non_empty(texts))
    token_count = len(tokenizer.encode(text))
    rune_count = sum(1 for ch in text if contains_old_turkic(ch))
    return token_count / max(rune_count, 1), token_count, rune_count


def measure_fertility(
    tokenizer_dir: str | Path,
    *,
    english: Iterable[str] = DEFAULT_ENGLISH,
    turkish: Iterable[str] = DEFAULT_TURKISH,
    old_turkic: Iterable[str] = DEFAULT_OLD_TURKIC,
) -> FertilityMetrics:
    """Measure fertility for one tokenizer directory."""
    tokenizer = load_tokenizer(tokenizer_dir)
    english_bpt, english_tokens = _bytes_per_token(tokenizer, english)
    turkish_bpt, turkish_tokens = _bytes_per_token(tokenizer, turkish)
    rune_tpr, rune_tokens, rune_count = _tokens_per_rune(tokenizer, old_turkic)
    return FertilityMetrics(
        tokenizer=str(tokenizer_dir),
        english_bytes_per_token=english_bpt,
        turkish_bytes_per_token=turkish_bpt,
        old_turkic_tokens_per_rune=rune_tpr,
        english_tokens=english_tokens,
        turkish_tokens=turkish_tokens,
        old_turkic_tokens=rune_tokens,
        old_turkic_runes=rune_count,
    )


def evaluate_fertility_gate(
    candidate_dir: str | Path,
    *,
    baseline_dir: str | Path | None = None,
    english: Iterable[str] = DEFAULT_ENGLISH,
    turkish: Iterable[str] = DEFAULT_TURKISH,
    old_turkic: Iterable[str] = DEFAULT_OLD_TURKIC,
    min_turkish_bytes_per_token: float = 3.5,
    max_old_turkic_tokens_per_rune: float = 1.5,
    max_english_regression: float = 0.0,
) -> FertilityGate:
    """Evaluate the pre-R3 fertility gate.

    ``max_english_regression`` is a fractional tolerance. The default ``0.0``
    means the candidate must be at least as fertile on English as the baseline.
    """
    candidate = measure_fertility(
        candidate_dir, english=english, turkish=turkish, old_turkic=old_turkic
    )
    baseline = (
        measure_fertility(
            baseline_dir, english=english, turkish=turkish, old_turkic=old_turkic
        )
        if baseline_dir is not None
        else None
    )

    checks = {
        "turkish_target": candidate.turkish_bytes_per_token >= min_turkish_bytes_per_token,
        "old_turkic_target": candidate.old_turkic_tokens_per_rune <= max_old_turkic_tokens_per_rune,
    }
    if baseline is not None:
        floor = baseline.english_bytes_per_token * (1.0 - max_english_regression)
        checks["english_no_regression"] = candidate.english_bytes_per_token >= floor

    return FertilityGate(
        candidate=candidate,
        baseline=baseline,
        passed=all(checks.values()),
        checks=checks,
        targets={
            "min_turkish_bytes_per_token": min_turkish_bytes_per_token,
            "max_old_turkic_tokens_per_rune": max_old_turkic_tokens_per_rune,
            "max_english_regression": max_english_regression,
        },
    )
