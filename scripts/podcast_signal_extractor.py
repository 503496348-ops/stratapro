"""Timestamp-backed podcast signal model for Stratapro.

This module does not provide investment advice. It structures public audio or
transcript evidence into auditable signals that can be reviewed alongside market
and fundamental factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class SignalDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SignalCategory(str, Enum):
    COMPANY = "company"
    METRIC = "metric"
    FORECAST = "forecast"
    RISK = "risk"
    CATALYST = "catalyst"
    SENTIMENT_SHIFT = "sentiment_shift"


@dataclass(frozen=True)
class PodcastSignal:
    entity: str
    category: SignalCategory
    claim: str
    timestamp_seconds: int
    quote: str
    source_episode: str
    direction: SignalDirection = SignalDirection.NEUTRAL
    ticker: str | None = None
    confidence: float = 0.5

    def is_verifiable(self) -> bool:
        return (
            bool(self.entity.strip())
            and bool(self.claim.strip())
            and self.timestamp_seconds >= 0
            and bool(self.quote.strip())
            and bool(self.source_episode.strip())
            and 0 <= self.confidence <= 1
        )


def rank_signals(signals: Sequence[PodcastSignal]) -> tuple[PodcastSignal, ...]:
    """Rank by verifiability, confidence, and earlier timestamp for review."""
    return tuple(sorted(
        signals,
        key=lambda s: (not s.is_verifiable(), -s.confidence, s.timestamp_seconds),
    ))


def cluster_repeated_entities(signals: Iterable[PodcastSignal], minimum_mentions: int = 3) -> dict[str, list[PodcastSignal]]:
    """Find entities repeatedly mentioned across evidence packets."""
    grouped: dict[str, list[PodcastSignal]] = {}
    for signal in signals:
        if signal.is_verifiable():
            grouped.setdefault(signal.entity.lower(), []).append(signal)
    return {entity: items for entity, items in grouped.items() if len(items) >= minimum_mentions}


def evidence_summary(signal: PodcastSignal) -> str:
    ticker = f" / {signal.ticker}" if signal.ticker else ""
    minute, second = divmod(signal.timestamp_seconds, 60)
    return (
        f"[{minute:02d}:{second:02d}] {signal.entity}{ticker} · "
        f"{signal.category.value} · {signal.direction.value} · "
        f"confidence={signal.confidence:.2f} — {signal.claim}\n"
        f"证据原句：{signal.quote}"
    )
