import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from podcast_signal_extractor import (
    PodcastSignal,
    SignalCategory,
    SignalDirection,
    cluster_repeated_entities,
    evidence_summary,
    rank_signals,
)


def test_podcast_signal_requires_timestamp_quote_and_source():
    signal = PodcastSignal(
        entity="Acme AI",
        ticker="ACME",
        category=SignalCategory.FORECAST,
        claim="CEO says capex will double in 2026",
        timestamp_seconds=125,
        quote="We expect to double infrastructure spending next year.",
        source_episode="episode-42",
        direction=SignalDirection.POSITIVE,
        confidence=0.82,
    )

    assert signal.is_verifiable() is True
    assert "02:05" in evidence_summary(signal)
    assert "证据原句" in evidence_summary(signal)


def test_repeated_entity_clusters_need_three_verifiable_mentions():
    signals = [
        PodcastSignal("Energy Storage", SignalCategory.SENTIMENT_SHIFT, "mentioned", i, "quote", f"ep-{i}", confidence=0.6)
        for i in range(3)
    ] + [
        PodcastSignal("No Timestamp", SignalCategory.RISK, "bad", -1, "quote", "ep-x")
    ]

    clusters = cluster_repeated_entities(signals)
    assert set(clusters) == {"energy storage"}
    assert rank_signals(signals)[0].entity == "Energy Storage"
