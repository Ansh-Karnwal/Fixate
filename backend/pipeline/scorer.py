from __future__ import annotations

import io
from statistics import mean

import numpy as np
from PIL import Image

from agents.openai_client import complete_vision_json, openai_live_enabled
from models import BuyerSignal, FixationRegion, RegionScore, ScoreResult, ZoneType
from pipeline.attention import predict_saliency


SIGNALS: tuple[BuyerSignal, ...] = (
    "attention",
    "desire",
    "trust",
    "memory",
    "cognitive_load",
    "self_relevance",
    "cta_strength",
)


def _clip(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _average_signals(region_scores: list[RegionScore]) -> dict[BuyerSignal, float]:
    if not region_scores:
        return {signal: 0.0 for signal in SIGNALS}
    return {
        signal: round(mean(region.signals[signal] for region in region_scores), 4)
        for signal in SIGNALS
    }


def _text_features(text: str) -> dict[str, float]:
    lower = text.lower()
    words = [w for w in lower.replace("\n", " ").split(" ") if w]
    cta_terms = ("start", "try", "buy", "shop", "join", "book", "get", "claim", "sign up", "learn")
    trust_terms = ("trusted", "guarantee", "secure", "proof", "customer", "review", "rated", "certified")
    desire_terms = ("save", "grow", "faster", "easy", "better", "free", "new", "exclusive", "proven")
    relevance_terms = ("you", "your", "teams", "founders", "marketers", "business")
    return {
        "word_count": min(1.0, len(words) / 220.0),
        "cta": min(1.0, sum(1 for term in cta_terms if term in lower) / 3.0),
        "trust": min(1.0, sum(1 for term in trust_terms if term in lower) / 3.0),
        "desire": min(1.0, sum(1 for term in desire_terms if term in lower) / 4.0),
        "relevance": min(1.0, sum(1 for term in relevance_terms if term in lower) / 5.0),
    }


def _fallback_region_signals(
    regions: list[FixationRegion],
    saliency_map: np.ndarray,
    text: str,
) -> list[dict[BuyerSignal, float]]:
    features = _text_features(text)
    height, width = saliency_map.shape
    out: list[dict[BuyerSignal, float]] = []
    for region in regions:
        x1, y1, x2, y2 = region.bbox
        center_y = ((y1 + y2) / 2) / max(height, 1)
        center_x = ((x1 + x2) / 2) / max(width, 1)
        above_fold = 1.0 - min(1.0, center_y)
        centered = 1.0 - min(1.0, abs(center_x - 0.5) * 2)
        sal = region.saliency_score
        load = _clip(0.28 + features["word_count"] * 0.42 + (1.0 - centered) * 0.12)
        out.append(
            {
                "attention": _clip(0.22 + sal * 0.72),
                "desire": _clip(0.28 + features["desire"] * 0.36 + sal * 0.18 + above_fold * 0.10),
                "trust": _clip(0.30 + features["trust"] * 0.42 + (1.0 - load) * 0.12),
                "memory": _clip(0.25 + sal * 0.25 + centered * 0.20 + features["desire"] * 0.15),
                "cognitive_load": load,
                "self_relevance": _clip(0.24 + features["relevance"] * 0.44 + features["desire"] * 0.12),
                "cta_strength": _clip(0.22 + features["cta"] * 0.52 + above_fold * 0.10),
            }
        )
    return out


def _classify_zone(saliency: float, signals: dict[BuyerSignal, float]) -> ZoneType:
    buyer_strength = mean(
        [
            signals["desire"],
            signals["trust"],
            signals["self_relevance"],
            signals["cta_strength"],
            signals["memory"],
        ]
    ) - signals["cognitive_load"] * 0.25
    if saliency >= 0.40 and buyer_strength >= 0.48:
        return "power_zone"
    if saliency >= 0.40:
        return "attention_trap"
    if buyer_strength >= 0.52:
        return "hidden_value"
    return "dead_zone"


def _fixate_score(signal_scores: dict[BuyerSignal, float], regions: list[RegionScore]) -> float:
    base = (
        0.19 * signal_scores["attention"]
        + 0.18 * signal_scores["desire"]
        + 0.16 * signal_scores["trust"]
        + 0.12 * signal_scores["memory"]
        + 0.16 * signal_scores["self_relevance"]
        + 0.15 * signal_scores["cta_strength"]
        + 0.04 * (1.0 - signal_scores["cognitive_load"])
    )
    zone_bonus = sum(0.025 for region in regions if region.zone == "power_zone")
    trap_penalty = sum(0.018 for region in regions if region.zone == "attention_trap")
    hidden_penalty = sum(0.012 for region in regions if region.zone == "hidden_value")
    return round(max(0.0, min(1.0, base + zone_bonus - trap_penalty - hidden_penalty)) * 100, 1)


def _blockers(signal_scores: dict[BuyerSignal, float], regions: list[RegionScore]) -> list[str]:
    blockers: list[str] = []
    if signal_scores["cta_strength"] < 0.45:
        blockers.append("weak_cta")
    if signal_scores["trust"] < 0.45:
        blockers.append("low_trust")
    if signal_scores["desire"] < 0.45:
        blockers.append("low_desire")
    if signal_scores["cognitive_load"] > 0.62:
        blockers.append("high_cognitive_load")
    if any(region.zone == "hidden_value" for region in regions):
        blockers.append("valuable_content_is_hidden")
    if any(region.zone == "attention_trap" for region in regions):
        blockers.append("attention_trap")
    return blockers[:5]


def _valid_zone(value: str) -> ZoneType:
    return value if value in {"power_zone", "attention_trap", "hidden_value", "dead_zone"} else "dead_zone"


async def _openai_score_result(
    screenshot_png: bytes,
    text: str,
    regions: list[FixationRegion],
    fallback: ScoreResult,
) -> ScoreResult:
    if not openai_live_enabled():
        return fallback
    data = await complete_vision_json(
        "You are Fixate's OpenAI buyer-response scorer. Return strict JSON only.",
        (
            "Score this marketing asset using buyer psychology. Use these exact seven signals "
            "from 0 to 1: attention, desire, trust, memory, cognitive_load, self_relevance, cta_strength. "
            "For each provided region bbox, return its signals and zone, where zone is one of "
            "power_zone, attention_trap, hidden_value, dead_zone. Also return overall signal_scores, "
            "fixate_score from 0 to 100, and blockers. JSON shape: "
            "{\"fixate_score\":0,\"signal_scores\":{...7 signals...},"
            "\"regions\":[{\"bbox\":[...],\"saliency\":0,\"signals\":{...},\"zone\":\"power_zone\"}],"
            "\"blockers\":[\"low_trust\"]}. "
            f"Regions: {[r.model_dump() for r in regions]}. Visible text excerpt: {text[:2500]}"
        ),
        screenshot_png,
        fallback.model_dump(),
    )
    try:
        scored_regions: list[RegionScore] = []
        raw_regions = data.get("regions") if isinstance(data.get("regions"), list) else []
        for index, region in enumerate(regions):
            raw = raw_regions[index] if index < len(raw_regions) and isinstance(raw_regions[index], dict) else {}
            raw_signals = raw.get("signals") if isinstance(raw.get("signals"), dict) else {}
            signals = {
                signal: _clip(raw_signals.get(signal, fallback.regions[index].signals[signal]))
                for signal in SIGNALS
            }
            scored_regions.append(
                RegionScore(
                    bbox=region.bbox,
                    saliency=_clip(raw.get("saliency", region.saliency_score)),
                    signals=signals,
                    zone=_valid_zone(str(raw.get("zone", fallback.regions[index].zone))),
                )
            )
        raw_signal_scores = data.get("signal_scores") if isinstance(data.get("signal_scores"), dict) else {}
        signal_scores = {
            signal: _clip(raw_signal_scores.get(signal, mean(r.signals[signal] for r in scored_regions)))
            for signal in SIGNALS
        }
        blockers = data.get("blockers") if isinstance(data.get("blockers"), list) else fallback.blockers
        return ScoreResult(
            fixate_score=round(max(0.0, min(100.0, float(data.get("fixate_score", fallback.fixate_score)))), 1),
            signal_scores=signal_scores,
            regions=scored_regions,
            blockers=[str(b) for b in blockers[:5]],
        )
    except Exception:
        return fallback


async def score_regions(
    screenshot_png: bytes,
    saliency_map: np.ndarray | None,
    text: str,
) -> ScoreResult:
    if saliency_map is None:
        attention = predict_saliency(screenshot_png)
        saliency_map = attention.saliency_map
        regions = attention.regions
    else:
        from pipeline.attention import _top_regions

        regions = _top_regions(saliency_map)

    with Image.open(io.BytesIO(screenshot_png)) as img:
        width, height = img.size
    if saliency_map.shape != (height, width):
        saliency_map = np.array(
            Image.fromarray((saliency_map * 255).astype(np.uint8)).resize((width, height))
        ).astype(np.float32) / 255.0

    scored_regions: list[RegionScore] = []
    signal_sets = _fallback_region_signals(regions, saliency_map, text)
    for region, signals in zip(regions, signal_sets):
        zone = _classify_zone(region.saliency_score, signals)
        scored_regions.append(
            RegionScore(
                bbox=region.bbox,
                saliency=region.saliency_score,
                signals=signals,
                zone=zone,
            )
        )

    signal_scores = _average_signals(scored_regions)
    fallback = ScoreResult(
        fixate_score=_fixate_score(signal_scores, scored_regions),
        signal_scores=signal_scores,
        regions=scored_regions,
        blockers=_blockers(signal_scores, scored_regions),
    )
    return await _openai_score_result(screenshot_png, text, regions, fallback)
