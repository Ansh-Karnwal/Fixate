from __future__ import annotations

import json

from agents.openai_client import complete_json
from models import BuyerReaction, ScoreResult


def _fallback_reactions(score: ScoreResult, text: str) -> list[BuyerReaction]:
    signals = score.signal_scores
    reactions: list[BuyerReaction] = []
    checks = [
        ("confusion", "high_cognitive_load", signals["cognitive_load"], 0.62, "Copy feels dense or asks the buyer to process too much at once."),
        ("trust", "low_trust", 1 - signals["trust"], 0.55, "The page needs more proof, specificity, or reassurance before the ask."),
        ("desire", "low_desire", 1 - signals["desire"], 0.55, "The benefit is not yet strong enough to create pull."),
        ("urgency", "low_urgency", 0.62 if "now" not in text.lower() and "today" not in text.lower() else 0.25, 0.5, "There is little reason to act immediately."),
        ("cta_strength", "weak_cta", 1 - signals["cta_strength"], 0.55, "The next step could be clearer or more prominent."),
    ]
    for dimension, blocker, value, threshold, explanation in checks:
        if value >= threshold:
            severity = "high" if value >= threshold + 0.18 else "medium"
            reactions.append(
                BuyerReaction(
                    dimension=dimension,
                    severity=severity,
                    blocker=blocker,
                    explanation=explanation,
                )
            )
        else:
            reactions.append(
                BuyerReaction(
                    dimension=dimension,
                    severity="low",
                    blocker="none",
                    explanation=f"{dimension.replace('_', ' ').title()} is not a major blocker.",
                )
            )
    return reactions


async def run_buyer_panel(score: ScoreResult, text: str) -> list[BuyerReaction]:
    fallback = {"reactions": [r.model_dump() for r in _fallback_reactions(score, text)]}
    data = await complete_json(
        "You are a buyer reaction panel for conversion optimization. Return strict JSON.",
        json.dumps(
            {
                "task": "Return five reactions for confusion, trust, desire, urgency, cta_strength. Each item needs dimension, severity low|medium|high, blocker, explanation.",
                "score": score.model_dump(),
                "text_excerpt": text[:5000],
            }
        ),
        fallback,
    )
    return [BuyerReaction(**item) for item in data.get("reactions", fallback["reactions"])[:5]]

