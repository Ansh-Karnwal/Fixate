from __future__ import annotations

import json

from agents.openai_client import complete_json
from models import BuyerReaction, Diagnosis, ScoreResult


def _fallback_diagnosis(score: ScoreResult, reactions: list[BuyerReaction]) -> Diagnosis:
    power = sum(1 for r in score.regions if r.zone == "power_zone")
    traps = sum(1 for r in score.regions if r.zone == "attention_trap")
    hidden = sum(1 for r in score.regions if r.zone == "hidden_value")
    hurting = [r.blocker for r in reactions if r.severity in {"medium", "high"} and r.blocker != "none"]
    return Diagnosis(
        working=[f"{power} regions are already acting as power zones."] if power else ["The page earns attention in the first viewport."],
        ignored=[f"{hidden} valuable regions are under-attended."] if hidden else ["No major hidden-value zone detected."],
        hurting_conversion=hurting or score.blockers or ["general_clarity"],
        summary="Fix the highest-impact buyer blocker while preserving existing attention.",
    )


async def diagnose(score: ScoreResult, reactions: list[BuyerReaction]) -> tuple[Diagnosis, bool]:
    fallback = _fallback_diagnosis(score, reactions)
    data, live = await complete_json(
        "You are a conversion strategist. Return strict JSON matching working, ignored, hurting_conversion, summary.",
        json.dumps({"score": score.model_dump(), "buyer_reactions": [r.model_dump() for r in reactions]}),
        fallback.model_dump(),
    )
    merged = {**fallback.model_dump(), **data}
    for key in ("working", "ignored", "hurting_conversion"):
        value = merged.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            merged[key] = getattr(fallback, key)
    if not isinstance(merged.get("summary"), str):
        merged["summary"] = fallback.summary
    return Diagnosis(**merged), live

