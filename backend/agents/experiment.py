from __future__ import annotations

import json

from agents.openai_client import complete_json
from models import ExperimentPlan, VariantResult


def _fallback_plan(best_variant: VariantResult | None, target_customer: str, goal: str) -> ExperimentPlan:
    variant_desc = best_variant.description if best_variant else "the clearer variant"
    return ExperimentPlan(
        hypothesis=f"If {target_customer} sees {variant_desc}, then {goal} will improve because the main blocker is addressed earlier.",
        recommended_channel="Primary landing page traffic or the highest-volume paid social ad set.",
        target_audience=target_customer,
        success_metric=goal,
        ab_test_setup="Split traffic 50/50 between the original and winning Fixate variant for one full business cycle or until sample size is reached.",
        next_step="Launch the variant as Treatment B and monitor conversion rate, CTA click-through, and bounce rate.",
    )


async def build_experiment_plan(
    best_variant: VariantResult | None,
    target_customer: str,
    goal: str,
) -> ExperimentPlan:
    fallback = _fallback_plan(best_variant, target_customer, goal)
    data = await complete_json(
        "You are an experiment planner. Return strict JSON for hypothesis, recommended_channel, target_audience, success_metric, ab_test_setup, next_step.",
        json.dumps(
            {
                "best_variant": best_variant.model_dump() if best_variant else None,
                "target_customer": target_customer,
                "goal": goal,
            }
        ),
        fallback.model_dump(),
    )
    merged = {**fallback.model_dump(), **data}
    for key in (
        "hypothesis",
        "recommended_channel",
        "target_audience",
        "success_metric",
        "ab_test_setup",
        "next_step",
    ):
        value = merged.get(key, "")
        if not isinstance(value, str):
            merged[key] = json.dumps(value, ensure_ascii=True)
    return ExperimentPlan(**merged)
