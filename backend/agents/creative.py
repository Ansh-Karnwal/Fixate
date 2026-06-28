from __future__ import annotations

import json
import re
from uuid import uuid4

from agents.openai_client import complete_json
from models import Constraints, DemographicSegment, Diagnosis, VariantBrief


def _flatten_to_text(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        parts: list[str] = []
        for v in value.values():
            parts.extend(_flatten_to_text(v))
        return parts
    if isinstance(value, list):
        parts = []
        for v in value:
            parts.extend(_flatten_to_text(v))
        return parts
    return []


def _coerce_text(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    parts = _flatten_to_text(value)
    return " ".join(parts) if parts else fallback


def _fallback_brief(
    diagnosis: Diagnosis,
    constraints: Constraints,
    text: str,
    iteration: int,
    demographic: DemographicSegment | None = None,
) -> VariantBrief:
    blocker = diagnosis.hurting_conversion[0] if diagnosis.hurting_conversion else "general_clarity"
    color = constraints.brand.colors[0] if constraints.brand.colors else None
    font = constraints.brand.fonts[0] if constraints.brand.fonts else None
    tone = constraints.brand.tone or "clear and direct"
    audience = demographic.name if demographic else "target buyers"
    copy_by_blocker = {
        "low_trust": f"Show {audience} the proof they need before they commit.",
        "low_urgency": f"Give {audience} a clear reason to act today.",
        "high_confusion": f"Make the offer instantly clear for {audience}.",
        "high_cognitive_load": f"Help {audience} understand the value in one quick pass.",
        "valuable_content_is_hidden": f"Move the strongest {audience} proof point into the opening section.",
        "attention_trap": f"Shift attention from decoration to the promise {audience} cares about.",
        "weak_cta": f"Give {audience} one obvious next step",
    }
    return VariantBrief(
        id=f"variant-{iteration}-{uuid4().hex[:6]}",
        target_blocker=blocker,
        rewritten_copy=copy_by_blocker.get(blocker, f"Clarify the offer for {tone} buyers."),
        cta_instruction="Make the primary CTA more specific and keep it near the main promise.",
        visual_instruction=(
            demographic.visual_direction
            if demographic
            else ("Apply a subtle branded emphasis treatment to the primary CTA." if constraints.aggressiveness != "conservative" else "")
        ),
        layout_instruction="Move the CTA closer to the highest-attention region." if constraints.aggressiveness in {"balanced", "aggressive"} else "",
        demographic_focus=audience,
        color=color,
        font=font,
        explanation=f"Targets {blocker} for {audience} with a {tone} variant that stays inside the selected constraints.",
    )


async def generate_variant_brief(
    diagnosis: Diagnosis,
    constraints: Constraints,
    text: str,
    target_customer: str,
    goal: str,
    iteration: int,
    demographic: DemographicSegment | None = None,
) -> tuple[VariantBrief, bool]:
    fallback = _fallback_brief(diagnosis, constraints, text, iteration, demographic)
    data, live = await complete_json(
        "You are Fixate's creative agent. Return one JSON variant brief and obey all constraints. Never edit locked elements.",
        json.dumps(
            {
                "task": "Generate one variant brief with id, target_blocker, rewritten_copy, cta_instruction, visual_instruction, layout_instruction, demographic_focus, color, font, explanation. Do not touch locked elements.",
                "diagnosis": diagnosis.model_dump(),
                "constraints": constraints.model_dump(),
                "target_customer": target_customer,
                "selected_demographic": demographic.model_dump() if demographic else None,
                "goal": goal,
                "text_excerpt": text[:5000],
                "iteration": iteration,
            }
        ),
        fallback.model_dump(),
    )
    merged = {**fallback.model_dump(), **data}
    if not merged.get("id"):
        merged["id"] = fallback.id
    for key in (
        "target_blocker",
        "rewritten_copy",
        "cta_instruction",
        "visual_instruction",
        "layout_instruction",
        "demographic_focus",
        "explanation",
    ):
        merged[key] = _coerce_text(merged.get(key), getattr(fallback, key))
    touches = merged.get("touches_locked_element")
    if touches is not None and not isinstance(touches, str):
        parts = _flatten_to_text(touches)
        merged["touches_locked_element"] = " ".join(parts) if parts else None
    if constraints.brand.colors:
        allowed_colors = {color.upper() for color in constraints.brand.colors}
        proposed_color = str(merged.get("color") or "").upper()
        if proposed_color not in allowed_colors or not re.match(r"^#[0-9A-F]{6}$", proposed_color):
            merged["color"] = constraints.brand.colors[0]
    elif merged.get("color") and not re.match(r"^#[0-9A-Fa-f]{6}$", str(merged["color"])):
        merged["color"] = None

    if constraints.brand.fonts:
        allowed_fonts = {font.strip().lower() for font in constraints.brand.fonts}
        proposed_font = str(merged.get("font") or "").strip().lower()
        if proposed_font not in allowed_fonts:
            merged["font"] = constraints.brand.fonts[0]
    elif merged.get("font") and len(str(merged["font"])) > 40:
        merged["font"] = None
    return VariantBrief(**merged), live
