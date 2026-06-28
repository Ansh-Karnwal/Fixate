from __future__ import annotations

import json
import re

from agents.openai_client import complete_vision_json
from models import DemographicSegment


def _segment_id(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or f"segment-{index}"


def _fallback_segments(target_customer: str, goal: str) -> list[DemographicSegment]:
    base = target_customer.strip() or "likely buyers"
    return [
        DemographicSegment(
            id="primary-intent",
            name=f"{base.title()} With Active Intent",
            summary=f"People already looking for a solution and close to taking action on {goal}.",
            messaging_angle="Lead with the concrete outcome, proof, and a direct next step.",
            visual_direction="Make the offer and CTA visually dominant, with minimal supporting detail.",
            recommended_channel="Search, retargeting, or high-intent landing page traffic.",
            why_it_fits="They need fast clarity and enough confidence to act now.",
        ),
        DemographicSegment(
            id="value-sensitive",
            name="Value-Conscious Evaluators",
            summary="Buyers comparing options and trying to reduce perceived risk before committing.",
            messaging_angle="Emphasize value, guarantees, proof, and what they avoid by choosing this.",
            visual_direction="Use trust signals, comparison cues, and plain-language benefit hierarchy.",
            recommended_channel="Paid social, email nurture, or comparison landing pages.",
            why_it_fits="The asset can work harder by lowering uncertainty and making the tradeoff obvious.",
        ),
        DemographicSegment(
            id="aspirational-switchers",
            name="Aspirational Switchers",
            summary="People dissatisfied with the status quo and open to a better-looking, better-feeling alternative.",
            messaging_angle="Show the before/after transformation and make the desired identity feel attainable.",
            visual_direction="Use emotionally specific imagery, strong hero copy, and a confident CTA.",
            recommended_channel="Instagram, TikTok, display ads, or creator partnerships.",
            why_it_fits="They respond to emotional pull as much as rational proof.",
        ),
    ]


def _coerce_segments(raw: object, fallback: list[DemographicSegment]) -> list[DemographicSegment]:
    if not isinstance(raw, list):
        return fallback
    segments: list[DemographicSegment] = []
    for index, item in enumerate(raw[:5], start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"Segment {index}").strip()
        segment = {
            "id": str(item.get("id") or _segment_id(name, index)),
            "name": name,
            "summary": str(item.get("summary") or fallback[min(index - 1, len(fallback) - 1)].summary),
            "messaging_angle": str(item.get("messaging_angle") or fallback[0].messaging_angle),
            "visual_direction": str(item.get("visual_direction") or fallback[0].visual_direction),
            "recommended_channel": str(item.get("recommended_channel") or fallback[0].recommended_channel),
            "why_it_fits": str(item.get("why_it_fits") or fallback[0].why_it_fits),
        }
        segments.append(DemographicSegment(**segment))
    return segments or fallback


async def discover_demographics(
    image_png: bytes,
    text: str,
    target_customer: str,
    goal: str,
) -> tuple[list[DemographicSegment], bool]:
    fallback = _fallback_segments(target_customer, goal)
    data, live = await complete_vision_json(
        "You are Fixate's demographics agent. Return strict JSON only.",
        json.dumps(
            {
                "task": (
                    "Identify 3-5 practical outreach demographics or buyer segments for this product/campaign. "
                    "For each segment return id, name, summary, messaging_angle, visual_direction, "
                    "recommended_channel, and why_it_fits. These are targeting hypotheses for creative tuning, "
                    "not protected-class exclusion rules."
                ),
                "target_customer_hint": target_customer,
                "goal": goal,
                "asset_text": text[:3500],
            }
        ),
        image_png,
        {"segments": [segment.model_dump() for segment in fallback]},
    )
    if not live:
        return fallback, False
    return _coerce_segments(data.get("segments"), fallback), True


def select_demographic(
    segments: list[DemographicSegment],
    requested: str | None,
    target_customer: str,
) -> DemographicSegment:
    if requested:
        normalized = requested.strip().lower()
        for segment in segments:
            if normalized in {segment.id.lower(), segment.name.lower()}:
                return segment
        return DemographicSegment(
            id=_segment_id(requested, 0),
            name=requested.strip(),
            summary=f"User-selected outreach audience: {requested.strip()}.",
            messaging_angle=f"Make the message feel immediately relevant to {requested.strip()}.",
            visual_direction=f"Tune imagery, proof, and CTA language for {requested.strip()}.",
            recommended_channel="Use the channel where this audience already discovers comparable offers.",
            why_it_fits="The user selected this demographic as the outreach focus.",
        )
    return segments[0] if segments else _fallback_segments(target_customer, "increase conversions")[0]
