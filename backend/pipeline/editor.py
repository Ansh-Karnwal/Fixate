from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

from agents.openai_client import openai_live_enabled, openai_model, openai_required
from models import Constraints, VariantBrief


def _prepare_openai_image(image_png: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_png)).convert("RGB")
    width, height = image.size
    max_side = 1536
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _image_data_url(image_png: bytes) -> str:
    prepared = _prepare_openai_image(image_png)
    return "data:image/png;base64," + base64.b64encode(prepared).decode("ascii")


def _find_generated_image(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) > 200 else None
    if isinstance(value, list):
        for item in value:
            found = _find_generated_image(item)
            if found:
                return found
        return None
    if hasattr(value, "model_dump"):
        return _find_generated_image(value.model_dump())
    if isinstance(value, dict):
        for key in ("result", "b64_json", "image_base64", "data"):
            found = _find_generated_image(value.get(key))
            if found:
                return found
        for item in value.values():
            found = _find_generated_image(item)
            if found:
                return found
    return None


def _decode_generated_image(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


async def _openai_edit(image_png: bytes, brief: VariantBrief, constraints: Constraints) -> tuple[bytes | None, str | None]:
    if not openai_live_enabled():
        return None, "OPENAI_LIVE=true but OPENAI_API_KEY is not configured." if openai_required() else "OPENAI_API_KEY is not configured."
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = (
            "You are Fixate's image editing agent. Edit the provided marketing asset itself. "
            "Do not place a generic banner on top of the image. Do not add a random text block. "
            "Preserve the original composition, product, brand feel, and recognizable layout, but make real visual edits "
            "to the existing creative so it better serves the strategy below. Improve hierarchy, CTA visibility, "
            "proof, clarity, cropping, contrast, and audience relevance as appropriate. If copy changes are needed, "
            "integrate them naturally into existing text regions or UI elements rather than covering the design. "
            f"Target blocker: {brief.target_blocker}. "
            f"New/revised copy: {brief.rewritten_copy}. "
            f"CTA instruction: {brief.cta_instruction}. "
            f"Visual instruction: {brief.visual_instruction}. "
            f"Layout instruction: {brief.layout_instruction}. "
            f"Demographic focus: {brief.demographic_focus}. "
            f"Brand colors: {constraints.brand.colors}. Fonts: {constraints.brand.fonts}. Tone: {constraints.brand.tone}. "
            f"Locked elements: {[item.model_dump() for item in constraints.locked_elements]}. "
            "Return one polished edited image."
        )
        response = await client.responses.create(
            model=openai_model(),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _image_data_url(image_png)},
                    ],
                }
            ],
            tools=[{"type": "image_generation"}],
            tool_choice={"type": "image_generation"},
        )
        generated = _find_generated_image(getattr(response, "output", None))
        if not generated:
            generated = _find_generated_image(response)
        if not generated:
            return None, "The model/tool call completed but returned no generated image."
        return _decode_generated_image(generated), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def apply_edits(
    image_png: bytes,
    edit_instructions: VariantBrief,
    constraints: Constraints,
) -> tuple[bytes | None, bool, str | None]:
    edited, error = await _openai_edit(image_png, edit_instructions, constraints)
    return edited, edited is not None, error
