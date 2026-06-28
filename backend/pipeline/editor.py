from __future__ import annotations

import base64
import io
import os

from PIL import Image, ImageDraw, ImageFont

from agents.openai_client import openai_live_enabled
from models import Constraints, VariantBrief


def _wrap(text: str, max_chars: int = 42) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines[:3]


def _fallback_edit(image_png: bytes, brief: VariantBrief, constraints: Constraints) -> bytes:
    image = Image.open(io.BytesIO(image_png)).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    brand = brief.color or (constraints.brand.colors[0] if constraints.brand.colors else "#0D7D59")
    try:
        fill = tuple(int(brand.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) + (215,)
    except Exception:
        fill = (13, 125, 89, 215)

    band_h = max(130, min(260, height // 5))
    draw.rectangle((0, 0, width, band_h), fill=fill)
    font_title = ImageFont.load_default(size=34)
    font_cta = ImageFont.load_default(size=22)
    y = 26
    for line in _wrap(brief.rewritten_copy, 48):
        draw.text((32, y), line, fill=(255, 255, 255, 255), font=font_title)
        y += 40
    cta = brief.cta_instruction.replace("Make the primary CTA", "CTA").split(".")[0][:58]
    draw.rounded_rectangle((32, band_h - 54, min(width - 32, 470), band_h - 16), radius=8, fill=(255, 255, 255, 245))
    draw.text((50, band_h - 45), cta or "Start now", fill=(18, 28, 22, 255), font=font_cta)
    edited = Image.alpha_composite(image, overlay).convert("RGB")
    out = io.BytesIO()
    edited.save(out, format="PNG")
    return out.getvalue()


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


async def _openai_edit(image_png: bytes, brief: VariantBrief, constraints: Constraints) -> bytes | None:
    if not openai_live_enabled():
        return None
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prepared = _prepare_openai_image(image_png)
        image_file = io.BytesIO(prepared)
        image_file.name = "asset.png"
        prompt = (
            "Edit this marketing asset according to these instructions while preserving the original layout as much as possible. "
            f"New/revised copy: {brief.rewritten_copy}. "
            f"CTA instruction: {brief.cta_instruction}. "
            f"Visual instruction: {brief.visual_instruction}. "
            f"Layout instruction: {brief.layout_instruction}. "
            f"Brand colors: {constraints.brand.colors}. Fonts: {constraints.brand.fonts}. Tone: {constraints.brand.tone}. "
            "Return a polished launch-ready marketing image. Do not add unrelated elements."
        )
        response = await client.images.edit(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            image=image_file,
            prompt=prompt,
            size="1024x1024",
            quality="medium",
        )
        item = response.data[0]
        b64 = getattr(item, "b64_json", None)
        if b64:
            return base64.b64decode(b64)
        url = getattr(item, "url", None)
        if url:
            import httpx

            async with httpx.AsyncClient(timeout=45) as http:
                result = await http.get(url)
                result.raise_for_status()
                return result.content
    except Exception:
        return None
    return None


async def apply_edits(image_png: bytes, edit_instructions: VariantBrief, constraints: Constraints) -> bytes:
    edited = await _openai_edit(image_png, edit_instructions, constraints)
    return edited or _fallback_edit(image_png, edit_instructions, constraints)
