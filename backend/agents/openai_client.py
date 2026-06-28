from __future__ import annotations

import json
import os
from typing import Any


def openai_live_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


async def complete_json(system: str, user: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if not openai_live_enabled():
        return fallback
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return data if isinstance(data, dict) else fallback
    except Exception:
        return fallback


async def complete_vision_json(
    system: str,
    prompt: str,
    image_png: bytes,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not openai_live_enabled():
        return fallback
    try:
        import base64
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        image_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return data if isinstance(data, dict) else fallback
    except Exception:
        return fallback
