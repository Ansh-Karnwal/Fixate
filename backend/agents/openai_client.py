from __future__ import annotations

import base64
import json
import os
import re
from typing import Any


class OpenAIIntegrationError(RuntimeError):
    pass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def openai_live_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and not _env_flag("OPENAI_DISABLED")


def openai_required() -> bool:
    return _env_flag("OPENAI_LIVE")


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.4")


def _json_from_text(content: str, fallback: dict[str, Any]) -> dict[str, Any]:
    content = content.strip()
    if not content:
        return fallback
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else fallback
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return fallback
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else fallback
        except json.JSONDecodeError:
            return fallback


def _output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def _ensure_json_mode_prompt(text: str) -> str:
    if "json" in text.lower():
        return text
    return f"Return a json object only.\n\n{text}"


def _handle_error(exc: Exception, fallback: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    message = f"OpenAI API failed using {openai_model()}: {type(exc).__name__}: {exc}"
    if openai_required():
        raise OpenAIIntegrationError(message) from exc
    print(message, flush=True)
    return fallback, False


async def complete_json(system: str, user: str, fallback: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not openai_live_enabled():
        if openai_required():
            raise OpenAIIntegrationError("OPENAI_LIVE=true but OPENAI_API_KEY is not configured.")
        return fallback, False
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.responses.create(
            model=openai_model(),
            instructions=system,
            input=_ensure_json_mode_prompt(user),
            text={"format": {"type": "json_object"}},
        )
        data = _json_from_text(_output_text(response), fallback)
        return data, True
    except Exception as exc:
        return _handle_error(exc, fallback)


async def complete_vision_json(
    system: str,
    prompt: str,
    image_png: bytes,
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if not openai_live_enabled():
        if openai_required():
            raise OpenAIIntegrationError("OPENAI_LIVE=true but OPENAI_API_KEY is not configured.")
        return fallback, False
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        image_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        prompt = _ensure_json_mode_prompt(prompt)
        response = await client.responses.create(
            model=openai_model(),
            instructions=system,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            text={"format": {"type": "json_object"}},
        )
        data = _json_from_text(_output_text(response), fallback)
        return data, True
    except Exception as exc:
        return _handle_error(exc, fallback)
