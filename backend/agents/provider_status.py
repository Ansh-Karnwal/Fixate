from __future__ import annotations

import os

from agents.openai_client import openai_live_enabled, openai_model


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def meta_tribe_demo_enabled() -> bool:
    return _env_flag("META_TRIBE_DEMO")


def meta_tribe_model() -> str:
    return os.getenv("META_TRIBE_MODEL", "meta-tribe-demo-adapter")


def runtime_provider() -> str:
    return "openai" if openai_live_enabled() else "local_fallback"


def provider_status() -> dict:
    runtime = runtime_provider()
    demo_enabled = meta_tribe_demo_enabled()
    if demo_enabled:
        runtime_label = f"OpenAI {openai_model()}" if runtime == "openai" else "local fallback"
        return {
            "provider_label": "Meta TRIBE demo adapter",
            "presentation_provider": "meta_tribe_demo_adapter",
            "runtime_provider": runtime,
            "runtime_label": runtime_label,
            "meta_tribe_demo": True,
            "meta_tribe_model": meta_tribe_model(),
            "model_loaded": False,
            "note": "Demo adapter only. No Meta model is downloaded or executed; Fixate keeps the existing runtime path.",
        }
    return {
        "provider_label": "OpenAI API" if runtime == "openai" else "Local fallback",
        "presentation_provider": runtime,
        "runtime_provider": runtime,
        "runtime_label": f"OpenAI {openai_model()}" if runtime == "openai" else "local fallback",
        "meta_tribe_demo": False,
        "meta_tribe_model": None,
        "model_loaded": False,
        "note": None,
    }
