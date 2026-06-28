from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from agents.openai_client import complete_vision_json, openai_live_enabled
from models import FixationRegion


@dataclass
class AttentionResult:
    saliency_map: np.ndarray
    regions: list[FixationRegion]
    scan_path: list[list[int]]
    live: bool


def _normalise(saliency: np.ndarray) -> np.ndarray:
    saliency = saliency.astype(np.float32)
    mn = float(saliency.min())
    mx = float(saliency.max())
    if mx <= mn:
        return np.zeros_like(saliency, dtype=np.float32)
    return ((saliency - mn) / (mx - mn)).astype(np.float32)


def _heuristic_saliency(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    y_norm = y / max(height, 1)
    x_norm = x / max(width, 1)
    top_scan = np.exp(-y_norm * 7.5) * 0.48
    second_scan = np.exp(-np.abs(y_norm - 0.27) * 11.0) * 0.30
    left_bias = np.exp(-x_norm * 2.8) * 0.14
    center_bias = np.exp(-(((x_norm - 0.5) ** 2) / 0.18 + ((y_norm - 0.38) ** 2) / 0.12)) * 0.18
    return _normalise(top_scan + second_scan + left_bias + center_bias)


def _top_regions(saliency: np.ndarray, top_k: int = 6) -> list[FixationRegion]:
    height, width = saliency.shape
    work = saliency.copy()
    box_w = max(96, width // 4)
    box_h = max(80, height // 5)
    regions: list[FixationRegion] = []
    for rank in range(1, top_k + 1):
        py, px = np.unravel_index(int(np.argmax(work)), work.shape)
        x1 = max(0, int(px) - box_w // 2)
        x2 = min(width, int(px) + box_w // 2)
        y1 = max(0, int(py) - box_h // 2)
        y2 = min(height, int(py) + box_h // 2)
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(saliency[y1:y2, x1:x2].mean())
        regions.append(
            FixationRegion(
                rank=rank,
                bbox=[x1, y1, x2, y2],
                saliency_score=round(score, 4),
                peak_coords=[int(px), int(py)],
            )
        )
        work[y1:y2, x1:x2] = 0.0
    return regions


def _saliency_from_regions(height: int, width: int, regions: list[FixationRegion]) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    saliency = np.zeros((height, width), dtype=np.float32)
    for region in regions:
        px, py = region.peak_coords
        x1, y1, x2, y2 = region.bbox
        sigma_x = max(48, (x2 - x1) / 2)
        sigma_y = max(48, (y2 - y1) / 2)
        blob = region.saliency_score * np.exp(
            -(((x - px) ** 2) / (2 * sigma_x**2) + ((y - py) ** 2) / (2 * sigma_y**2))
        )
        saliency = np.maximum(saliency, blob.astype(np.float32))
    return _normalise(saliency) if regions else _heuristic_saliency(height, width)


def _coerce_openai_regions(data: dict, width: int, height: int) -> list[FixationRegion]:
    regions: list[FixationRegion] = []
    raw_regions = data.get("regions") if isinstance(data.get("regions"), list) else []
    for index, raw in enumerate(raw_regions[:6], start=1):
        try:
            bbox = [int(v) for v in raw.get("bbox", [])[:4]]
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            peak = raw.get("peak_coords") or [(x1 + x2) // 2, (y1 + y2) // 2]
            px = max(0, min(width - 1, int(peak[0])))
            py = max(0, min(height - 1, int(peak[1])))
            score = max(0.0, min(1.0, float(raw.get("saliency_score", 0.5))))
            regions.append(
                FixationRegion(
                    rank=int(raw.get("rank", index)),
                    bbox=[x1, y1, x2, y2],
                    saliency_score=round(score, 4),
                    peak_coords=[px, py],
                )
            )
        except Exception:
            continue
    return regions


def predict_saliency(screenshot_png: bytes) -> AttentionResult:
    image = Image.open(io.BytesIO(screenshot_png)).convert("RGB")
    width, height = image.size
    saliency = _heuristic_saliency(height, width)
    saliency = _normalise(saliency)
    regions = _top_regions(saliency)
    return AttentionResult(
        saliency_map=saliency,
        regions=regions,
        scan_path=[r.peak_coords for r in regions],
        live=False,
    )


async def predict_saliency_openai(screenshot_png: bytes, text: str = "") -> AttentionResult:
    image = Image.open(io.BytesIO(screenshot_png)).convert("RGB")
    width, height = image.size
    fallback = predict_saliency(screenshot_png)
    if not openai_live_enabled():
        return fallback

    data = await complete_vision_json(
        "You are an expert eye-tracking and marketing attention analyst. Return strict JSON only.",
        (
            "Analyze this marketing/page screenshot and predict the first six human fixation regions. "
            "Use pixel coordinates in the screenshot's coordinate system. Return JSON exactly as "
            "{\"regions\":[{\"rank\":1,\"bbox\":[x1,y1,x2,y2],\"saliency_score\":0.0-1.0,"
            "\"peak_coords\":[x,y],\"reason\":\"short reason\"}]}. "
            f"Screenshot size is {width}x{height}. Visible text excerpt: {text[:1200]}"
        ),
        screenshot_png,
        {"regions": [r.model_dump() for r in fallback.regions]},
    )
    regions = _coerce_openai_regions(data, width, height) or fallback.regions
    saliency = _saliency_from_regions(height, width, regions)
    return AttentionResult(
        saliency_map=saliency,
        regions=regions,
        scan_path=[r.peak_coords for r in regions],
        live=True,
    )


def render_heatmap_overlay(screenshot_png: bytes, saliency_map: np.ndarray) -> bytes:
    base = Image.open(io.BytesIO(screenshot_png)).convert("RGBA")
    width, height = base.size
    sal = Image.fromarray((_normalise(saliency_map) * 255).astype(np.uint8)).resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )
    sal_arr = np.array(sal, dtype=np.float32) / 255.0
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(255 * sal_arr, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(220 * (1 - np.abs(sal_arr - 0.55) / 0.55), 0, 220).astype(np.uint8)
    rgba[..., 2] = np.clip(255 * (1 - sal_arr), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(35 + 165 * sal_arr, 0, 190).astype(np.uint8)
    overlay = Image.alpha_composite(base, Image.fromarray(rgba, "RGBA"))
    out = io.BytesIO()
    overlay.convert("RGB").save(out, format="PNG")
    return out.getvalue()
