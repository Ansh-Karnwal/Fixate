from __future__ import annotations

import asyncio
import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agents.openai_client import complete_vision_json, openai_live_enabled
from models import ElementBox, FixationRegion

BAND_HEIGHT = 800
MAX_BANDS = 12
MAX_REGIONS_PER_BAND = 8
MAX_ELEMENTS_PER_BAND = 30
MIN_HEATMAP_SCORE = 0.18
NMS_RADIUS = 48
SCAN_PATH_COUNT = 12
MAX_BOX_AREA_FRAC = 0.32


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


def _visual_saliency(image: Image.Image) -> np.ndarray:
    """Content-aware fallback saliency.

    This is not a replacement for real eye tracking or OpenAI vision, but it is
    much better than returning the same top-left heatmap for every same-size
    image. It combines contrast edges, color saturation, dark/light text-like
    contrast, and a modest above-the-fold scan bias.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    edges = _normalise(grad_x + grad_y)

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = _normalise(max_channel - min_channel)
    contrast = _normalise(np.abs(gray - float(gray.mean())))

    # Text/buttons often create compact high-contrast areas. Emphasize those
    # without letting a large photo or full-color background dominate.
    text_like = _normalise(edges * (0.45 + contrast) * (0.55 + saturation * 0.45))
    position_bias = _heuristic_saliency(height, width)
    saliency = 0.42 * text_like + 0.25 * edges + 0.18 * saturation + 0.15 * position_bias
    return _normalise(saliency)


def _nearest_element(px: int, py: int, element_boxes: list[ElementBox]) -> list[int] | None:
    best: list[int] | None = None
    best_dist = float("inf")
    for el in element_boxes:
        x1, y1, x2, y2 = el.bbox
        if x1 <= px <= x2 and y1 <= py <= y2:
            return [x1, y1, x2, y2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = (cx - px) ** 2 + (cy - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best = [x1, y1, x2, y2]
    return best if best_dist < 140**2 else None


def _top_regions(
    saliency: np.ndarray,
    top_k: int = 8,
    element_boxes: list[ElementBox] | None = None,
) -> list[FixationRegion]:
    height, width = saliency.shape
    work = saliency.copy()
    box_w = max(96, width // 4)
    box_h = max(80, height // 5)
    regions: list[FixationRegion] = []
    for rank in range(1, top_k + 1):
        py, px = np.unravel_index(int(np.argmax(work)), work.shape)
        snapped = _nearest_element(int(px), int(py), element_boxes) if element_boxes else None
        if snapped:
            x1, y1, x2, y2 = snapped
        else:
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
        clear_y1, clear_y2 = max(0, y1), min(height, y2)
        clear_x1, clear_x2 = max(0, x1), min(width, x2)
        work[clear_y1:clear_y2, clear_x1:clear_x2] = 0.0
    return regions


def _saliency_from_regions(height: int, width: int, regions: list[FixationRegion]) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    saliency = np.zeros((height, width), dtype=np.float32)
    for region in regions:
        px, py = region.peak_coords
        x1, y1, x2, y2 = region.bbox
        sigma_x = min(140, max(34, (x2 - x1) / 2.2))
        sigma_y = min(140, max(34, (y2 - y1) / 2.2))
        blob = region.saliency_score * np.exp(
            -(((x - px) ** 2) / (2 * sigma_x**2) + ((y - py) ** 2) / (2 * sigma_y**2))
        )
        saliency = np.maximum(saliency, blob.astype(np.float32))
    return _normalise(saliency) if regions else _heuristic_saliency(height, width)


def _coerce_openai_regions(
    data: dict,
    width: int,
    height: int,
    element_boxes: list[ElementBox] | None = None,
) -> list[FixationRegion]:
    elements = element_boxes or []
    regions: list[FixationRegion] = []
    raw_regions = data.get("regions") if isinstance(data.get("regions"), list) else []
    for index, raw in enumerate(raw_regions[:MAX_REGIONS_PER_BAND], start=1):
        try:
            element_id = raw.get("element_id")
            bbox = None
            if isinstance(element_id, int) and 0 <= element_id < len(elements):
                bbox = list(elements[element_id].bbox)
            if bbox is None:
                bbox = [int(v) for v in raw.get("bbox", [])[:4]]
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            # Drop diffuse whole-section boxes (the model occasionally returns one big
            # low-confidence rectangle covering a whole card); the tight element boxes
            # inside it already carry the real hotspots.
            if (x2 - x1) * (y2 - y1) > MAX_BOX_AREA_FRAC * width * height:
                continue
            peak = raw.get("peak_coords") or [(x1 + x2) // 2, (y1 + y2) // 2]
            px = max(0, min(width - 1, int(peak[0])))
            py = max(0, min(height - 1, int(peak[1])))
            score = max(0.0, min(1.0, float(raw.get("saliency_score", 0.5))))
            reason = raw.get("reason")
            regions.append(
                FixationRegion(
                    rank=int(raw.get("rank", index)),
                    bbox=[x1, y1, x2, y2],
                    saliency_score=round(score, 4),
                    peak_coords=[px, py],
                    reason=str(reason)[:300] if isinstance(reason, str) else "",
                )
            )
        except Exception:
            continue
    return regions


def _band_ranges(height: int) -> list[tuple[int, int]]:
    if height <= BAND_HEIGHT:
        return [(0, height)]
    num_bands = min(MAX_BANDS, -(-height // BAND_HEIGHT))
    band_height = -(-height // num_bands)
    ranges: list[tuple[int, int]] = []
    y = 0
    while y < height and len(ranges) < num_bands:
        y2 = min(height, y + band_height)
        ranges.append((y, y2))
        y = y2
    return ranges


def _crop_png(image: Image.Image, y1: int, y2: int) -> bytes:
    band = image.crop((0, y1, image.width, y2))
    out = io.BytesIO()
    band.save(out, format="PNG")
    return out.getvalue()


def _band_elements(element_boxes: list[ElementBox], y1: int, y2: int) -> list[ElementBox]:
    band: list[ElementBox] = []
    for el in element_boxes:
        ex1, ey1, ex2, ey2 = el.bbox
        if ey2 <= y1 or ey1 >= y2:
            continue
        ty1 = max(0, ey1 - y1)
        ty2 = min(y2 - y1, ey2 - y1)
        if ty2 - ty1 < 6:
            continue
        band.append(ElementBox(tag=el.tag, bbox=[ex1, ty1, ex2, ty2]))
        if len(band) >= MAX_ELEMENTS_PER_BAND:
            break
    return band


async def _predict_band_regions(
    image: Image.Image,
    y1: int,
    y2: int,
    text: str,
    band_elements: list[ElementBox],
) -> tuple[list[FixationRegion], bool]:
    width = image.width
    band_height = y2 - y1
    fallback_regions = _top_regions(_visual_saliency(image.crop((0, y1, width, y2))), element_boxes=band_elements)
    elements_payload = [{"id": i, "tag": el.tag, "bbox": el.bbox} for i, el in enumerate(band_elements)]
    data, live = await complete_vision_json(
        "You are an expert eye-tracking and marketing attention analyst. Return strict JSON only.",
        (
            f"Analyze this section of a marketing/page screenshot and predict up to {MAX_REGIONS_PER_BAND} "
            "human fixation regions within THIS SECTION ONLY, favoring several smaller, distinct hotspots "
            "over one large area. Use pixel coordinates local to this section, where the section's own "
            "top-left corner is (0,0). A list of real page elements in this section is provided below — "
            "whenever a fixation point lands on one of them, set element_id to that element's id so the "
            "region bbox snaps exactly to that element's real bounds, instead of guessing a bbox. Only "
            "provide a freehand bbox when no listed element fits. Return JSON exactly as "
            "{\"regions\":[{\"rank\":1,\"element_id\":0,\"bbox\":[x1,y1,x2,y2],\"saliency_score\":0.0-1.0,"
            "\"peak_coords\":[x,y],\"reason\":\"short reason\"}]} (element_id and bbox are both optional, "
            "but at least one must be present). "
            f"Section size is {width}x{band_height}. Visible text excerpt: {text[:1200]}. "
            f"Elements: {elements_payload}"
        ),
        _crop_png(image, y1, y2),
        {"regions": [r.model_dump() for r in fallback_regions]},
    )
    if not live:
        return fallback_regions, False
    regions = _coerce_openai_regions(data, width, band_height, band_elements) or fallback_regions
    return regions, True


async def explain_region(
    screenshot_png: bytes,
    region: FixationRegion,
    text: str = "",
) -> str:
    """Lazily explain why a single fixation region draws the eye. Called on demand (one
    click = one focused call on the cropped region), never eagerly for the whole page."""
    image = Image.open(io.BytesIO(screenshot_png)).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = region.bbox
    pad_x = max(40, (x2 - x1) // 3)
    pad_y = max(40, (y2 - y1) // 3)
    crop_box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    if not openai_live_enabled():
        if region.reason:
            return region.reason
        pct = round(region.saliency_score * 100)
        return (
            f"Fixation point #{region.rank}: predicted to capture roughly {pct}% relative "
            "attention based on its position, size, and contrast within the layout."
        )

    crop = image.crop(crop_box)
    out = io.BytesIO()
    crop.save(out, format="PNG")
    data, live = await complete_vision_json(
        "You are an eye-tracking analyst. Return strict JSON only.",
        (
            "The highlighted element in this cropped section was predicted as fixation point "
            f"#{region.rank} with a relative saliency of {region.saliency_score:.2f} (0-1). "
            "In one or two short sentences, explain WHY a buyer's eye lands here — reference "
            "concrete visual drivers like size, color/contrast, position, whitespace, or being "
            "a headline/number/CTA. Be specific to what you see. "
            'Return JSON: {"explanation":"..."}. '
            + (f"Model's quick note: {region.reason}. " if region.reason else "")
            + f"Nearby page text: {text[:600]}"
        ),
        out.getvalue(),
        {"explanation": region.reason or "This element stands out within its surrounding layout."},
    )
    explanation = data.get("explanation") if isinstance(data, dict) else None
    if isinstance(explanation, str) and explanation.strip():
        return explanation.strip()[:600]
    return region.reason or "This element stands out within its surrounding layout."


def predict_saliency(screenshot_png: bytes, element_boxes: list[ElementBox] | None = None) -> AttentionResult:
    image = Image.open(io.BytesIO(screenshot_png)).convert("RGB")
    width, height = image.size
    saliency = _visual_saliency(image)
    saliency = _normalise(saliency)
    regions = _top_regions(saliency, element_boxes=element_boxes)
    return AttentionResult(
        saliency_map=saliency,
        regions=regions,
        scan_path=[r.peak_coords for r in regions],
        live=False,
    )


def _suppress_overlaps(regions: list[FixationRegion], radius: float) -> list[FixationRegion]:
    """Greedy non-max suppression by peak distance: keep the highest-scoring region in
    any cluster so we don't stack many dots on the same spot, but keep horizontally or
    vertically adjacent distinct elements (e.g. side-by-side stat columns)."""
    kept: list[FixationRegion] = []
    for region in sorted(regions, key=lambda r: r.saliency_score, reverse=True):
        px, py = region.peak_coords
        if all((px - k.peak_coords[0]) ** 2 + (py - k.peak_coords[1]) ** 2 >= radius**2 for k in kept):
            kept.append(region)
    return kept


def _select_heatmap_regions(band_regions: list[list[FixationRegion]]) -> list[FixationRegion]:
    """Build the full set of regions for the heatmap. Every band contributes its real
    hotspots so there are no blind spots: keep each band's top region unconditionally,
    plus any region clearing the salience floor, then suppress exact-overlap duplicates."""
    pooled: list[FixationRegion] = []
    seen: set[int] = set()
    for regions in band_regions:
        if not regions:
            continue
        ranked = sorted(regions, key=lambda r: r.saliency_score, reverse=True)
        # Guarantee each non-empty band's strongest hotspot, regardless of global score.
        pooled.append(ranked[0])
        seen.add(id(ranked[0]))
        for region in ranked[1:]:
            if region.saliency_score >= MIN_HEATMAP_SCORE and id(region) not in seen:
                pooled.append(region)
                seen.add(id(region))
    return _suppress_overlaps(pooled, NMS_RADIUS)


async def predict_saliency_openai(
    screenshot_png: bytes,
    text: str = "",
    element_boxes: list[ElementBox] | None = None,
) -> AttentionResult:
    image = Image.open(io.BytesIO(screenshot_png)).convert("RGB")
    width, height = image.size
    fallback = predict_saliency(screenshot_png, element_boxes)
    if not openai_live_enabled():
        return fallback

    bands = _band_ranges(height)
    band_results = await asyncio.gather(
        *(
            _predict_band_regions(image, y1, y2, text, _band_elements(element_boxes or [], y1, y2))
            for y1, y2 in bands
        )
    )
    if not all(live for _, live in band_results):
        return fallback

    offset_band_regions: list[list[FixationRegion]] = []
    for (y1, _), (band_regions, _) in zip(bands, band_results):
        offset: list[FixationRegion] = []
        for region in band_regions:
            x1, ry1, x2, ry2 = region.bbox
            px, py = region.peak_coords
            offset.append(
                region.model_copy(update={
                    "bbox": [x1, ry1 + y1, x2, ry2 + y1],
                    "peak_coords": [px, py + y1],
                })
            )
        offset_band_regions.append(offset)

    selected = _select_heatmap_regions(offset_band_regions)
    selected.sort(key=lambda r: r.saliency_score, reverse=True)
    ranked_regions = [
        region.model_copy(update={"rank": rank}) for rank, region in enumerate(selected, start=1)
    ]
    saliency = _saliency_from_regions(height, width, ranked_regions)
    scan_path = [r.peak_coords for r in ranked_regions[:SCAN_PATH_COUNT]]
    return AttentionResult(
        saliency_map=saliency,
        regions=ranked_regions,
        scan_path=scan_path,
        live=True,
    )


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    dash: float = 9,
    gap: float = 7,
    fill: tuple[int, int, int, int] = (255, 255, 255, 235),
    width: int = 2,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    distance = 0.0
    drawing = True
    while distance < length:
        seg_end = min(distance + (dash if drawing else gap), length)
        if drawing:
            draw.line(
                [(x1 + dx * distance, y1 + dy * distance), (x1 + dx * seg_end, y1 + dy * seg_end)],
                fill=fill,
                width=width,
            )
        distance = seg_end
        drawing = not drawing


def _draw_scan_path(image: Image.Image, regions: list[FixationRegion]) -> Image.Image:
    # Only the top SCAN_PATH_COUNT regions get numbered dots + connecting lines so the
    # overlay stays readable; the heat itself already covers every region.
    ordered = [r for r in sorted(regions, key=lambda r: r.rank) if r.rank <= SCAN_PATH_COUNT]
    draw = ImageDraw.Draw(image, "RGBA")
    points = [tuple(r.peak_coords) for r in ordered]
    for p1, p2 in zip(points, points[1:]):
        _draw_dashed_line(draw, p1, p2)
    radius = 13
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    for region in ordered:
        x, y = region.peak_coords
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(46, 27, 110, 235),
            outline=(255, 255, 255, 255),
            width=2,
        )
        text = str(region.rank)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw / 2 - bbox[0], y - th / 2 - bbox[1]), text, fill=(255, 255, 255, 255), font=font)
    return image


def render_heatmap_overlay(
    screenshot_png: bytes,
    saliency_map: np.ndarray,
    regions: list[FixationRegion] | None = None,
) -> bytes:
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
    if regions:
        overlay = _draw_scan_path(overlay, regions)
    out = io.BytesIO()
    overlay.convert("RGB").save(out, format="PNG")
    return out.getvalue()
