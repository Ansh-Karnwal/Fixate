from __future__ import annotations

import asyncio
import base64
import binascii
import io
import sys
from urllib.parse import urlparse

from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from agents.openai_client import complete_vision_json
from models import CaptureResult, ElementBox


VIEWPORT = {"width": 1280, "height": 800}

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def _normalise_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url!r}")
    return url


def _extract_visible_text(page) -> str:
    return page.evaluate(
        """
        () => {
          const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            {
              acceptNode(node) {
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                if (['SCRIPT','STYLE','NOSCRIPT','HEAD'].includes(parent.tagName)) {
                  return NodeFilter.FILTER_REJECT;
                }
                const style = window.getComputedStyle(parent);
                if (
                  style.display === 'none' ||
                  style.visibility === 'hidden' ||
                  style.opacity === '0'
                ) return NodeFilter.FILTER_REJECT;
                const box = parent.getBoundingClientRect();
                if (box.width === 0 || box.height === 0) return NodeFilter.FILTER_REJECT;
                return node.textContent.trim()
                  ? NodeFilter.FILTER_ACCEPT
                  : NodeFilter.FILTER_SKIP;
              }
            }
          );
          const nodes = [];
          let node;
          while ((node = walker.nextNode())) {
            const text = node.textContent.replace(/\\s+/g, ' ').trim();
            if (text) {
              const rect = node.parentElement.getBoundingClientRect();
              nodes.push({ text, top: rect.top + window.scrollY, left: rect.left + window.scrollX });
            }
          }
          nodes.sort((a, b) => Math.abs(a.top - b.top) < 8 ? a.left - b.left : a.top - b.top);
          return nodes.map(n => n.text).join('\\n');
        }
        """
    )


def _extract_element_boxes(page) -> list[ElementBox]:
    raw = page.evaluate(
        """
        () => {
          const selector = 'h1,h2,h3,h4,h5,h6,p,button,a,img,svg,input,textarea,li,[role="button"]';
          const els = Array.from(document.querySelectorAll(selector));
          const boxes = [];
          for (const el of els) {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
            const rect = el.getBoundingClientRect();
            if (rect.width < 16 || rect.height < 10) continue;
            const top = rect.top + window.scrollY;
            const left = rect.left + window.scrollX;
            boxes.push({
              tag: el.tagName.toLowerCase(),
              bbox: [
                Math.round(left),
                Math.round(top),
                Math.round(left + rect.width),
                Math.round(top + rect.height),
              ],
            });
            if (boxes.length >= 400) break;
          }
          return boxes;
        }
        """
    )
    return [ElementBox(tag=item["tag"], bbox=item["bbox"]) for item in raw]


def _scroll_full_page(page) -> None:
    page.evaluate(
        """
        async () => {
          const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
          const height = Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight
          );
          const step = Math.max(240, Math.floor(window.innerHeight * 0.8));
          for (let y = 0; y < height; y += step) {
            window.scrollTo(0, y);
            await delay(60);
          }
          window.scrollTo(0, 0);
          await delay(100);
        }
        """
    )


def _capture_loaded_page(page) -> CaptureResult:
    _scroll_full_page(page)
    text = _extract_visible_text(page)
    element_boxes = _extract_element_boxes(page)
    png = page.screenshot(type="png", full_page=True)
    with Image.open(io.BytesIO(png)) as img:
        width, height = img.size
    return CaptureResult(screenshot_png=png, text=text, width=width, height=height, element_boxes=element_boxes)


def _capture_url_sync(url: str) -> CaptureResult:
    url = _normalise_url(url)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--no-sandbox"])
            try:
                context = browser.new_context(viewport=VIEWPORT)
                try:
                    page = context.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10_000)
                    except PlaywrightTimeoutError:
                        pass
                    return _capture_loaded_page(page)
                finally:
                    context.close()
            finally:
                browser.close()
    except PlaywrightTimeoutError as exc:
        raise RuntimeError(f"Timed out while loading {url}.") from exc
    except PlaywrightError as exc:
        raise RuntimeError(f"Could not capture {url}: {exc}") from exc


async def capture_url(url: str) -> CaptureResult:
    return await asyncio.to_thread(_capture_url_sync, url)


def _capture_html_sync(html: str) -> CaptureResult:
    if not html.strip():
        raise ValueError("HTML cannot be empty.")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--no-sandbox"])
            try:
                context = browser.new_context(viewport=VIEWPORT)
                try:
                    page = context.new_page()
                    page.set_content(html, wait_until="domcontentloaded", timeout=20_000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=5_000)
                    except PlaywrightTimeoutError:
                        pass
                    return _capture_loaded_page(page)
                finally:
                    context.close()
            finally:
                browser.close()
    except PlaywrightTimeoutError as exc:
        raise RuntimeError("Timed out while rendering uploaded HTML.") from exc
    except PlaywrightError as exc:
        raise RuntimeError(f"Could not render uploaded HTML: {exc}") from exc


async def capture_html(html: str) -> CaptureResult:
    return await asyncio.to_thread(_capture_html_sync, html)


def _decode_image_base64(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image must be valid base64 data.") from exc


def _normalise_image_bytes(image_base64: str) -> tuple[bytes, int, int]:
    raw = _decode_image_base64(image_base64)
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("Uploaded image could not be opened.") from exc

    width, height = image.size
    max_side = 2200
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="PNG")
    final_width, final_height = image.size
    return out.getvalue(), final_width, final_height


async def _describe_uploaded_image(image_png: bytes, image_name: str | None) -> str:
    fallback = {
        "visible_text": "",
        "description": "Uploaded marketing image.",
        "offer": "",
        "audience_clues": [],
    }
    data, live = await complete_vision_json(
        "You are Fixate's capture agent. Return strict JSON only.",
        (
            "Read this uploaded marketing image or campaign asset. Extract any visible text, "
            "summarize the product/offer, and note audience clues. Return JSON with "
            "visible_text, description, offer, and audience_clues. "
            f"File name: {image_name or 'uploaded image'}."
        ),
        image_png,
        fallback,
    )
    if not live:
        return f"Uploaded image: {image_name or 'marketing asset'}."
    visible_text = data.get("visible_text") if isinstance(data.get("visible_text"), str) else ""
    description = data.get("description") if isinstance(data.get("description"), str) else fallback["description"]
    offer = data.get("offer") if isinstance(data.get("offer"), str) else ""
    clues = data.get("audience_clues") if isinstance(data.get("audience_clues"), list) else []
    clue_text = ", ".join(str(clue) for clue in clues[:6])
    parts = [
        f"Visible text: {visible_text}" if visible_text else "",
        f"Asset description: {description}" if description else "",
        f"Offer: {offer}" if offer else "",
        f"Audience clues: {clue_text}" if clue_text else "",
    ]
    return "\n".join(part for part in parts if part).strip() or fallback["description"]


async def capture_image(image_base64: str, image_name: str | None = None) -> CaptureResult:
    image_png, width, height = await asyncio.to_thread(_normalise_image_bytes, image_base64)
    text = await _describe_uploaded_image(image_png, image_name)
    return CaptureResult(
        screenshot_png=image_png,
        text=text,
        width=width,
        height=height,
        element_boxes=[],
    )

