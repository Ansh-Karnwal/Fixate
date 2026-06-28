from __future__ import annotations

import io
from urllib.parse import urlparse

from PIL import Image
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from models import CaptureResult, ElementBox


VIEWPORT = {"width": 1280, "height": 800}


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


async def _extract_visible_text(page) -> str:
    return await page.evaluate(
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


async def _extract_element_boxes(page) -> list[ElementBox]:
    raw = await page.evaluate(
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


async def _scroll_full_page(page) -> None:
    await page.evaluate(
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


async def _capture_loaded_page(page) -> CaptureResult:
    await _scroll_full_page(page)
    text = await _extract_visible_text(page)
    element_boxes = await _extract_element_boxes(page)
    png = await page.screenshot(type="png", full_page=True)
    with Image.open(io.BytesIO(png)) as img:
        width, height = img.size
    return CaptureResult(screenshot_png=png, text=text, width=width, height=height, element_boxes=element_boxes)


async def capture_url(url: str) -> CaptureResult:
    url = _normalise_url(url)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
            context = await browser.new_context(viewport=VIEWPORT)
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except PlaywrightTimeoutError:
                pass
            result = await _capture_loaded_page(page)
            await context.close()
            await browser.close()
            return result
    except PlaywrightTimeoutError as exc:
        raise RuntimeError(f"Timed out while loading {url}.") from exc
    except PlaywrightError as exc:
        raise RuntimeError(f"Could not capture {url}: {exc}") from exc


async def capture_html(html: str) -> CaptureResult:
    if not html.strip():
        raise ValueError("HTML cannot be empty.")
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
            context = await browser.new_context(viewport=VIEWPORT)
            page = await context.new_page()
            await page.set_content(html, wait_until="domcontentloaded", timeout=20_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5_000)
            except PlaywrightTimeoutError:
                pass
            result = await _capture_loaded_page(page)
            await context.close()
            await browser.close()
            return result
    except PlaywrightTimeoutError as exc:
        raise RuntimeError("Timed out while rendering uploaded HTML.") from exc
    except PlaywrightError as exc:
        raise RuntimeError(f"Could not render uploaded HTML: {exc}") from exc

