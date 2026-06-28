from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from models import CaptureRequest, CaptureResponse, OptimizeRequest
from agents.openai_client import openai_live_enabled
from agents.openai_client import openai_model, openai_required
from pipeline.attention import explain_region, predict_saliency_openai, render_heatmap_overlay
from pipeline.capture import capture_html, capture_image as capture_uploaded_image, capture_url
from pipeline.loop import create_job, jobs
from pipeline.scorer import score_regions


load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Fixate API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path(os.getenv("FIXATE_JOBS_DIR", tempfile.gettempdir())) / "fixate_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _capture_dir(capture_id: str) -> Path:
    return JOBS_DIR / capture_id


def _read_capture(capture_id: str) -> tuple[bytes, str]:
    folder = _capture_dir(capture_id)
    screenshot_path = folder / "screenshot.png"
    text_path = folder / "text.txt"
    if not screenshot_path.exists() or not text_path.exists():
        raise HTTPException(status_code=404, detail="Capture not found.")
    return screenshot_path.read_bytes(), text_path.read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "external_api": "openai",
        "openai_configured": openai_live_enabled(),
        "openai_required": openai_required(),
        "openai_model": openai_model(),
        "attention": "openai_vision" if openai_live_enabled() else "local_fallback",
        "scoring": "openai_vision" if openai_live_enabled() else "local_fallback",
        "image_editing": "responses_image_generation_tool" if openai_live_enabled() else "disabled",
    }


@app.get("/debug/openai")
async def debug_openai() -> dict:
    if not openai_live_enabled():
        return {"ok": False, "model": openai_model(), "error": "OPENAI_API_KEY is not configured."}
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.responses.create(
            model=openai_model(),
            input='Return exactly {"ok":true} as JSON.',
            text={"format": {"type": "json_object"}},
        )
        return {"ok": True, "model": openai_model(), "response": getattr(response, "output_text", "")}
    except Exception as exc:
        return {
            "ok": False,
            "model": openai_model(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


@app.post("/capture", response_model=CaptureResponse)
async def capture(req: CaptureRequest) -> CaptureResponse:
    try:
        if req.url:
            result = await capture_url(req.url)
        elif req.image_base64:
            result = await capture_uploaded_image(req.image_base64, req.image_name)
        else:
            result = await capture_html(req.html or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    capture_id = str(uuid.uuid4())
    folder = _capture_dir(capture_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "screenshot.png").write_bytes(result.screenshot_png)
    (folder / "text.txt").write_text(result.text, encoding="utf-8")
    (folder / "meta.json").write_text(
        json.dumps({"width": result.width, "height": result.height}),
        encoding="utf-8",
    )
    return CaptureResponse(
        capture_id=capture_id,
        text=result.text,
        width=result.width,
        height=result.height,
        image_url=f"/capture/{capture_id}/image",
    )


@app.get("/capture/{capture_id}/image")
async def capture_image(capture_id: str):
    path = _capture_dir(capture_id) / "screenshot.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Capture image not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/capture/{capture_id}/heatmap")
async def capture_heatmap(capture_id: str):
    screenshot_png, text = _read_capture(capture_id)
    attention = await predict_saliency_openai(screenshot_png, text)
    png = render_heatmap_overlay(screenshot_png, attention.saliency_map, attention.regions)
    path = _capture_dir(capture_id) / "heatmap.png"
    path.write_bytes(png)
    np.save(_capture_dir(capture_id) / "saliency.npy", attention.saliency_map)
    return FileResponse(path, media_type="image/png")


@app.post("/score/{capture_id}")
async def score_capture(capture_id: str):
    screenshot_png, text = _read_capture(capture_id)
    saliency_path = _capture_dir(capture_id) / "saliency.npy"
    saliency = np.load(saliency_path) if saliency_path.exists() else None
    result, live = await score_regions(screenshot_png, saliency, text)
    return {**result.model_dump(), "live": live}


@app.post("/optimize")
async def optimize(req: OptimizeRequest) -> dict:
    job = create_job(req)
    return {"job_id": job.job_id}


@app.get("/job/{job_id}/stream")
async def stream_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def events():
        seen: set[int] = set()
        for event in list(job.events):
            seen.add(int(event.get("seq", 0)))
            yield f"event: {event['event']}\ndata: {json.dumps(event)}\n\n"
        while job.status == "running" or not job.queue.empty():
            event = await job.queue.get()
            seq = int(event.get("seq", 0))
            if seq in seen:
                continue
            seen.add(seq)
            yield f"event: {event['event']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/job/{job_id}/result")
async def job_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "Job failed.")
    if job.status != "complete":
        raise HTTPException(status_code=202, detail="Job still running.")
    return job.result


@app.get("/job/{job_id}/image")
async def job_image(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.artifact_dir:
        raise HTTPException(status_code=404, detail="Job image not found.")
    path = job.artifact_dir / "screenshot.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job image not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/job/{job_id}/heatmap")
async def job_heatmap(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.artifact_dir:
        raise HTTPException(status_code=404, detail="Job heatmap not found.")
    path = job.artifact_dir / "heatmap.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job heatmap not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/job/{job_id}/region/{rank}/explain")
async def job_region_explain(job_id: str, rank: int):
    job = jobs.get(job_id)
    if not job or not job.artifact_dir:
        raise HTTPException(status_code=404, detail="Job not found.")
    if rank in job.region_explanations:
        return {"rank": rank, "explanation": job.region_explanations[rank], "cached": True}
    region = next((r for r in job.attention_regions if r.rank == rank), None)
    if region is None:
        raise HTTPException(status_code=404, detail="Region not found.")
    screenshot_path = job.artifact_dir / "screenshot.png"
    if not screenshot_path.exists():
        raise HTTPException(status_code=404, detail="Capture not found.")
    explanation = await explain_region(screenshot_path.read_bytes(), region, job.heatmap_text)
    job.region_explanations[rank] = explanation
    return {"rank": rank, "explanation": explanation, "cached": False}


@app.get("/job/{job_id}/variant/{iteration}/image")
async def job_variant_image(job_id: str, iteration: int):
    job = jobs.get(job_id)
    if not job or not job.artifact_dir:
        raise HTTPException(status_code=404, detail="Variant image not found.")
    path = job.artifact_dir / f"variant_{iteration}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Variant image not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/job/{job_id}/best-image")
async def job_best_image(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.artifact_dir:
        raise HTTPException(status_code=404, detail="Best image not found.")
    path = job.artifact_dir / "best.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Best image not found.")
    return FileResponse(path, media_type="image/png")
