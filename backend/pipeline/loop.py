from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agents.buyer_panel import run_buyer_panel
from agents.creative import generate_variant_brief
from agents.demographics import discover_demographics, select_demographic
from agents.experiment import build_experiment_plan
from agents.openai_client import openai_required
from agents.strategist import diagnose
from models import BlockedEdit, DemographicSegment, FixationRegion, OptimizeRequest, ScoreResult, VariantBrief, VariantResult
from pipeline.attention import predict_saliency_openai, render_heatmap_overlay
from pipeline.capture import capture_html, capture_image, capture_url
from pipeline.constraints import violates_constraints
from pipeline.editor import apply_edits
from pipeline.scorer import score_regions


@dataclass
class Job:
    job_id: str
    status: str = "running"
    events: list[dict] = field(default_factory=list)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    result: dict | None = None
    error: str | None = None
    artifact_dir: Path | None = None
    # Fixation regions matching the currently saved heatmap.png, plus a lazily-filled
    # cache of per-region explanations (one entry per region the user clicks).
    attention_regions: list[FixationRegion] = field(default_factory=list)
    heatmap_text: str = ""
    region_explanations: dict[int, str] = field(default_factory=dict)


jobs: dict[str, Job] = {}


EVENT_AGENTS = {
    "capture_started": "Capture Agent",
    "capture_done": "Capture Agent",
    "demographics_started": "Demographics Agent",
    "demographics_ready": "Demographics Agent",
    "heatmap_ready": "Attention Agent",
    "scored": "Buyer-Response Scorer",
    "buyer_panel": "Buyer Panel Agents",
    "diagnosis_ready": "Growth Strategist Agent",
    "blocker_found": "Growth Strategist Agent",
    "variant_proposed": "Creative Agent",
    "variant_applied": "Image Editing Agent",
    "variant_image_failed": "Image Editing Agent",
    "variant_scored": "Buyer-Response Scorer",
    "edit_blocked": "Constraint Guard",
    "iteration_done": "Experiment Loop",
    "job_complete": "Experiment Agent",
    "job_error": "System",
}


async def emit(job: Job, event: str, data: dict | None = None) -> None:
    payload = {
        "seq": len(job.events) + 1,
        "event": event,
        "agent": EVENT_AGENTS.get(event, "Fixate"),
        "ts": time.time(),
        **(data or {}),
    }
    job.events.append(payload)
    await job.queue.put(payload)


def _updated_text(current_text: str, brief: VariantBrief) -> str:
    if brief.rewritten_copy in current_text:
        return current_text
    lines = [brief.rewritten_copy]
    if current_text.strip():
        lines.append(current_text.strip())
    return "\n\n".join(lines)


def _maybe_force_demo_violation(brief: VariantBrief, req: OptimizeRequest, iteration: int) -> VariantBrief:
    """Make locked-layout runs visibly exercise the constraint gate in fallback mode."""
    if iteration != 1:
        return brief
    if any(el.type == "layout" and el.value == "fixed" for el in req.constraints.locked_elements):
        clone = brief.model_copy(deep=True)
        clone.layout_instruction = clone.layout_instruction or "Move the primary CTA above the fold."
        return clone
    return brief


def _variant_record(
    brief: VariantBrief,
    *,
    before_score: float,
    after_score: float,
    accepted: bool,
    image_url: str | None,
) -> VariantResult:
    return VariantResult(
        id=brief.id,
        target_blocker=brief.target_blocker,
        description=brief.explanation or brief.visual_instruction or brief.cta_instruction,
        rewritten_copy=brief.rewritten_copy,
        cta_instruction=brief.cta_instruction,
        visual_instruction=brief.visual_instruction,
        before_score=before_score,
        after_score=after_score,
        delta=round(after_score - before_score, 1),
        accepted=accepted,
        image_url=image_url,
        demographic_focus=brief.demographic_focus,
        explanation=brief.explanation,
    )


def _image_fingerprint(image_png: bytes) -> str:
    return hashlib.sha256(image_png).hexdigest()


def _source_type(req: OptimizeRequest) -> str:
    if req.url:
        return "url"
    if req.image_base64:
        return "image"
    return "html"


async def _capture_request(req: OptimizeRequest):
    if req.url:
        return await capture_url(req.url)
    if req.image_base64:
        return await capture_image(req.image_base64, req.image_name)
    return await capture_html(req.html or "")


async def run_job(req: OptimizeRequest, job: Job) -> None:
    try:
        artifact_root = Path(os.getenv("FIXATE_JOBS_DIR", tempfile.gettempdir())) / "fixate_jobs"
        job.artifact_dir = artifact_root / job.job_id
        job.artifact_dir.mkdir(parents=True, exist_ok=True)

        source_type = _source_type(req)
        await emit(job, "capture_started", {"source": source_type})
        capture = await _capture_request(req)
        (job.artifact_dir / "screenshot.png").write_bytes(capture.screenshot_png)
        await emit(
            job,
            "capture_done",
            {
                "width": capture.width,
                "height": capture.height,
                "text_chars": len(capture.text),
                "image_url": f"/job/{job.job_id}/image",
            },
        )

        attention = await predict_saliency_openai(capture.screenshot_png, capture.text, capture.element_boxes)
        heatmap_png = render_heatmap_overlay(capture.screenshot_png, attention.saliency_map, attention.regions)
        (job.artifact_dir / "heatmap.png").write_bytes(heatmap_png)
        job.attention_regions = list(attention.regions)
        job.heatmap_text = capture.text
        job.region_explanations.clear()
        await emit(
            job,
            "heatmap_ready",
            {
                "attention_model": "openai_vision" if attention.live else "local_fallback",
                "regions": [r.model_dump() for r in attention.regions],
                "scan_path_count": len(attention.scan_path),
                "image_width": capture.width,
                "image_height": capture.height,
                "heatmap_bytes": len(heatmap_png),
                "heatmap_url": f"/job/{job.job_id}/heatmap",
                "live": attention.live,
            },
        )

        current_image = capture.screenshot_png
        current_text = capture.text
        current_attention = attention
        current, score_live = await score_regions(current_image, current_attention.saliency_map, current_text)
        await emit(job, "scored", {"fixate_score": current.fixate_score, "blockers": current.blockers, "live": score_live})

        demographic_segments: list[DemographicSegment] = []
        selected_demographic: DemographicSegment | None = None
        demographics_live = False
        if req.auto_find_demographics:
            await emit(job, "demographics_started", {"target_customer_hint": req.target_customer})
            demographic_segments, demographics_live = await discover_demographics(
                current_image,
                current_text,
                req.target_customer,
                req.goal,
            )
            selected_demographic = select_demographic(
                demographic_segments,
                req.demographic_target,
                req.target_customer,
            )
            await emit(
                job,
                "demographics_ready",
                {
                    "segments": [segment.model_dump() for segment in demographic_segments],
                    "selected": selected_demographic.model_dump(),
                    "live": demographics_live,
                },
            )
        elif req.demographic_target:
            selected_demographic = select_demographic([], req.demographic_target, req.target_customer)

        baseline = current
        effective_target = selected_demographic.name if selected_demographic else req.target_customer
        reactions, reactions_live = await run_buyer_panel(current, current_text, effective_target, selected_demographic)
        await emit(job, "buyer_panel", {"reactions": [r.model_dump() for r in reactions], "live": reactions_live})
        diagnosis, diagnosis_live = await diagnose(current, reactions)
        await emit(job, "diagnosis_ready", {"diagnosis": diagnosis.model_dump(), "live": diagnosis_live})

        variants: list[VariantResult] = []
        blocked_edits: list[BlockedEdit] = []
        best_variant: VariantResult | None = None
        for iteration in range(1, req.iterations + 1):
            active_blockers = [r.blocker for r in reactions if r.severity in {"medium", "high"} and r.blocker != "none"]
            blocker = active_blockers[0] if active_blockers else (current.blockers[0] if current.blockers else "general_clarity")
            await emit(job, "blocker_found", {"iteration": iteration, "blocker": blocker})
            diagnosis_for_iteration = diagnosis.model_copy(deep=True)
            diagnosis_for_iteration.hurting_conversion = [blocker] + [
                b for b in diagnosis_for_iteration.hurting_conversion if b != blocker
            ]
            brief, brief_live = await generate_variant_brief(
                diagnosis_for_iteration,
                req.constraints,
                current_text,
                req.target_customer,
                req.goal,
                iteration,
                selected_demographic,
            )
            brief = _maybe_force_demo_violation(brief, req, iteration)
            await emit(job, "variant_proposed", {"iteration": iteration, "variant": brief.model_dump(), "live": brief_live})

            before_fingerprint = _image_fingerprint(current_image)
            candidate_image, edit_live, edit_error = await apply_edits(current_image, brief, req.constraints)
            if candidate_image is None:
                if openai_required():
                    raise RuntimeError(edit_error or "OpenAI image generation returned no image.")
                await emit(
                    job,
                    "variant_image_failed",
                    {
                        "iteration": iteration,
                        "variant_id": brief.id,
                        "reason": edit_error or "Image generation returned no image.",
                        "live": False,
                    },
                )
                await emit(job, "iteration_done", {"iteration": iteration, "accepted": False, "image_failed": True})
                continue
            image_changed = _image_fingerprint(candidate_image) != before_fingerprint
            if not image_changed:
                await emit(
                    job,
                    "variant_image_failed",
                    {
                        "iteration": iteration,
                        "variant_id": brief.id,
                        "reason": "Image generation returned an unchanged image.",
                        "live": edit_live,
                    },
                )
                await emit(job, "iteration_done", {"iteration": iteration, "accepted": False, "image_failed": True})
                continue
            candidate_text = _updated_text(current_text, brief)
            candidate_attention = await predict_saliency_openai(candidate_image, candidate_text)
            candidate_score, candidate_score_live = await score_regions(
                candidate_image,
                candidate_attention.saliency_map,
                candidate_text,
            )
            delta = round(candidate_score.fixate_score - current.fixate_score, 1)

            violates, reason = violates_constraints(brief, req.constraints)
            if violates:
                blocked = BlockedEdit(
                    blocker=brief.target_blocker,
                    reason=reason,
                    estimated_gain=max(0.0, delta),
                    variant=brief,
                )
                blocked_edits.append(blocked)
                await emit(
                    job,
                    "edit_blocked",
                    {
                        "iteration": iteration,
                        "blocker": brief.target_blocker,
                        "reason": reason,
                        "estimated_gain": blocked.estimated_gain,
                        "variant": brief.model_dump(),
                    },
                )
                await emit(job, "iteration_done", {"iteration": iteration, "accepted": False, "blocked": True})
                continue

            variant_path = job.artifact_dir / f"variant_{iteration}.png"
            variant_path.write_bytes(candidate_image)
            image_url = f"/job/{job.job_id}/variant/{iteration}/image"
            await emit(
                job,
                "variant_applied",
                {
                    "iteration": iteration,
                    "variant_id": brief.id,
                    "image_url": image_url,
                    "live": edit_live,
                    "image_changed": image_changed,
                    "fallback_reason": edit_error if not edit_live else None,
                },
            )

            accepted = delta > 0
            await emit(
                job,
                "variant_scored",
                {
                    "iteration": iteration,
                    "variant_id": brief.id,
                    "fixate_score": candidate_score.fixate_score,
                    "delta": delta,
                    "accepted": accepted,
                    "live": candidate_score_live,
                },
            )
            variant_record = _variant_record(
                brief,
                before_score=current.fixate_score,
                after_score=candidate_score.fixate_score,
                accepted=accepted,
                image_url=image_url,
            )
            variants.append(variant_record)
            if accepted:
                current = candidate_score
                current_image = candidate_image
                current_text = candidate_text
                current_attention = candidate_attention
                best_variant = variant_record
                (job.artifact_dir / "best.png").write_bytes(candidate_image)
                heatmap_png = render_heatmap_overlay(current_image, current_attention.saliency_map, current_attention.regions)
                (job.artifact_dir / "heatmap.png").write_bytes(heatmap_png)
                job.attention_regions = list(current_attention.regions)
                job.heatmap_text = current_text
                job.region_explanations.clear()
                heatmap_height, heatmap_width = current_attention.saliency_map.shape
                await emit(
                    job,
                    "heatmap_ready",
                    {
                        "iteration": iteration,
                        "attention_model": "openai_vision" if current_attention.live else "local_fallback",
                        "regions": [r.model_dump() for r in current_attention.regions],
                        "scan_path_count": len(current_attention.scan_path),
                        "image_width": heatmap_width,
                        "image_height": heatmap_height,
                        "heatmap_bytes": len(heatmap_png),
                        "heatmap_url": f"/job/{job.job_id}/heatmap?v={iteration}",
                        "live": current_attention.live,
                    },
                )
                reactions, reactions_live = await run_buyer_panel(current, current_text, effective_target, selected_demographic)
                diagnosis, diagnosis_live = await diagnose(current, reactions)
                await emit(job, "buyer_panel", {"iteration": iteration, "reactions": [r.model_dump() for r in reactions], "live": reactions_live})
                await emit(job, "diagnosis_ready", {"iteration": iteration, "diagnosis": diagnosis.model_dump(), "live": diagnosis_live})
            await emit(job, "iteration_done", {"iteration": iteration, "accepted": accepted})

        experiment_plan, experiment_plan_live = await build_experiment_plan(
            best_variant,
            req.target_customer,
            req.goal,
            selected_demographic,
        )
        job.result = {
            "job_id": job.job_id,
            "source_type": source_type,
            "image_url": f"/job/{job.job_id}/image",
            "heatmap_url": f"/job/{job.job_id}/heatmap",
            "best_image_url": f"/job/{job.job_id}/best-image" if (job.artifact_dir / "best.png").exists() else None,
            "baseline": baseline.model_dump(),
            "final": current.model_dump(),
            "buyer_reactions": [r.model_dump() for r in reactions],
            "diagnosis": diagnosis.model_dump(),
            "best_variant": best_variant.model_dump() if best_variant else None,
            "variants": [v.model_dump() for v in variants],
            "blocked_edits": [b.model_dump() for b in blocked_edits],
            "experiment_plan": experiment_plan.model_dump(),
            "constraints": req.constraints.model_dump(),
            "target_customer": req.target_customer,
            "demographics": [segment.model_dump() for segment in demographic_segments],
            "selected_demographic": selected_demographic.model_dump() if selected_demographic else None,
            "goal": req.goal,
        }
        job.status = "complete"
        await emit(
            job,
            "job_complete",
            {
                "baseline_score": baseline.fixate_score,
                "final_score": current.fixate_score,
                "delta": round(current.fixate_score - baseline.fixate_score, 1),
                "live": experiment_plan_live,
            },
        )
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        await emit(job, "job_error", {"message": str(exc)})


def create_job(req: OptimizeRequest) -> Job:
    job = Job(job_id=str(uuid.uuid4()))
    jobs[job.job_id] = job
    asyncio.create_task(run_job(req, job))
    return job
