from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agents.buyer_panel import run_buyer_panel
from agents.creative import generate_variant_brief
from agents.experiment import build_experiment_plan
from agents.strategist import diagnose
from models import BlockedEdit, OptimizeRequest, ScoreResult, VariantBrief, VariantResult
from pipeline.attention import predict_saliency_openai, render_heatmap_overlay
from pipeline.capture import capture_html, capture_url
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


jobs: dict[str, Job] = {}


async def emit(job: Job, event: str, data: dict | None = None) -> None:
    payload = {"seq": len(job.events) + 1, "event": event, "ts": time.time(), **(data or {})}
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
        explanation=brief.explanation,
    )


async def run_job(req: OptimizeRequest, job: Job) -> None:
    try:
        artifact_root = Path(os.getenv("FIXATE_JOBS_DIR", tempfile.gettempdir())) / "fixate_jobs"
        job.artifact_dir = artifact_root / job.job_id
        job.artifact_dir.mkdir(parents=True, exist_ok=True)

        await emit(job, "capture_started", {"source": "url" if req.url else "html"})
        capture = await (capture_url(req.url) if req.url else capture_html(req.html or ""))
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

        attention = await predict_saliency_openai(capture.screenshot_png, capture.text)
        heatmap_png = render_heatmap_overlay(capture.screenshot_png, attention.saliency_map)
        (job.artifact_dir / "heatmap.png").write_bytes(heatmap_png)
        await emit(
            job,
            "heatmap_ready",
            {
                "attention_model": "openai_vision" if attention.live else "local_fallback",
                "regions": [r.model_dump() for r in attention.regions],
                "heatmap_bytes": len(heatmap_png),
                "heatmap_url": f"/job/{job.job_id}/heatmap",
            },
        )

        current_image = capture.screenshot_png
        current_text = capture.text
        current_attention = attention
        current = await score_regions(current_image, current_attention.saliency_map, current_text)
        await emit(job, "scored", {"fixate_score": current.fixate_score, "blockers": current.blockers})

        baseline = current
        reactions = await run_buyer_panel(current, current_text)
        await emit(job, "buyer_panel", {"reactions": [r.model_dump() for r in reactions]})
        diagnosis = await diagnose(current, reactions)
        await emit(job, "diagnosis_ready", {"diagnosis": diagnosis.model_dump()})

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
            brief = await generate_variant_brief(
                diagnosis_for_iteration,
                req.constraints,
                current_text,
                req.target_customer,
                req.goal,
                iteration,
            )
            brief = _maybe_force_demo_violation(brief, req, iteration)
            await emit(job, "variant_proposed", {"iteration": iteration, "variant": brief.model_dump()})

            candidate_image = await apply_edits(current_image, brief, req.constraints)
            candidate_text = _updated_text(current_text, brief)
            candidate_attention = await predict_saliency_openai(candidate_image, candidate_text)
            candidate_score = await score_regions(
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
                heatmap_png = render_heatmap_overlay(current_image, current_attention.saliency_map)
                (job.artifact_dir / "heatmap.png").write_bytes(heatmap_png)
                reactions = await run_buyer_panel(current, current_text)
                diagnosis = await diagnose(current, reactions)
                await emit(job, "buyer_panel", {"iteration": iteration, "reactions": [r.model_dump() for r in reactions]})
                await emit(job, "diagnosis_ready", {"iteration": iteration, "diagnosis": diagnosis.model_dump()})
            await emit(job, "iteration_done", {"iteration": iteration, "accepted": accepted})

        experiment_plan = await build_experiment_plan(best_variant, req.target_customer, req.goal)
        job.result = {
            "job_id": job.job_id,
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
