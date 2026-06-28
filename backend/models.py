from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


BuyerSignal = Literal[
    "attention",
    "desire",
    "trust",
    "memory",
    "cognitive_load",
    "self_relevance",
    "cta_strength",
]

ZoneType = Literal["power_zone", "attention_trap", "hidden_value", "dead_zone"]


class ElementBox(BaseModel):
    tag: str
    bbox: list[int] = Field(min_length=4, max_length=4)


class CaptureResult(BaseModel):
    screenshot_png: bytes
    text: str
    width: int
    height: int
    element_boxes: list[ElementBox] = Field(default_factory=list)


class CaptureRequest(BaseModel):
    url: str | None = None
    html: str | None = None
    image_base64: str | None = None
    image_name: str | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> "CaptureRequest":
        sources = [bool(self.url), bool(self.html), bool(self.image_base64)]
        if sum(sources) != 1:
            raise ValueError("Provide exactly one of url, html, or image_base64.")
        return self


class CaptureResponse(BaseModel):
    capture_id: str
    text: str
    width: int
    height: int
    image_url: str


class FixationRegion(BaseModel):
    rank: int
    bbox: list[int] = Field(min_length=4, max_length=4)
    saliency_score: float
    peak_coords: list[int] = Field(min_length=2, max_length=2)
    reason: str = ""


class RegionScore(BaseModel):
    bbox: list[int] = Field(min_length=4, max_length=4)
    saliency: float
    signals: dict[BuyerSignal, float]
    zone: ZoneType


class ScoreResult(BaseModel):
    fixate_score: float
    signal_scores: dict[BuyerSignal, float]
    regions: list[RegionScore]
    blockers: list[str]


Aggressiveness = Literal["conservative", "balanced", "aggressive"]


class BrandConstraints(BaseModel):
    colors: list[str] = Field(default_factory=list)
    fonts: list[str] = Field(default_factory=list)
    tone: str = ""
    logo_present: bool = False


class LockedElement(BaseModel):
    type: str
    bbox: list[int] | None = None
    value: str | None = None


class Constraints(BaseModel):
    brand: BrandConstraints = Field(default_factory=BrandConstraints)
    locked_elements: list[LockedElement] = Field(default_factory=list)
    aggressiveness: Aggressiveness = "balanced"


class DemographicSegment(BaseModel):
    id: str
    name: str
    summary: str
    messaging_angle: str
    visual_direction: str
    recommended_channel: str
    why_it_fits: str


class OptimizeRequest(BaseModel):
    url: str | None = None
    html: str | None = None
    image_base64: str | None = None
    image_name: str | None = None
    target_customer: str = "busy small-business buyer"
    goal: str = "increase clicks"
    iterations: int = Field(default=2, ge=1, le=10)
    constraints: Constraints = Field(default_factory=Constraints)
    demographic_target: str | None = None
    auto_find_demographics: bool = True

    @model_validator(mode="after")
    def require_one_source(self) -> "OptimizeRequest":
        sources = [bool(self.url), bool(self.html), bool(self.image_base64)]
        if sum(sources) != 1:
            raise ValueError("Provide exactly one of url, html, or image_base64.")
        return self


class BuyerReaction(BaseModel):
    dimension: str
    severity: Literal["low", "medium", "high"]
    blocker: str
    explanation: str


class Diagnosis(BaseModel):
    working: list[str]
    ignored: list[str]
    hurting_conversion: list[str]
    summary: str


class VariantBrief(BaseModel):
    id: str
    target_blocker: str
    rewritten_copy: str
    cta_instruction: str
    visual_instruction: str
    layout_instruction: str = ""
    demographic_focus: str = ""
    color: str | None = None
    font: str | None = None
    touches_locked_element: str | None = None
    explanation: str = ""


class BlockedEdit(BaseModel):
    blocker: str
    reason: str
    estimated_gain: float
    variant: VariantBrief | dict


class VariantResult(BaseModel):
    id: str
    target_blocker: str
    description: str
    rewritten_copy: str
    cta_instruction: str
    visual_instruction: str
    before_score: float
    after_score: float
    delta: float
    accepted: bool
    image_url: str | None = None
    demographic_focus: str = ""
    explanation: str = ""


class ExperimentPlan(BaseModel):
    hypothesis: str
    recommended_channel: str
    target_audience: str
    success_metric: str
    ab_test_setup: str
    next_step: str
