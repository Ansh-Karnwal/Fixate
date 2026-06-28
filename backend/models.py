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


class CaptureResult(BaseModel):
    screenshot_png: bytes
    text: str
    width: int
    height: int


class CaptureRequest(BaseModel):
    url: str | None = None
    html: str | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> "CaptureRequest":
        if bool(self.url) == bool(self.html):
            raise ValueError("Provide exactly one of url or html.")
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


class OptimizeRequest(BaseModel):
    url: str | None = None
    html: str | None = None
    target_customer: str = "busy small-business buyer"
    goal: str = "increase clicks"
    iterations: int = Field(default=2, ge=1, le=10)
    constraints: Constraints = Field(default_factory=Constraints)

    @model_validator(mode="after")
    def require_one_source(self) -> "OptimizeRequest":
        if bool(self.url) == bool(self.html):
            raise ValueError("Provide exactly one of url or html.")
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
    explanation: str = ""


class ExperimentPlan(BaseModel):
    hypothesis: str
    recommended_channel: str
    target_audience: str
    success_metric: str
    ab_test_setup: str
    next_step: str
