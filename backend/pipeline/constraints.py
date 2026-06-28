from __future__ import annotations

import re

from models import Constraints, VariantBrief


HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def normalise_hex(value: str) -> str:
    value = value.strip()
    return value.upper() if HEX_RE.match(value) else value


def violates_constraints(proposed_edit: VariantBrief, constraints: Constraints) -> tuple[bool, str]:
    locked = constraints.locked_elements

    if proposed_edit.color:
        allowed = {normalise_hex(c) for c in constraints.brand.colors}
        proposed = normalise_hex(proposed_edit.color)
        if allowed and proposed not in allowed:
            return True, f"Uses color {proposed} outside the allowed brand palette."

    if proposed_edit.font:
        allowed_fonts = {f.strip().lower() for f in constraints.brand.fonts if f.strip()}
        if allowed_fonts and proposed_edit.font.strip().lower() not in allowed_fonts:
            return True, f"Uses font {proposed_edit.font!r}, which is not in the allowed font list."

    if any(el.type == "layout" and el.value == "fixed" for el in locked):
        instruction = f"{proposed_edit.layout_instruction} {proposed_edit.cta_instruction}".lower()
        if any(term in instruction for term in ("move", "reposition", "elevate", "above", "layout")):
            return True, "Layout is locked, so moving or repositioning elements is not allowed."

    if proposed_edit.touches_locked_element:
        touched = proposed_edit.touches_locked_element.lower()
        for el in locked:
            if el.type.lower() == touched:
                return True, f"Edit touches locked element: {el.type}."

    if constraints.aggressiveness == "conservative":
        visual = proposed_edit.visual_instruction.lower()
        layout = proposed_edit.layout_instruction.lower()
        if visual and any(term in visual for term in ("background", "remove", "color", "image", "visual")):
            return True, "Conservative mode allows copy and CTA tweaks only, not visual edits."
        if layout and any(term in layout for term in ("move", "reposition", "layout")):
            return True, "Conservative mode allows copy and CTA tweaks only, not layout moves."

    if constraints.aggressiveness == "balanced":
        visual = proposed_edit.visual_instruction.lower()
        if any(term in visual for term in ("replace background", "remove object", "generative fill")):
            return True, "Balanced mode allows copy and layout changes, but not heavy visual/background edits."

    return False, ""

