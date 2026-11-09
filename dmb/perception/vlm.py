"""Edge-VLM baseline: a small vision model as the detector.

Same trigger/budget interface as the rule stub (`StubEdgePerception`),
but detections come from a VLM reading the observation frame instead of
sim ground truth. The model follows a strict text protocol; the parser
is total (malformed lines are dropped, never crash).

Live use needs a vision-capable OpenAI-compatible endpoint
(DMB_VLM_BASE_URL / DMB_VLM_API_KEY / DMB_VLM_MODEL). Unit tests inject
a stub completer — no network.
"""

from __future__ import annotations

from typing import Callable

from dmb.perception.edge import Detection, StubEdgePerception

VLM_PROMPT = """You are an onboard drone detector. The image is a sensor view:
each white dot is a victim contact; dot size hints category (small=child,
medium=human, varied=animal). The grid origin (0,0) is top-left; each cell
is 48 pixels. List EVERY contact you see, one per line, as:
category,x,y
category is human, child, or animal; x,y are grid integers. No other text."""


def parse_detections(text: str) -> list[Detection]:
    """Parse `category,x,y[,confidence]` lines; drop malformed ones."""
    out: list[Detection] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or parts[0] not in ("human", "child", "animal"):
            continue
        try:
            x, y = int(parts[1]), int(parts[2])
            conf = float(parts[3]) if len(parts) > 3 else 0.5
        except ValueError:
            continue
        out.append(Detection(label=parts[0], x=x, y=y, confidence=conf))
    return out


Completer = Callable[[str, str], str]  # (image_data_url, prompt) -> text


def vlm_from_env() -> Completer:
    """Build the default completer from DMB_VLM_BASE_URL/_API_KEY/_MODEL."""
    import os
    from functools import partial

    base_url = os.environ.get("DMB_VLM_BASE_URL", "").strip()
    api_key = os.environ.get("DMB_VLM_API_KEY", "").strip()
    model = os.environ.get("DMB_VLM_MODEL", "").strip()
    if not (base_url and api_key and model):
        raise RuntimeError(
            "Set DMB_VLM_BASE_URL, DMB_VLM_API_KEY, and DMB_VLM_MODEL "
            "for live VLM perception (or inject a stub completer)."
        )
    return partial(openai_completer, model=model, base_url=base_url, api_key=api_key)


def openai_completer(image_data_url: str, prompt: str, *, model: str, base_url: str, api_key: str) -> str:
    """Default completer over any OpenAI-compatible vision endpoint."""
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


class VLMEdgePerception(StubEdgePerception):
    """Trigger logic inherited; detection delegated to a VLM + frame."""

    def __init__(self, bridge, config, budget, completer: Completer) -> None:
        super().__init__(bridge, config, budget)
        self.completer = completer

    def _scan(self) -> tuple[list[Detection], list]:
        from dmb.perception.frames import render_sensor_frame, to_data_url  # noqa: PLC0415

        state = self.bridge.get_state()
        grid = {"width": self.bridge.width, "height": self.bridge.height}
        png = render_sensor_frame(state, grid, self.config.detection_radius)
        try:
            text = self.completer(to_data_url(png), VLM_PROMPT)
        except Exception:
            return [], []
        detections = parse_detections(text)
        novel = []
        for det in detections:
            match = self._match_victim(det)
            if match is not None and id(match) not in self._known_ids:
                self._known_ids.add(id(match))
                novel.append(match)
                self.bridge.mark_found(match)
        return detections, novel

    def _match_victim(self, det: Detection):
        """Ground a reported detection to the nearest true victim (±1 cell)."""
        best, best_dist = None, None
        for victim in self.bridge.victims:
            if victim.category != det.label:
                continue
            dist = abs(victim.x - det.x), abs(victim.y - det.y)
            if dist[0] > 1 or dist[1] > 1:
                continue
            manhattan = dist[0] + dist[1]
            if best is None or manhattan < best_dist:
                best, best_dist = victim, manhattan
        return best
