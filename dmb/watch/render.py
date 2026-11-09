"""T2 mock replay: render an L2 JSONL trajectory to GIF + final PNG.

World layout is rebuilt deterministically from (scenario, seed) and
verified against the logged world_hash; drone path + detections come from
the enriched L2 events (pos on actions, detections on edge events).

Post-hoc audit tool: replays reveal hidden victim positions. Never feed
rendered frames back as model input.

``python -m dmb.watch.render <sample.jsonl> --out runs/<id>/replays/``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

CELL = 48
MARGIN = 8
HEADER = 28
BG = (16, 20, 28)
GRID = (48, 56, 72)
HAZARD = (150, 40, 40)
VICTIM_HIDDEN = (200, 170, 60)
VICTIM_FOUND = (70, 190, 90)
DRONE = (90, 160, 255)
PATH = (90, 160, 255)
FLEET_COLORS = [
    (90, 160, 255),
    (255, 170, 60),
    (110, 220, 130),
    (200, 130, 255),
]


def load_world(scenario: str, seed: int, world_hash: str) -> dict:
    from dmb.scenarios import build_samples  # noqa: PLC0415

    meta = build_samples(scenario, seeds=[seed])[0].metadata
    if meta["world_hash"] != world_hash:
        raise ValueError(
            f"world_hash mismatch for {scenario} seed {seed}: log has "
            f"{world_hash}, rebuild gives {meta['world_hash']}"
        )
    return meta


def draw_frame(meta: dict, trails: dict, found: set, caption: str) -> Image.Image:
    grid = meta["grid"]
    w, h = int(grid["width"]), int(grid["height"])
    img = Image.new("RGB", (w * CELL + 2 * MARGIN, h * CELL + 2 * MARGIN + HEADER), BG)
    d = ImageDraw.Draw(img)

    def cell_xy(x: int, y: int) -> tuple[int, int, int, int]:
        return (
            MARGIN + x * CELL,
            MARGIN + HEADER + y * CELL,
            MARGIN + (x + 1) * CELL,
            MARGIN + HEADER + (y + 1) * CELL,
        )

    for x in range(w):
        for y in range(h):
            d.rectangle(cell_xy(x, y), outline=GRID)
    for hz in meta.get("hazards", []):
        hx, hy, r = int(hz["x"]), int(hz["y"]), float(hz.get("radius", 1.0))
        d.ellipse(
            [
                MARGIN + (hx - r + 0.5) * CELL,
                MARGIN + HEADER + (hy - r + 0.5) * CELL,
                MARGIN + (hx + r + 0.5) * CELL,
                MARGIN + HEADER + (hy + r + 0.5) * CELL,
            ],
            fill=HAZARD,
        )
    for v in meta.get("victim_placements", []):
        x0, y0, x1, y1 = cell_xy(int(v["x"]), int(v["y"]))
        color = VICTIM_FOUND if (v["x"], v["y"]) in found else VICTIM_HIDDEN
        d.ellipse([x0 + 12, y0 + 12, x1 - 12, y1 - 12], fill=color)
    for idx in sorted(trails):
        trail = trails[idx]
        color = FLEET_COLORS[idx % len(FLEET_COLORS)]
        if len(trail) > 1:
            d.line(
                [
                    (MARGIN + (x + 0.5) * CELL, MARGIN + HEADER + (y + 0.5) * CELL)
                    for x, y in trail
                ],
                fill=color,
                width=3,
            )
        if trail:
            x, y = trail[-1]
            x0, y0, x1, y1 = cell_xy(x, y)
            d.rectangle([x0 + 6, y0 + 6, x1 - 6, y1 - 6], outline=color, width=3)
    d.text((MARGIN, 6), caption, fill=(220, 220, 220))
    return img


def render(path: str | Path, out_dir: str | Path) -> dict:
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    start = next(e for e in events if e.get("event") == "episode_start")
    meta = load_world(start["scenario"], int(start["seed"]), start["world_hash"])
    score = next((e for e in events if e["kind"] == "score"), {})

    grid = meta["grid"]
    gw, gh = int(grid["width"]), int(grid["height"])
    pads = [(0, 0), (gw - 1, 0), (0, gh - 1), (gw - 1, gh - 1)]
    fleet_size = int(start.get("fleet_size", 1) or 1)
    trails: dict[int, list[tuple[int, int]]] = {
        i: [pads[i % len(pads)]] for i in range(fleet_size)
    }
    found: set[tuple[int, int]] = set()
    frames: list[Image.Image] = []
    for e in events:
        if e["kind"] == "edge":
            for det in e.get("detections", []) or []:
                found.add((det["x"], det["y"]))
        elif e["kind"] == "action":
            pos = e.get("pos")
            if pos is None:
                raise ValueError(
                    "L2 predates pos enrichment; re-run the episode to replay it"
                )
            idx = int((e.get("args") or {}).get("drone_idx", 0))
            trails.setdefault(idx, [pads[idx % len(pads)]]).append(
                (int(pos[0]), int(pos[1]))
            )
            args = ",".join(f"{k}={v}" for k, v in (e.get("args") or {}).items())
            frames.append(
                draw_frame(
                    meta, trails, found,
                    f"t={e['t']:.0f}s step={e['step']} {e.get('tool')}({args})",
                )
            )
    if not frames:
        frames.append(draw_frame(meta, trails, found, "no actions"))
    frames.append(
        draw_frame(
            meta, trails, found,
            f"done: composite={score.get('composite', 0):.3f} "
            f"found={len(found)}/{len(meta.get('victim_placements', []))}",
        )
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    gif_path = out / f"{stem}.gif"
    png_path = out / f"{stem}_final.png"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
        duration=600, loop=0,
    )
    frames[-1].save(png_path)
    return {"gif": str(gif_path), "png": str(png_path), "frames": len(frames)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an L2 trajectory to GIF+PNG.")
    parser.add_argument("path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.path, args.out), indent=2))


if __name__ == "__main__":
    main()
