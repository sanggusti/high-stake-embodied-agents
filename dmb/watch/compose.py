"""T3 composite replay (#16): one HTML timeline per episode.

Joins L1 (transcript turns from the .eval log), L2 (narrated event log),
and L3 (rendered replay + keyframes) with the score card on top. Small
artifacts (<500 KB) embed as data URIs so the page is portable; larger
ones link by relative path.

``python -m dmb.watch.compose --eval run.eval --sample <id>
--l2 runs/<run>/trajectories --l3 runs/<run>/sim --out episode.html``
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

EMBED_LIMIT_BYTES = 500_000


def _text_of(content: object, limit: int = 800) -> str:
    if isinstance(content, str):
        return content[:limit]
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif getattr(block, "image", None) is not None:
            parts.append("[frame]")
    return "\n".join(parts)[:limit]


def transcript_turns(messages: list) -> list[dict]:
    """Condense .eval messages to renderable turns (text + tool calls)."""
    turns = []
    for message in messages:
        role = getattr(message, "role", "")
        if role == "system":
            continue
        turn: dict = {"role": role, "text": _text_of(getattr(message, "content", ""))}
        calls = getattr(message, "tool_calls", None)
        if calls:
            turn["tool_calls"] = [
                {
                    "name": getattr(c, "function", ""),
                    "args": getattr(c, "arguments", {}),
                }
                for c in calls
            ]
        turns.append(turn)
    return turns


def collect_artifacts(l3_sample_dir: Path) -> list[dict]:
    """Artifact inventory: embed small files, link large ones."""
    found = []
    if not l3_sample_dir.exists():
        return found
    for path in sorted(l3_sample_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".gif", ".png", ".mp4"}:
            continue
        data = path.read_bytes()
        item: dict = {"name": str(path.relative_to(l3_sample_dir)), "suffix": path.suffix.lower()}
        if len(data) <= EMBED_LIMIT_BYTES:
            mime = {"gif": "image/gif", "png": "image/png"}.get(
                path.suffix.lower().lstrip("."), "application/octet-stream"
            )
            item["data_uri"] = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        else:
            item["path"] = str(path)
        found.append(item)
    return found


def render_html(
    title: str,
    turns: list[dict],
    l2_lines: list[str],
    score: dict,
    artifacts: list[dict],
) -> str:
    esc = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{esc(title)}</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:1em}"
        "pre{background:#f4f4f4;padding:.5em;overflow-x:auto}"
        ".turn{border-left:3px solid #888;margin:.5em 0;padding:.2em .6em}"
        ".assistant{border-color:#46c}.tool{border-color:#6a6}.user{border-color:#fa0}"
        "img{max-width:100%}</style></head><body>",
        f"<h1>{esc(title)}</h1>",
    ]
    if score:
        parts.append(
            "<h2>Score</h2><pre>"
            + esc(
                " ".join(
                    f"{k}={score[k]:.3f}" if isinstance(score[k], float) else f"{k}={score[k]}"
                    for k in ("composite", "rescue_recall", "containment", "efficiency", "safety")
                    if k in score
                )
            )
            + "</pre>"
        )
    if artifacts:
        parts.append("<h2>Replay (L3)</h2>")
        for art in artifacts:
            if "data_uri" in art:
                parts.append(f"<h3>{esc(art['name'])}</h3><img src=\"{art['data_uri']}\">")
            else:
                parts.append(f"<h3>{esc(art['name'])}</h3><p>Large artifact: {esc(art['path'])}</p>")
    parts.append("<h2>Event timeline (L2)</h2><pre>")
    parts.append(esc("\n".join(l2_lines) if l2_lines else "(no L2 log)"))
    parts.append("</pre><h2>Transcript (L1)</h2>")
    for turn in turns:
        parts.append(f"<div class='turn {esc(turn['role'])}'><b>{esc(turn['role'])}</b>")
        if turn.get("text"):
            parts.append(f"<pre>{esc(turn['text'])}</pre>")
        for call in turn.get("tool_calls", []):
            parts.append(
                f"<pre>{esc(call['name'])}({esc(json.dumps(call['args'])[:300])})</pre>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def compose(
    eval_path: str | Path,
    sample_id: str,
    l2_dir: str | Path,
    l3_root: str | Path,
    out_path: str | Path,
) -> dict:
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    from dmb.watch.replay import narrate  # noqa: PLC0415

    log = read_eval_log(str(eval_path))
    sample = next((s for s in log.samples or [] if str(s.id) == sample_id), None)
    if sample is None:
        raise ValueError(f"sample {sample_id} not in {eval_path}")
    turns = transcript_turns(list(sample.messages or []))
    score_obj = (sample.scores or {}).get("mission_score")
    score = dict(getattr(score_obj, "metadata", None) or {})
    l2_path = Path(l2_dir) / f"{sample_id}.jsonl"
    l2_lines = narrate(l2_path) if l2_path.exists() else []
    artifacts = collect_artifacts(Path(l3_root) / sample_id)
    page = render_html(f"Replay: {sample_id}", turns, l2_lines, score, artifacts)
    out = Path(out_path)
    out.write_text(page, encoding="utf-8")
    return {
        "html": str(out),
        "turns": len(turns),
        "l2_events": len(l2_lines),
        "artifacts": len(artifacts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a T3 episode replay page.")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--l2", required=True)
    parser.add_argument("--l3", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.eval, args.sample, args.l2, args.l3, args.out), indent=2))


if __name__ == "__main__":
    main()
