"""Export teacher runs into post-training datasets.

Inputs: an inspect-ai ``.eval`` log (L1 transcripts + scores) and the run's
L2 trajectory dir. Outputs, under ``out_dir``:

- ``sft.jsonl`` — success-filtered chat pairs: each assistant turn that made
  tool calls becomes one sample (conversation prefix + that turn, CoT
  included). Images are replaced with a ``[frame omitted]`` marker.
- ``rlvr.jsonl`` — one record per episode: sample id, scenario, seed, the
  full metric reward vector, steps, tool calls.
- ``manifest.json`` — splits, filters, teacher, code commit, file hashes,
  world hashes. The determinism guarantee for the published dataset.

``python -m dmb.data.export --log run.eval --l2 runs/<id>/trajectories
--out data/v1 --split train --min-composite 0.5``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from dmb.data.splits import split_for_seed


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif getattr(block, "image", None) is not None:
            parts.append("[frame omitted]")
    return "\n".join(parts)


def _assistant_turn(message: object) -> dict:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append(
            {
                "id": getattr(call, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(call, "function", ""),
                    "arguments": json.dumps(getattr(call, "arguments", {})),
                },
            }
        )
    turn: dict = {"role": "assistant", "content": _text_of(getattr(message, "content", ""))}
    if calls:
        turn["tool_calls"] = calls
    return turn


def messages_to_sft_pairs(messages: list) -> list[list[dict]]:
    """One SFT sample per assistant turn that made tool calls.

    Each sample is the full conversation prefix (system/user/assistant/tool
    roles, OpenAI chat shape) ending with that assistant turn.
    """
    pairs: list[list[dict]] = []
    prefix: list[dict] = []
    for message in messages:
        role = getattr(message, "role", "")
        if role == "system":
            prefix.append({"role": "system", "content": _text_of(message.content)})
        elif role == "user":
            prefix.append({"role": "user", "content": _text_of(message.content)})
        elif role == "assistant":
            turn = _assistant_turn(message)
            prefix.append(turn)
            if turn.get("tool_calls"):
                pairs.append([dict(m) for m in prefix])
        elif role == "tool":
            prefix.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(message, "tool_call_id", ""),
                    "content": _text_of(getattr(message, "content", ""))[:2000],
                }
            )
    return pairs


def export_run(
    log_path: str | Path,
    l2_dir: str | Path,
    out_dir: str | Path,
    *,
    split: str,
    min_composite: float = 0.5,
    teacher: str = "",
) -> dict:
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    log = read_eval_log(str(log_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    l2 = Path(l2_dir)

    sft_path = out / "sft.jsonl"
    rlvr_path = out / "rlvr.jsonl"
    sft_fh = sft_path.open("w", encoding="utf-8")
    rlvr_fh = rlvr_path.open("w", encoding="utf-8")
    kept_episodes = 0
    sft_pairs = 0
    world_hashes: dict[str, str] = {}
    try:
        for sample in log.samples or []:
            meta = dict(sample.metadata or {})
            seed = int(meta.get("seed", -1))
            if split_for_seed(seed) != split:
                raise ValueError(
                    f"sample {sample.id} seed {seed} is not in split {split!r}"
                )
            score = (sample.scores or {}).get("mission_score")
            composite = float(score.value) if score is not None else 0.0
            breakdown = dict(getattr(score, "metadata", None) or {})
            world_hashes[str(sample.id)] = str(meta.get("world_hash", ""))
            rlvr_fh.write(
                json.dumps(
                    {
                        "sample_id": sample.id,
                        "scenario": meta.get("scenario"),
                        "seed": seed,
                        "world_hash": meta.get("world_hash"),
                        "rewards": breakdown,
                        "composite": composite,
                        "l2": str(l2 / f"{sample.id}.jsonl"),
                    }
                )
                + "\n"
            )
            if composite < min_composite:
                continue
            kept_episodes += 1
            for pair in messages_to_sft_pairs(list(sample.messages or [])):
                sft_pairs += 1
                sft_fh.write(
                    json.dumps(
                        {"messages": pair, "sample_id": sample.id, "seed": seed}
                    )
                    + "\n"
                )
    finally:
        sft_fh.close()
        rlvr_fh.close()

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except Exception:  # noqa: BLE001
        commit = "unknown"

    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "teacher": teacher,
        "code_commit": commit,
        "eval_log": str(log_path),
        "split": split,
        "min_composite": min_composite,
        "episodes_total": len(log.samples or []),
        "episodes_kept": kept_episodes,
        "sft_pairs": sft_pairs,
        "files": {
            "sft.jsonl": _sha256(sft_path),
            "rlvr.jsonl": _sha256(rlvr_path),
        },
        "world_hashes": world_hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a teacher run to SFT/RLVR data.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--l2", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--min-composite", type=float, default=0.5)
    parser.add_argument("--teacher", default="")
    args = parser.parse_args()
    manifest = export_run(
        args.log,
        args.l2,
        args.out,
        split=args.split,
        min_composite=args.min_composite,
        teacher=args.teacher,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
