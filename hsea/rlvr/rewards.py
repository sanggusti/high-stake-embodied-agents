"""Verifiable rewards for the issue-74 RLVR/GRPO pilot (Stage 5).

Pure functions (no torch): the Modal GRPO pilot inlines mirrors of
these (the remote has no repo checkout), and the RFT baseline uses
``passes_verifier`` as its filter. Conventions:

- SWE-RL field-match: dispatch completions must name the gold
  priority (P1/P2/P3 or PRIORITY 1/2/3) AND a concrete action verb.
- R1 rule-rewards: no contract footer in dispatch (+0.2), priority
  exact (+0.5/+1.0), never hedged (downgrade words cost).
- Search-R1 masking: tool-output spans never take loss. The pilot
  formats rollouts so ``[tool result]`` turns are appended to the
  PROMPT (masked by default in TRL); ``mask_tool_spans`` builds the
  equivalent loss mask for flattened transcripts.
"""

from __future__ import annotations

import re

_PRIORITY_RE = re.compile(r"\bP\s*([123])\b|\bPRIORITY\s*([123])\b",
                          re.IGNORECASE)

# Concrete dispatch verbs (tasking language); hedges are their absence
# plus explicit downgrade/notify-first phrasing.
ACTION_VERBS = (
    "task", "dispatch", "deploy", "send", "mobilize", "activate",
    "route", "assign", "launch", "scramble",
)
HEDGE_MARKERS = (
    "notify before acting", "notify first", "before acting",
    "consider", "maybe", "perhaps", "when convenient",
    "routine cycle", "revisit", "monitor", "stand by",
)


def priority_of(text: str) -> str | None:
    """Gold/predicted priority letter (P1/P2/P3) or None."""
    match = _PRIORITY_RE.search(text or "")
    if not match:
        return None
    return f"P{match.group(1) or match.group(2)}"


def has_concrete_action(text: str) -> bool:
    """True when a concrete tasking verb is present."""
    lowered = (text or "").lower()
    return any(v in lowered for v in ACTION_VERBS)


def has_footer(text: str) -> bool:
    """True when a contract footer leaked into dispatch text."""
    upper = (text or "").upper()
    return "DECISION:" in upper or "ACTION:" in upper


def is_hedged(text: str) -> bool:
    """True when the dispatch hedges instead of tasking."""
    lowered = (text or "").lower()
    return any(m in lowered for m in HEDGE_MARKERS)


def field_match_reward(completion: str, gold_priority: str) -> float:
    """SWE-RL field-match: 1.0 priority+action, 0.5 priority, else 0."""
    pred = priority_of(completion)
    if pred is None or pred != gold_priority:
        return 0.0
    return 1.0 if has_concrete_action(completion) else 0.5


def rule_reward_dispatch(completion: str, gold_priority: str) -> float:
    """R1 rule-reward: field-match + footer/hedge shaping in [0, 1.2].

    field-match in {0, 0.5, 1.0}; +0.2 no-footer; -0.3 hedge (floors
    the total at 0). Deterministic and total.
    """
    total = field_match_reward(completion, gold_priority)
    if not has_footer(completion):
        total += 0.2
    if is_hedged(completion):
        total -= 0.3
    return max(0.0, round(total, 3))


def passes_verifier(completion: str, gold_priority: str) -> bool:
    """RFT filter: priority exact + concrete action + no footer."""
    return (field_match_reward(completion, gold_priority) >= 1.0
            and not has_footer(completion))


def mask_tool_spans(segments: list[tuple[str, int]]) -> list[int]:
    """Loss mask over flattened transcript segments (Search-R1).

    ``segments`` is ``(kind, n_tokens)`` with kind in ``prompt``,
    ``assistant``, ``tool``. Prompt and tool spans mask to 0 (no
    loss); assistant spans mask to 1. Tool results ride in the prompt
    in the pilot, so this mirror exists for audit and tests.
    """
    mask: list[int] = []
    for kind, n in segments:
        if kind not in ("prompt", "assistant", "tool"):
            raise ValueError(f"unknown segment kind {kind!r}")
        mask.extend([1 if kind == "assistant" else 0] * n)
    return mask
