"""Inject verbatim template blocks into docs/judge-playbook.md (#62).

The single source is ``hsea.judge.templates``; the playbook embeds each
block between ``VERBATIM:<name>:START/END`` sentinels. Run after any
template edit, then run ``tests/test_judge_templates.py``::

    .venv/bin/python scripts/judge_playbook_sync.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.judge import templates as T  # noqa: E402

PLAYBOOK = ROOT / "docs" / "judge-playbook.md"

BLOCKS = {
    "OUTPUT_SCHEMA": T.OUTPUT_SCHEMA,
    "PAIRWISE_SCHEMA": T.PAIRWISE_SCHEMA,
    "FOOTAGE_SYSTEM": T.FOOTAGE_SYSTEM,
    "FOOTAGE_RUBRIC": T.FOOTAGE_RUBRIC,
    "FOOTAGE_FEWSHOTS": T.canonical_fewshots("footage"),
    "SAFETY_SYSTEM": T.SAFETY_SYSTEM,
    "SAFETY_RUBRIC": T.SAFETY_RUBRIC,
    "SAFETY_FEWSHOTS": T.canonical_fewshots("safety"),
    "ABSTENTION_SYSTEM": T.ABSTENTION_SYSTEM,
    "ABSTENTION_RUBRIC": T.ABSTENTION_RUBRIC,
    "ABSTENTION_FEWSHOTS": T.canonical_fewshots("abstention"),
    "REDTEAM_SYSTEM": T.REDTEAM_SYSTEM,
    "REDTEAM_RUBRIC": T.REDTEAM_RUBRIC,
    "REDTEAM_FEWSHOTS": T.canonical_fewshots("redteam"),
    "DISPATCH_SYSTEM": T.DISPATCH_SYSTEM,
    "DISPATCH_RUBRIC": T.DISPATCH_RUBRIC,
    "DISPATCH_FEWSHOTS": T.canonical_fewshots("dispatch"),
}


def main() -> None:
    body = PLAYBOOK.read_text()
    for name, text in BLOCKS.items():
        pattern = re.compile(
            r"<!-- VERBATIM:" + name + r":START -->\n(.*?)"
            r"<!-- VERBATIM:" + name + r":END -->",
            re.DOTALL,
        )
        replacement = ("<!-- VERBATIM:" + name + ":START -->\n```\n"
                       + text.rstrip() + "\n```\n"
                       + "<!-- VERBATIM:" + name + ":END -->")
        body, count = pattern.subn(replacement, body, count=1)
        if count != 1:
            raise SystemExit(f"sentinel pair missing for {name}")
    PLAYBOOK.write_text(body)
    print(f"injected {len(BLOCKS)} blocks into {PLAYBOOK}")


if __name__ == "__main__":
    main()
