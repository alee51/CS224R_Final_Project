"""Format judge trace JSON into readable markdown without dumping huge rollout text."""

import json
import sys
from pathlib import Path

HEAD = 800
TAIL = 800


def clip(text: str) -> str:
    if not isinstance(text, str):
        return repr(text)
    if len(text) <= HEAD + TAIL + 50:
        return text
    return (
        text[:HEAD]
        + f"\n\n--- [TRUNCATED — {len(text) - HEAD - TAIL} chars hidden] ---\n\n"
        + text[-TAIL:]
    )


def main(src: str, dst: str) -> None:
    data = json.loads(Path(src).read_text())
    meta = data["meta"]
    out = []
    out.append("# Judge trace — 4B verification\n")
    out.append("## Meta")
    out.append("```json")
    out.append(json.dumps(meta, indent=2, default=str))
    out.append("```\n")

    out.append("## Final cluster assignments")
    out.append("```json")
    out.append(json.dumps(data.get("final_cluster_ids"), indent=2, default=str))
    out.append("```\n")

    if "judge_parse" in data:
        out.append("## Judge parse")
        out.append("```json")
        out.append(json.dumps(data["judge_parse"], indent=2, default=str))
        out.append("```\n")

    if "judge_parsed_assignment" in data:
        out.append("## Judge parsed assignment (cluster IDs)")
        out.append("```json")
        out.append(json.dumps(data["judge_parsed_assignment"], indent=2, default=str))
        out.append("```\n")

    out.append("## Decoded problem")
    out.append("```")
    out.append(clip(data.get("decoded_problem", "")))
    out.append("```\n")

    rollouts = data.get("rollouts", [])
    out.append(f"## Rollouts ({len(rollouts)} total, each clipped to ~{HEAD}+{TAIL} chars)")
    for i, r in enumerate(rollouts):
        out.append(f"\n### Rollout {i}")
        out.append("```")
        out.append(clip(r))
        out.append("```")

    msgs = data.get("judge_messages", {})
    sys_msg = msgs.get("system", "")
    user_msg = msgs.get("user", "")
    out.append("\n## Judge prompt — system (first 800 chars)")
    out.append("```")
    out.append(sys_msg[:800] + ("\n..." if len(sys_msg) > 800 else ""))
    out.append("```\n")
    out.append(f"## Judge prompt — user (full size {len(user_msg)} chars; first 1500 only)")
    out.append("```")
    out.append(user_msg[:1500] + ("\n..." if len(user_msg) > 1500 else ""))
    out.append("```\n")

    raw = data.get("judge_raw_response", "")
    if raw:
        out.append(f"## Judge raw response ({len(raw)} chars)")
        out.append("```")
        out.append(clip(raw))
        out.append("```\n")

    Path(dst).write_text("\n".join(out))
    print(f"wrote {dst} ({Path(dst).stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
