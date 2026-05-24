"""Clean answer extraction and canonicalization for Run 0 retrospective relabeling.

Re-extracts final answers from immutable completion text using brace-balanced
``\\boxed{...}`` parsing, improved LaTeX normalization, and run-on rejection
for Answer:/last-line fallbacks. Does not modify production ``answer_parse.py``.
"""

from __future__ import annotations

import hashlib
import re

from pilot.train.answer_parse import BOXED_RE, _ANSWER_LINE

_BOXED_OPENER = re.compile(r"\\boxed\{")
_FRAC = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_INT_RE = re.compile(r"^-?\d+$")
_PERCENT = re.compile(r"^(-?\d+)\s*(?:\\%|%)$")
_MATH_TOKEN = re.compile(
    r"\\[a-zA-Z]+|[\d$\\{}^_=+\-*/().,]|\\frac|\\sqrt|\\boxed"
)
_WORD = re.compile(r"[a-zA-Z]{2,}")


def last_brace_balanced_boxed_inner(text: str) -> str | None:
    """Return inner text of the last ``\\boxed{...}`` with brace balancing."""
    last_inner: str | None = None
    for m in _BOXED_OPENER.finditer(text):
        start = m.end()
        depth = 1
        j = start
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last_inner = text[start : j - 1].strip()
    return last_inner


def is_runon(text: str, max_chars: int = 80) -> bool:
    """Heuristic: long prose-like tail unsuitable as a parsed answer."""
    if not text or not str(text).strip():
        return False
    s = str(text).strip()
    if len(s) > max_chars:
        return True
    words = _WORD.findall(s)
    math_hits = len(_MATH_TOKEN.findall(s))
    # Prose: many English words, little math structure
    if len(words) >= 6 and math_hits < 3:
        return True
    if len(words) >= 4 and math_hits == 0 and not _INT_RE.match(s.replace(",", "")):
        return True
    return False


def extract_answer_clean(completion: str) -> tuple[str, str]:
    """Extract parsed answer and path label from completion.

    Returns:
        (parsed, extract_path) where extract_path is one of:
        ``boxed_balanced``, ``answer_line``, ``last_line``,
        ``runon_rejected``, ``empty``.
    """
    if not completion or not str(completion).strip():
        return ("", "empty")

    text = completion.strip()
    inner = last_brace_balanced_boxed_inner(text)
    if inner is not None:
        return (inner, "boxed_balanced")

    line_matches = list(_ANSWER_LINE.finditer(text))
    if line_matches:
        candidate = line_matches[-1].group(1).strip()
        if is_runon(candidate):
            return ("", "runon_rejected")
        return (candidate, "answer_line")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        candidate = lines[-1]
        if is_runon(candidate):
            return ("", "runon_rejected")
        return (candidate, "last_line")

    return ("", "empty")


def normalize_answer_clean(text: str) -> str:
    """Canonical string for equality, clustering, and semantic buckets.

    Strips wrappers (``\\( \\)``, ``$``, percent), normalizes ``\\frac{a}{b}``
    to ``a/b``, maps integers to decimal strings. Does **not** globally strip ``}``.
    """
    if not text or not str(text).strip():
        return ""
    s = str(text).strip().replace(",", "")
    # Peel common LaTeX wrappers (order matters: longer tokens first)
    for token in (r"\text{", r"\textbf{", r"\(", r"\)", "$"):
        s = s.replace(token, "")
    s = s.strip()
    # Trailing punctuation from answer lines
    s = s.rstrip(".;,")
    # \frac{a}{b} → a/b
    fm = _FRAC.search(s)
    if fm:
        a, b = fm.group(1).strip(), fm.group(2).strip()
        s = f"{a}/{b}"
    elif "/" in s and not s.startswith("http"):
        parts = s.split("/", 1)
        if len(parts) == 2 and parts[0].lstrip("-").replace(".", "", 1).isdigit():
            if parts[1].lstrip("-").replace(".", "", 1).isdigit():
                s = f"{parts[0].strip()}/{parts[1].strip()}"
    # Percent → integer string
    pm = _PERCENT.match(s.replace(" ", ""))
    if pm:
        return str(int(pm.group(1)))
    # Pure integer
    if _INT_RE.match(s):
        return str(int(s))
    # Fraction already normalized (do not peel leading int from "1/2")
    if "/" in s:
        return s.lower()
    # Leading integer only when followed by prose (e.g. "202 mod ..."), not "3.19" or "3, 4"
    lead = re.match(r"^(-?\d+)\s+[a-zA-Z]", s)
    if lead and len(s) <= 32:
        return str(int(lead.group(1)))
    return s.lower()


def is_correct_clean(parsed: str, gold: str) -> bool:
    """True iff normalized parsed answer equals normalized gold."""
    if not parsed and not str(gold).strip():
        return True
    if not parsed:
        return False
    return normalize_answer_clean(parsed) == normalize_answer_clean(str(gold))


def cluster_id_clean(parsed: str) -> int:
    """Deterministic cluster id from canonical cleaned form (stable across processes)."""
    canon = normalize_answer_clean(parsed)
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def semantic_bucket_clean(parsed: str) -> str:
    """Heuristic human-readable bucket from cleaned parse (not training labels)."""
    canon = normalize_answer_clean(parsed)
    if not canon:
        return "empty"
    if "/" in canon and not canon.startswith("http"):
        parts = canon.split("/", 1)
        if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
            return f"frac:{parts[0]}/{parts[1]}"
    if _INT_RE.match(canon):
        return f"n:{int(canon)}"
    collapsed = re.sub(r"\s+", " ", canon.lower())[:120]
    return f"s:{collapsed}"


def shallow_boxed_inner_last(text: str) -> str | None:
    """Last shallow-regex ``\\boxed{...}`` inner (production diagnostic)."""
    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def nested_boxed_mismatch(completion: str) -> bool:
    """True when brace-balanced last boxed inner differs from shallow-regex last."""
    if "\\boxed{" not in completion:
        return False
    balanced = last_brace_balanced_boxed_inner(completion)
    if balanced is None:
        return False
    shallow = shallow_boxed_inner_last(completion)
    if shallow is None:
        return True
    return normalize_answer_clean(balanced) != normalize_answer_clean(shallow)


def is_truncated_boxed(completion: str) -> bool:
    """Completion ends mid-``\\boxed{`` (opener without balanced close)."""
    text = completion.rstrip()
    if not text.endswith("\\boxed{") and "\\boxed{" not in text:
        return False
    # Unclosed opener at end
    if text.rstrip().endswith("\\boxed{"):
        return True
    last_opener = text.rfind("\\boxed{")
    if last_opener < 0:
        return False
    start = last_opener + len("\\boxed{")
    depth = 1
    j = start
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return depth != 0
