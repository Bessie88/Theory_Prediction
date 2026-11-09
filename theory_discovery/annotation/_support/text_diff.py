"""Deterministic lossless character/token text alignment, never semantic Actions."""
from difflib import SequenceMatcher
from hashlib import sha256
import re
from .annotation_contracts import TextToken, DiffHunk, TextDiff
from theory_discovery.core.workflow_contracts import EvidenceSpan
from theory_discovery.core.schemas import require, stable_id

VERSION = "unicode-sequence-diff-v1"
TOKEN_PATTERN = r"\w+|[^\w\s]|\s+"


def tokenize(text):
    return tuple(TextToken(i, m.start(), m.end(), m.group())
                 for i, m in enumerate(re.finditer(TOKEN_PATTERN, text, flags=re.UNICODE)))


def _cover(tokens, start, end, length):
    if start == end:
        boundary = next((t.index for t in tokens if t.start >= start), len(tokens))
        crossing = next((t for t in tokens if t.start < start < t.end), None)
        return (crossing.index, crossing.index + 1, False) if crossing else (boundary, boundary, True)
    selected = [t for t in tokens if t.start < end and t.end > start]
    require(bool(selected), "Nonempty character span needs token coverage")
    return selected[0].index, selected[-1].index + 1, selected[0].start == start and selected[-1].end == end


def _char_boundary(tokens, index, length):
    return tokens[index].start if index < len(tokens) else length


def extract_diff(before, after):
    require(type(before) is str and type(after) is str, "Diff requires actual text strings")
    b_hash, a_hash = sha256(before.encode()).hexdigest(), sha256(after.encode()).hexdigest()
    bt, at = tokenize(before), tokenize(after)
    def hunk(kind, op, bs, be, ass, ae):
        b0, b1, bx = _cover(bt, bs, be, len(before))
        a0, a1, ax = _cover(at, ass, ae, len(after))
        key = stable_id("text_diff_hunk", [VERSION, b_hash, a_hash, kind, op, bs, be, ass, ae])
        return DiffHunk(key, kind, op.upper(),
            EvidenceSpan("before", bs, be, before[bs:be]),
            EvidenceSpan("after", ass, ae, after[ass:ae]), b0, b1, a0, a1, bx, ax)
    chars = tuple(hunk("CHARACTER", op, bs, be, ass, ae) for op, bs, be, ass, ae in
                  SequenceMatcher(None, before, after, autojunk=False).get_opcodes())
    words = tuple(hunk("TOKEN", op, _char_boundary(bt, bs, len(before)),
                      _char_boundary(bt, be, len(before)), _char_boundary(at, ass, len(after)),
                      _char_boundary(at, ae, len(after)))
                  for op, bs, be, ass, ae in SequenceMatcher(
                      None, tuple(t.text for t in bt), tuple(t.text for t in at), autojunk=False).get_opcodes())
    return TextDiff(b_hash, a_hash, VERSION, bt, at, chars, words)


def validate_diff(diff, before, after):
    require(diff == extract_diff(before, after), "Diff/version/offsets differ from deterministic source alignment")
