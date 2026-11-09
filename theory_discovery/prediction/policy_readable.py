"""Conservative readable wikitext projection with exact Unicode source spans.

Display markup is removed deterministically. Section-local approximate matching
accepts minor wording and punctuation differences and always returns raw evidence.
"""
from dataclasses import dataclass
from html import unescape
from difflib import SequenceMatcher
import re

PROJECTION_VERSION = "policy-readable-source-v3"
MIN_QUOTE_SIMILARITY = 0.90


@dataclass(frozen=True)
class ReadablePolicy:
    text: str
    source_spans: tuple[tuple[int, int], ...]


def readable_policy(text, offset=0, wikitext=True):
    chars = [(c, offset+i, offset+i+1) for i, c in enumerate(text)]
    def substitute(pattern, select):
        nonlocal chars
        original = "".join(c[0] for c in chars)
        result, cursor = [], 0
        for match in re.finditer(pattern, original):
            result.extend(chars[cursor:match.start()])
            result.extend(select(match, chars))
            cursor = match.end()
        chars = result + chars[cursor:]
    if wikitext:
        # Piped links use the explicit display label; unpiped links use the target.
        # Unsupported nested/empty-label syntax remains literal.
        def display_link(match, rows):
            group = 2 if match.group(2) is not None else 1
            label = rows[match.start(group):match.end(group)]
            # Include the complete markup when a quote reaches a label edge,
            # so a quote ending at a link can be projected again independently.
            label[0] = (label[0][0], rows[match.start()][1], label[0][2])
            label[-1] = (label[-1][0], label[-1][1], rows[match.end()-1][2])
            return label
        substitute(r"\[\[([^\[\]|]+)(?:\|([^\[\]|]+))?\]\]", display_link)
        substitute(r"'{2,5}", lambda m, rows: [])
        substitute(r"</?(?:span|b|i|em|strong)\b[^>]*>", lambda m, rows: [])
        substitute(r"&(?:#[0-9]+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]+);",
                   lambda m, rows: [(c, rows[m.start()][1], rows[m.end()-1][2])
                                    for c in unescape(m.group())])
    substitute(r"\s+", lambda m, rows: [(" ", rows[m.start()][1], rows[m.end()-1][2])])
    return ReadablePolicy("".join(c[0] for c in chars), tuple((c[1], c[2]) for c in chars))


def match_policy_quote(snapshot, section, quote):
    """Return source evidence and match audit; never searches another section."""
    from theory_discovery.core.schemas import require
    # A model may quote two source fragments separated by an explicit ellipsis.
    # Match fragments in source order and return the full intervening raw span.
    parts = [part.strip() for part in re.split(r"\[?\s*(?:\.{3,}|…+)\s*\]?", quote) if part.strip()]
    if len(parts) > 1:
        from types import SimpleNamespace
        matches = []
        cursor = section.start
        for part in parts:
            scoped = SimpleNamespace(section_id=section.section_id, start=cursor, end=section.end)
            matched = match_policy_quote(snapshot, scoped, part)
            matches.append(matched)
            cursor = matched["end"]
        start, end = matches[0]["start"], matches[-1]["end"]
        return {"start": start, "end": end, "quote": snapshot.content[start:end],
                "model_quote": quote, "match_mode": "ELLIPSIS",
                "similarity": min(m["similarity"] for m in matches),
                "section_id": section.section_id, "fragments": matches}
    projected = readable_policy(snapshot.content[section.start:section.end],
                                section.start, snapshot.metadata.content_format == "wikitext")
    needle = re.sub(r"\s+", " ", quote).strip()
    require(bool(needle), "Empty readable policy quote")
    first = projected.text.find(needle)
    if first >= 0:
        begin, finish, score, mode = first, first+len(needle), 1.0, "READABLE"
    else:
        # Word comparison tolerates typography/case and a small number of edits.
        # Short quotes only allow the same words: fuzzy short phrases are unsafe.
        words = list(re.finditer(r"\w+", projected.text, re.UNICODE))
        wanted = re.findall(r"\w+", needle.casefold(), re.UNICODE)
        tokens = [m.group().casefold() for m in words]
        require(bool(wanted), "Policy quote has no textual evidence")
        size = len(wanted)
        best = None
        # First prefer a complete normalized token match.
        for i in range(len(tokens)-size+1):
            if tokens[i:i+size] == wanted:
                best = (1.0, i, i+size)
                break
        if best is None and size >= 5:
            slack = max(1, int(size * 0.12))
            for width in range(max(1, size-slack), size+slack+1):
                for i in range(len(tokens)-width+1):
                    candidate = tokens[i:i+width]
                    score = SequenceMatcher(None, wanted, candidate, autojunk=False).ratio()
                    if best is None or score > best[0]:
                        best = (score, i, i+width)
        require(best is not None and best[0] >= MIN_QUOTE_SIMILARITY,
                "Policy quote has no sufficiently similar source in its declared section")
        score, i, j = best
        begin, finish = words[i].start(), words[j-1].end()
        mode = "NORMALIZED" if score == 1.0 else "APPROXIMATE"
    start = projected.source_spans[begin][0]
    end = projected.source_spans[finish-1][1]
    return {"start": start, "end": end, "quote": snapshot.content[start:end],
            "model_quote": quote, "match_mode": mode, "similarity": score,
            "section_id": section.section_id}


def resolve_readable_quote(snapshot, section, quote):
    match = match_policy_quote(snapshot, section, quote)
    return match["start"], match["end"], match["quote"]
