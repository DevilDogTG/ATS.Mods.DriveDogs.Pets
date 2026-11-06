"""Split SII/SUI text into units without misreading it.

The definition filter used to work line by line and was fooled by three things real mods contain: a byte-order
mark before the first unit, braces inside comments or quoted strings, and attribute lines that look like unit
headers ("price: 75"). This scanner tracks quotes and comments, so only real braces count, and a "class : name"
line is a unit header only when a "{" follows it.

A file is split into segments: ("text", lines) for anything between units (SiiNunit, braces, @include,
comments) and ("unit", class, name, lines) for a unit from its header line to its closing brace.

Copied unchanged from the ETS2 Pets repo's tools/sii_text.py (ADR 0001).
"""
from __future__ import annotations

import re

_HEAD = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*([.\w]+)\s*(\{)?\s*$")


def code_part(line: str) -> str:
    """The line without comments (# or //) that are outside quoted strings."""
    out, quoted, i = [], False, 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            quoted = not quoted
        elif not quoted and (ch == "#" or line.startswith("//", i)):
            break
        out.append(ch)
        i += 1
    return "".join(out)


def braces(line: str) -> int:
    """Opening minus closing braces outside quotes and comments."""
    code = re.sub(r'"[^"]*"', "", code_part(line))
    return code.count("{") - code.count("}")


def normalized_lines(text: str) -> list[str]:
    """Lines with the BOM removed and a closing brace followed by more code ("}accessory_x : y", Belka) split in
    two: the game reads SII as tokens, so that is valid, but this scanner works by line."""
    lines = []
    for line in text.lstrip("﻿").replace("\r\n", "\n").split("\n"):
        while True:
            match = re.match(r"^(\s*\})\s*(\S.*)$", line)
            if not match or match.group(2).startswith(("#", "//")):
                break
            lines.append(match.group(1))
            line = match.group(2)
        lines.append(line)
    return lines


def split_units(text: str) -> list[tuple]:
    lines = normalized_lines(text)
    segments: list[tuple] = []
    outside: list[str] = []
    i = 0
    while i < len(lines):
        head = _HEAD.match(code_part(lines[i]))
        if head and head.group(1) != "SiiNunit":
            # A header only when the brace is on this line or opens the next non-empty code line.
            j = i + 1
            while not head.group(3) and j < len(lines) and not code_part(lines[j]).strip():
                j += 1
            opens = head.group(3) or (j < len(lines) and code_part(lines[j]).strip().startswith("{"))
            if opens:
                if outside:
                    segments.append(("text", outside))
                    outside = []
                unit_lines, depth, k = [lines[i]], braces(lines[i]), i + 1
                if not head.group(3):
                    unit_lines += lines[i + 1:j + 1]
                    depth += sum(braces(x) for x in lines[i + 1:j + 1])
                    k = j + 1
                while depth > 0 and k < len(lines):
                    unit_lines.append(lines[k])
                    depth += braces(lines[k])
                    k += 1
                segments.append(("unit", head.group(1), head.group(2), unit_lines))
                i = k
                continue
        outside.append(lines[i])
        i += 1
    if outside:
        segments.append(("text", outside))
    return segments


def check_structure(text: str) -> list[str]:
    """Problems that make the game reject a file: unbalanced braces, a stray "{" between units, a nameless unit
    referenced but not defined in the file."""
    problems = []
    if sum(braces(line) for line in normalized_lines(text)) != 0:
        problems.append("unbalanced braces")
    defined = set()
    for seg in split_units(text):
        if seg[0] == "text":
            stray = [x for x in seg[1] if code_part(x).strip() == "{"]
            # The file's own "SiiNunit {" brace is the only allowed one between units.
            if len(stray) > 1 or (stray and not any(code_part(x).strip() == "SiiNunit" for x in seg[1])):
                problems.append("a '{' outside any unit")
        else:
            defined.add(seg[2])
    for ref in set(re.findall(r':\s*(\.[\w.]+)\s*$', text, re.MULTILINE)):
        if ref not in defined:
            problems.append(f"nameless unit {ref} referenced but not defined")
    return problems
