"""Lossless round-trip reader/writer for X-Plane .acf aircraft files.

An .acf is flat UTF-8 text with this skeleton:

    I                      <- platform byte ('I' = Intel/little-endian)
    1200 Version           <- format version (X-Plane version x100; 1200 = XP12)
    ACF                    <- file-type token
                           <- blank line
    PROPERTIES_BEGIN
    P <path> <value>       <- the entire flight model, one property per line
    ...
    PROPERTIES_END
    PANEL_2D_BEGIN ... PANEL_2D_END    <- 2D instrument panel (verbatim block)
    PANEL_3D_BEGIN ... PANEL_3D_END    <- 3D cockpit (verbatim block)

Verified against every stock XP12 aircraft: pure '\\n' endings, every property
line matches exactly ``P <path> <value>`` (single spaces), property paths are
unique, and there is no trailing whitespace on values.

This module models the file as an ordered list of lines. Property lines are
parsed into (path, value) so they can be read and mutated by path; every other
line (header, section markers, panel content) is preserved verbatim. An
*unmodified* property line is emitted from its original raw text, so loading any
stock .acf and saving it back produces byte-identical output. Only lines we
actually change are re-serialized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

_PROP_RE = re.compile(r"^P (\S+) (.*)$")

Scalar = Union[str, int, float, bool]


@dataclass
class _Line:
    """One physical line of the file.

    For a property line, ``path`` and ``value`` are populated. ``raw`` always
    holds the original text; it is re-used on output unless the value was
    changed, which guarantees a byte-faithful round trip for untouched lines.
    """

    raw: str
    path: Optional[str] = None
    value: Optional[str] = None
    _modified: bool = False

    @property
    def is_property(self) -> bool:
        return self.path is not None

    def render(self) -> str:
        if self.is_property and self._modified:
            return f"P {self.path} {self.value}"
        return self.raw


class ACF:
    """A parsed X-Plane .acf file: lossless to round-trip, mutable by path."""

    def __init__(self, lines: list[_Line], trailing_newline: bool = True):
        self._lines = lines
        self.trailing_newline = trailing_newline
        # Property paths are unique (verified across all stock aircraft), so a
        # path -> line map is safe and gives O(1) get/set.
        self._index: dict[str, _Line] = {
            ln.path: ln for ln in lines if ln.is_property
        }

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ACF":
        text = Path(path).read_bytes().decode("utf-8")
        trailing = text.endswith("\n")
        body = text[:-1] if trailing else text  # drop the single final newline
        lines: list[_Line] = []
        for raw in body.split("\n"):
            m = _PROP_RE.match(raw) if raw.startswith("P ") else None
            if m:
                lines.append(_Line(raw=raw, path=m.group(1), value=m.group(2)))
            else:
                lines.append(_Line(raw=raw))
        return cls(lines, trailing_newline=trailing)

    # ------------------------------------------------------------- serialize
    def render(self) -> str:
        out = "\n".join(ln.render() for ln in self._lines)
        if self.trailing_newline:
            out += "\n"
        return out

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_bytes(self.render().encode("utf-8"))

    # -------------------------------------------------------------- accessors
    def get(self, path: str) -> Optional[str]:
        ln = self._index.get(path)
        return ln.value if ln is not None else None

    def get_float(self, path: str) -> Optional[float]:
        v = self.get(path)
        return None if v is None else float(v)

    def get_int(self, path: str) -> Optional[int]:
        v = self.get(path)
        return None if v is None else int(float(v))

    def set(self, path: str, value: Scalar) -> "ACF":
        """Set an existing property. Raises KeyError if the path is absent.

        (Adding brand-new properties/parts is a later concern; templates we
        generate from already contain the full property set.)
        """
        ln = self._index.get(path)
        if ln is None:
            raise KeyError(f"property not present in this file: {path!r}")
        sval = self._fmt(value)
        if sval != ln.value:
            ln.value = sval
            ln._modified = True
        return self

    def match(self, prefix: str) -> dict[str, str]:
        """All properties whose path starts with ``prefix`` (order preserved)."""
        return {
            ln.path: ln.value
            for ln in self._lines
            if ln.is_property and ln.path.startswith(prefix)
        }

    def paths(self) -> Iterator[str]:
        return iter(self._index.keys())

    def __contains__(self, path: str) -> bool:
        return path in self._index

    def __getitem__(self, path: str) -> str:
        v = self.get(path)
        if v is None:
            raise KeyError(path)
        return v

    def __setitem__(self, path: str, value: Scalar) -> None:
        self.set(path, value)

    def __len__(self) -> int:
        return len(self._index)

    # ----------------------------------------------------------------- header
    @property
    def platform(self) -> str:
        return self._lines[0].raw if self._lines else ""

    @property
    def version(self) -> str:
        # Second line looks like "1200 Version".
        if len(self._lines) > 1:
            return self._lines[1].raw.split(" ", 1)[0]
        return ""

    # ----------------------------------------------------------------- format
    @staticmethod
    def _fmt(value: Scalar) -> str:
        """Render a Python value the way X-Plane writes them.

        Booleans -> 0/1, floats -> fixed 9-decimal (matches Plane Maker's float
        style; X-Plane parses tolerantly via atof, so exactness isn't required),
        everything else -> str().
        """
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            return f"{value:.9f}"
        return str(value)
