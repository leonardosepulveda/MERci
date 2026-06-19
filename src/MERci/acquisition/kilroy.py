# MERci/acquisition/kilroy.py
"""
Read Kilroy fluidics configuration files and resolve Dave fluidic steps to the
protocol names that actually exist in a given Kilroy config.

Why this exists
---------------
A Dave recipe references Kilroy *protocols* by name (``<valve_protocol>`` text).
Every protocol a Dave config uses **must** exist as a ``<protocol>`` in the
Kilroy config that will run the experiment — otherwise Kilroy fails at runtime.

Protocol names are not standardised across microscopes: e.g. one Kilroy config
calls the adaptor cleave step ``"Cleave Adaptors"`` while another calls it
``"Cleave Adaptor"``; one numbers hyb steps ``"Hybridize 1 Adaptors"`` while the
abstract Dave step is "hyb #1 (adaptors)".  Rather than hard-code names in the
Dave generator, we read the Kilroy config and resolve each logical step to the
real protocol name by token matching.

Public API
----------
- ``load_kilroy_protocols(path)`` — list of ``<protocol name=...>`` names.
- ``find_kilroy_config(microscope, kilroy_dir, fallback_microscope="MF2")`` —
  locate the newest Kilroy config for a microscope, falling back to another
  microscope's config when none exists.
- ``KilroyProtocolResolver`` — map logical Dave steps (cleave / hybridize k /
  readouts / image buffer) to concrete Kilroy protocol names; raises a clear
  error when a required step has no matching protocol.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Sequence, Set


# ── Kilroy config parsing ────────────────────────────────────────────────────

def load_kilroy_protocols(path: Path) -> List[str]:
    """
    Return the ordered list of protocol names declared in a Kilroy config XML.

    Reads the ``name`` attribute of every ``<protocol>`` element under
    ``<kilroy_protocols>``.  Kilroy files use ISO-8859-1 encoding.
    """
    text = Path(path).read_text(encoding="ISO-8859-1")
    root = ET.fromstring(text)
    names = [el.get("name") for el in root.iter("protocol")]
    return [n.strip() for n in names if n and n.strip()]


def find_kilroy_config(
    microscope:          str,
    kilroy_dir:          Path,
    fallback_microscope: str = "MF2",
) -> Path:
    """
    Locate the Kilroy config for *microscope* in *kilroy_dir*.

    Files are matched case-insensitively by microscope name and, when several
    match, the newest by trailing ``YYMMDD`` date stamp is chosen.  When no file
    matches *microscope*, the newest config for *fallback_microscope* is returned
    instead (border case: a microscope without its own Kilroy config).

    Raises
    ------
    FileNotFoundError
        if neither *microscope* nor *fallback_microscope* has a matching config.
    """
    kilroy_dir = Path(kilroy_dir)

    def _date(p: Path) -> int:
        m = re.search(r"(\d{6})\.xml$", p.name)
        return int(m.group(1)) if m else 0

    def _candidates(mic: str) -> List[Path]:
        return sorted(
            (p for p in kilroy_dir.glob("*.xml") if mic.lower() in p.name.lower()),
            key=_date,
        )

    cands = _candidates(microscope)
    if cands:
        return cands[-1]

    if fallback_microscope:
        fb = _candidates(fallback_microscope)
        if fb:
            return fb[-1]

    raise FileNotFoundError(
        f"No Kilroy config for microscope '{microscope}' (nor fallback "
        f"'{fallback_microscope}') found in {kilroy_dir}"
    )


# ── Protocol-name resolution ─────────────────────────────────────────────────

def _tokens(name: str) -> List[str]:
    """
    Normalise a protocol name to a list of comparison tokens: lowercase, split on
    whitespace, and crudely singularised (trailing ``s`` stripped from purely
    alphabetic tokens) so ``Adaptor``/``Adaptors`` and ``Readout``/``Readouts``
    compare equal.
    """
    toks: List[str] = []
    for raw in name.lower().split():
        t = raw.strip()
        if not t:
            continue
        if t.isalpha() and len(t) > 1 and t.endswith("s"):
            t = t[:-1]
        toks.append(t)
    return toks


class KilroyProtocolResolver:
    """
    Resolve abstract Dave fluidic steps to concrete Kilroy protocol names.

    Construct from the protocol-name list of a Kilroy config (see
    ``load_kilroy_protocols``).  Each resolver method returns the single Kilroy
    protocol whose name matches the requested step, raising ``ValueError`` if
    none match or if the match is ambiguous.
    """

    def __init__(self, protocols: Sequence[str]):
        self.protocols: List[str] = list(protocols)
        self._tokensets = {p: set(_tokens(p)) for p in self.protocols}

    # -- internal --------------------------------------------------------------

    def _unique(self, predicate: Callable[[Set[str], str], bool], role: str) -> str:
        matches = [p for p in self.protocols if predicate(self._tokensets[p], p)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"No Kilroy protocol matches the required fluidic step "
                f"'{role}'. Available protocols: {self.protocols}"
            )
        raise ValueError(
            f"Ambiguous Kilroy protocol for fluidic step '{role}': {matches}. "
            f"Cannot decide which to use."
        )

    # -- public step resolvers -------------------------------------------------

    def cleave(self, adaptors: bool) -> str:
        if adaptors:
            return self._unique(
                lambda ts, n: {"cleave", "adaptor"} <= ts, "cleave (adaptors)"
            )
        # Direct cleave: a "cleave" protocol that is not the adaptor one. Matches
        # both a bare "Cleave" and a qualified "Cleave direct" (mirrors hybridize).
        return self._unique(
            lambda ts, n: "cleave" in ts and "adaptor" not in ts, "cleave (direct)"
        )

    def hybridize(self, k: int, adaptors: bool) -> str:
        kk = str(int(k))
        if adaptors:
            return self._unique(
                lambda ts, n: {"hybridize", kk, "adaptor"} <= ts,
                f"hybridize {kk} (adaptors)",
            )
        return self._unique(
            lambda ts, n: {"hybridize", kk} <= ts and "adaptor" not in ts,
            f"hybridize {kk} (direct)",
        )

    def readouts(self) -> str:
        return self._unique(lambda ts, n: "readout" in ts, "hybridize readouts")

    def image_buffer(self) -> str:
        # Prefer an exact "Flow Image Buffer" when present, so configs that also
        # define e.g. "Flow SSC then Image Buffer" resolve unambiguously.
        for p in self.protocols:
            if " ".join(p.lower().split()) == "flow image buffer":
                return p
        return self._unique(lambda ts, n: {"image", "buffer"} <= ts, "image buffer")

    def validate(self, names: Sequence[str]) -> None:
        """Raise if any name in *names* is not an exact Kilroy protocol name."""
        missing = [n for n in names if n not in self.protocols]
        if missing:
            raise ValueError(
                f"Fluidics protocol(s) not found in Kilroy config: {missing}. "
                f"Available protocols: {self.protocols}"
            )
