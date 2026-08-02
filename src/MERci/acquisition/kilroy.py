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

Protocol/command consistency
----------------------------
A Kilroy config also has an *internal* consistency requirement, independent of any
Dave recipe: every ``<valve>``/``<pump>`` step inside a ``<protocol>`` names a
valve/pump command by its text, and that name **must** exist as a
``<valve_cmd>``/``<pump_cmd>`` in the command sections — otherwise Kilroy errors at
load. Typos and inconsistent naming (e.g. ``"Wash buffer"`` vs ``"Wash Buffer"``,
or a stray trailing space) break this silently until run time. The consistency API
below verifies that link and proposes fuzzy-matched fixes:

- ``load_kilroy_commands(path)`` — the defined valve/pump command names.
- ``check_kilroy_consistency(path)`` — list of ``ConsistencyIssue`` for every
  protocol step whose command name is not defined, each with the closest-matching
  defined command as a suggested fix.
- ``format_consistency_report(issues)`` — human-readable summary of the issues.
- ``fix_kilroy_consistency(path, fixes)`` — apply confirmed name corrections to the
  config file in place (backing up the original to ``*.bak``), preserving the
  file's CRLF line endings and ISO-8859-1 encoding.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


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


def load_protocol_durations(path: Path) -> Dict[str, float]:
    """
    Return ``{protocol_name: duration_seconds}`` for every ``<protocol>`` in a
    Kilroy config.

    A protocol runs its ``<valve>``/``<pump>`` steps sequentially, waiting each
    step's ``duration`` (seconds) before the next, so its total time is the sum of
    those durations — exactly how Kilroy's ``KilroyProtocols.requiredTime`` computes
    it (``fluidics/kilroyProtocols.py``). Steps without a numeric ``duration`` count
    as 0. Kilroy files use ISO-8859-1 encoding.

    Parameters
    ----------
    path : path to the Kilroy config XML

    Returns
    -------
    dict : protocol name → total duration in seconds
    """
    text = Path(path).read_text(encoding="ISO-8859-1")
    root = ET.fromstring(text)
    durations: Dict[str, float] = {}
    for proto in root.iter("protocol"):
        name = (proto.get("name") or "").strip()
        if not name:
            continue
        total = 0.0
        for step in proto:                       # <valve>/<pump> children
            raw = step.get("duration")
            if raw is None:
                continue
            try:
                total += float(raw)
            except ValueError:
                pass                              # non-numeric duration → skip
        durations[name] = total
    return durations


def protocol_valve_commands(path: Path, protocol_name: str) -> List[str]:
    """
    Return the ordered list of ``<valve>`` command names used within one protocol.

    Used to detect when one protocol's own trailing steps already perform the
    same action as another (standalone) protocol -- e.g. a ``"Hybridize N"``
    protocol that ends by setting/flowing the imaging buffer itself, making a
    separately-appended ``"Flow Image Buffer"`` step in the Dave recipe
    redundant (see ``dave.py``'s ``_add_fluidics``).
    """
    return [r.name.strip() for r in iter_protocol_references(path)
            if r.protocol == protocol_name and r.kind == "valve"]


def protocol_last_flowed_valve(path: Path, protocol_name: str) -> Optional[str]:
    """
    Return the last ``<valve>`` command in *protocol_name* that was actually
    followed by a ``<pump>`` step (a real flow) -- ``None`` if the protocol
    never flows anything.

    Deliberately NOT just ``protocol_valve_commands(...)[-1]``: some Kilroy
    configs end a protocol with a bare valve *reposition* move that has no
    ``<pump>`` after it (e.g. parking the valve at the next hyb's port ready
    for the following cycle) -- confirmed directly against a real config
    (``kilroy-config-st2-syringe-direct-and-adaptors-260731.xml``) where
    ``"Hybridize N"`` ends `<valve>Set Image</valve><pump>...</pump>
    <valve>Set Hyb 1</valve>` (no pump after the trailing valve). Taking the
    literal last valve name there ("Set Hyb 1") instead of the last one that
    actually flowed ("Set Image") made ``_add_fluidics``'s already-flowed
    detection (see ``dave.py``) wrongly conclude the image buffer was NOT
    already flowed, appending a real second "Flow Image Buffer" step after
    every hybridization -- a genuine double-flow this function exists to
    prevent detecting incorrectly.
    """
    current_valve: Optional[str] = None
    last_flowed:   Optional[str] = None
    for r in iter_protocol_references(path):
        if r.protocol != protocol_name:
            continue
        if r.kind == "valve":
            current_valve = r.name.strip()
        elif r.kind == "pump" and current_valve is not None:
            last_flowed = current_valve
    return last_flowed


def find_kilroy_config(
    microscope:          str,
    kilroy_dir:          Path,
    fallback_microscope: str = "MF2",
) -> Path:
    """
    Locate the Kilroy config for *microscope* in *kilroy_dir*.

    Files are matched case-insensitively by microscope name. When several match, a
    dated config (filename ending in a ``YYMMDD`` stamp) is always preferred over
    an undated one (e.g. a "thick"/draft variant with no date suffix), and the
    newest by that stamp is chosen among dated configs. When no file matches
    *microscope*, the newest config for *fallback_microscope* is returned instead
    (border case: a microscope without its own Kilroy config).

    Raises
    ------
    FileNotFoundError
        if neither *microscope* nor *fallback_microscope* has a matching config.
    """
    kilroy_dir = Path(kilroy_dir)

    def _date(p: Path) -> Optional[int]:
        m = re.search(r"(\d{6})\.xml$", p.name)
        return int(m.group(1)) if m else None

    def _candidates(mic: str) -> List[Path]:
        matches = [p for p in kilroy_dir.glob("*.xml") if mic.lower() in p.name.lower()]
        # Prefer a dated config (filename ending in a YYMMDD stamp) over an undated
        # one -- e.g. a "thick"/draft variant with no date suffix is never picked
        # while a real dated config for the same microscope exists. Among dated
        # configs, prefer the newest by that stamp.
        dated = [p for p in matches if _date(p) is not None]
        pool = dated if dated else matches
        return sorted(pool, key=lambda p: _date(p) or 0)

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
        # Prefer an exact "Cleave" or "Cleave Direct" match first, mirroring
        # image_buffer()'s exact-name preference below: some configs (e.g. ST2)
        # also define manual/alternate cleave variants ("Cleave then image") that
        # are not meant for automatic Dave-recipe selection but would otherwise
        # tie with the canonical direct-cleave protocol under the generic token
        # predicate and raise a false ambiguity.
        for exact in ("cleave", "cleave direct"):
            for p in self.protocols:
                if " ".join(p.strip().lower().split()) == exact:
                    return p
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


# ── Protocol ↔ command consistency ───────────────────────────────────────────

@dataclass
class ProtocolReference:
    """A single ``<valve>``/``<pump>`` step inside a protocol.

    Attributes
    ----------
    protocol : str
        Name of the ``<protocol>`` the step belongs to.
    kind : str
        ``"valve"`` or ``"pump"`` — which command section it must resolve against.
    name : str
        The command name referenced, i.e. the element's text **verbatim** (leading
        or trailing whitespace preserved, so whitespace-only mismatches are visible).
    """

    protocol: str
    kind: str
    name: str


@dataclass
class ConsistencyIssue:
    """One undefined command name referenced by one or more protocol steps.

    Attributes
    ----------
    kind : str
        ``"valve"`` or ``"pump"``.
    referenced : str
        The command name used in the protocol(s) that has no matching command
        definition (verbatim, including any stray whitespace).
    protocols : list of str
        Protocol names that reference this undefined command, in first-seen order.
    count : int
        Total number of steps (across all protocols) that reference it.
    suggestion : str or None
        Closest defined command of the same kind (highest string similarity), or
        ``None`` if that section defines no commands at all.
    score : float
        Similarity ratio in ``[0, 1]`` between *referenced* and *suggestion*
        (``difflib`` ratio, compared case-insensitively). 1.0 means they differ
        only by letter case and/or surrounding whitespace.
    normalized_match : bool
        ``True`` when *referenced* equals *suggestion* after stripping and
        case-folding — i.e. a trivial whitespace/case fix, safe to auto-apply.
    """

    kind: str
    referenced: str
    protocols: List[str]
    count: int
    suggestion: Optional[str]
    score: float
    normalized_match: bool


def load_kilroy_commands(path: Path) -> Dict[str, List[str]]:
    """
    Return the valve and pump command names defined in a Kilroy config.

    Reads the ``name`` attribute of every ``<valve_cmd>`` and ``<pump_cmd>``.
    Commands that are XML-commented out (e.g. an old ``<pump_commands>`` block) are
    ignored, matching what Kilroy actually loads. Kilroy files use ISO-8859-1.

    Parameters
    ----------
    path : Path
        Path to the Kilroy config XML.

    Returns
    -------
    dict
        ``{"valve": [names...], "pump": [names...]}`` in document order.
    """
    root = ET.fromstring(Path(path).read_text(encoding="ISO-8859-1"))
    valve = [el.get("name").strip() for el in root.iter("valve_cmd") if el.get("name")]
    pump = [el.get("name").strip() for el in root.iter("pump_cmd") if el.get("name")]
    return {"valve": valve, "pump": pump}


def iter_protocol_references(path: Path) -> List[ProtocolReference]:
    """
    Return every ``<valve>``/``<pump>`` step across all protocols in the config.

    The element text is kept verbatim (not stripped) so whitespace-only mismatches
    remain detectable. Non-command children of a protocol (e.g. XML comments) are
    skipped.
    """
    root = ET.fromstring(Path(path).read_text(encoding="ISO-8859-1"))
    refs: List[ProtocolReference] = []
    for proto in root.iter("protocol"):
        pname = (proto.get("name") or "").strip()
        for el in proto:
            if el.tag in ("valve", "pump"):
                refs.append(ProtocolReference(pname, el.tag, el.text or ""))
    return refs


def _closest_command(name: str, candidates: Sequence[str]) -> Tuple[Optional[str], float]:
    """
    Return the (candidate, similarity) whose case-folded form is most similar to
    *name*, using ``difflib.SequenceMatcher`` ratio. Returns ``(None, 0.0)`` when
    *candidates* is empty. Comparison is case-insensitive and whitespace-trimmed so
    a pure case/whitespace difference scores 1.0.
    """
    target = name.strip().casefold()
    best: Optional[str] = None
    best_score = 0.0
    for cand in candidates:
        score = SequenceMatcher(None, target, cand.strip().casefold()).ratio()
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def check_kilroy_consistency(path: Path) -> List[ConsistencyIssue]:
    """
    Verify that every protocol step names a command that actually exists.

    A step is consistent when its command name matches a defined command name
    **exactly**. Each distinct undefined name is reported once as a
    ``ConsistencyIssue`` carrying the protocols that use it and the closest-matching
    defined command as a suggested fix.

    Parameters
    ----------
    path : Path
        Path to the Kilroy config XML.

    Returns
    -------
    list of ConsistencyIssue
        Empty when the config is fully consistent. Sorted by kind then referenced
        name for stable display.
    """
    commands = load_kilroy_commands(path)
    defined = {"valve": set(commands["valve"]), "pump": set(commands["pump"])}

    # Group undefined references by (kind, verbatim name); track which protocols use
    # each and how many steps in total, so one report line covers all occurrences.
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for ref in iter_protocol_references(path):
        if ref.name in defined[ref.kind]:
            continue
        key = (ref.kind, ref.name)
        entry = grouped.setdefault(key, {"protocols": [], "count": 0})
        entry["count"] = int(entry["count"]) + 1  # type: ignore[assignment]
        protos: List[str] = entry["protocols"]  # type: ignore[assignment]
        if ref.protocol not in protos:
            protos.append(ref.protocol)

    issues: List[ConsistencyIssue] = []
    for (kind, referenced), entry in grouped.items():
        suggestion, score = _closest_command(referenced, commands[kind])
        normalized_match = (
            suggestion is not None
            and referenced.strip().casefold() == suggestion.strip().casefold()
        )
        issues.append(
            ConsistencyIssue(
                kind=kind,
                referenced=referenced,
                protocols=list(entry["protocols"]),  # type: ignore[arg-type]
                count=int(entry["count"]),  # type: ignore[arg-type]
                suggestion=suggestion,
                score=score,
                normalized_match=normalized_match,
            )
        )

    issues.sort(key=lambda i: (i.kind, i.referenced))
    return issues


def format_consistency_report(issues: Sequence[ConsistencyIssue]) -> str:
    """
    Render a human-readable summary of ``check_kilroy_consistency`` output.

    Shows each undefined command, the protocols that reference it, and the suggested
    fix with its similarity score. ``[normalized]`` marks case/whitespace-only
    differences (safe to apply); other suggestions are the nearest string match and
    should be reviewed before applying.
    """
    if not issues:
        return "OK - all protocol steps reference defined valve/pump commands."

    lines = [f"Found {len(issues)} undefined command reference(s):", ""]
    for issue in issues:
        protos = ", ".join(issue.protocols)
        lines.append(f"  [{issue.kind}] {issue.referenced!r}  (in {issue.count} step(s))")
        lines.append(f"        used by protocol(s): {protos}")
        if issue.suggestion is None:
            lines.append("        no defined commands of this kind to match against")
        else:
            tag = " [normalized]" if issue.normalized_match else ""
            lines.append(
                f"        suggested fix -> {issue.suggestion!r} "
                f"(similarity {issue.score:.2f}){tag}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def fix_kilroy_consistency(
    path:   Path,
    fixes:  "Mapping[Tuple[str, str], str] | Iterable[Tuple[str, str, str]]",
    *,
    backup: bool = True,
) -> Tuple[int, List[Tuple[str, str, str, int]]]:
    """
    Apply confirmed command-name corrections to a Kilroy config, in place.

    Each fix rewrites the text of matching ``<valve>``/``<pump>`` steps from a wrong
    command name to a correct one, wherever it appears in any protocol. Only the
    step text is touched: attributes (e.g. ``duration``), comments, formatting, the
    file's CRLF line endings, and its ISO-8859-1 encoding are all preserved (the
    edit is a targeted text substitution, not an XML re-serialisation, so the rich
    hand-written config layout is kept intact).

    Parameters
    ----------
    path : Path
        Path to the Kilroy config XML to modify.
    fixes : mapping or iterable
        Either a mapping ``{(kind, wrong_name): correct_name}`` or an iterable of
        ``(kind, wrong_name, correct_name)`` triples, where ``kind`` is ``"valve"``
        or ``"pump"``. Typically the confirmed subset of ``check_kilroy_consistency``
        suggestions.
    backup : bool, default True
        When any replacement is made, first copy the original file to
        ``<path>.bak`` (byte-for-byte) before overwriting.

    Returns
    -------
    (total, applied) : tuple
        *total* is the number of steps rewritten. *applied* is a list of
        ``(kind, wrong_name, correct_name, n_replaced)`` — one per requested fix,
        so callers can flag any fix that matched nothing (``n_replaced == 0``).
    """
    path = Path(path)
    items: List[Tuple[Tuple[str, str], str]] = (
        list(fixes.items())
        if isinstance(fixes, Mapping)
        else [((kind, wrong), right) for (kind, wrong, right) in fixes]
    )

    # Read as bytes then decode so newlines are NOT translated: this keeps the
    # config's Windows CRLF endings intact on write (text-mode I/O would rewrite
    # them and HAL/Kilroy require CRLF).
    original_bytes = path.read_bytes()
    text = original_bytes.decode("ISO-8859-1")

    total = 0
    applied: List[Tuple[str, str, str, int]] = []
    for (kind, wrong), right in items:
        # Match <valve ...>WRONG</valve> (or pump), capturing the open/close tags so
        # attributes are preserved; only the text between them is replaced.
        pattern = re.compile(
            r"(<" + kind + r"\b[^>]*>)" + re.escape(wrong) + r"(</" + kind + r">)"
        )
        text, n = pattern.subn(lambda m: m.group(1) + right + m.group(2), text)
        total += n
        applied.append((kind, wrong, right, n))

    if total:
        if backup:
            path.with_suffix(path.suffix + ".bak").write_bytes(original_bytes)
        path.write_bytes(text.encode("ISO-8859-1"))

    return total, applied
