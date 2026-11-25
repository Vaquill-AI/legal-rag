"""
Court-prestige tier classification for US legal sources.

Maps the court registry `court_id` (or, if missing, a court-name string) to a
prestige tier and a numeric multiplier used to boost rerank scores.

Why this exists
---------------
The retrieval pipeline returns a flat ranking of opinion chunks by semantic
similarity. For a query like "what is the Brady rule?", a state appellate
opinion that quotes Brady at length will out-score Brady v. Maryland itself
on text similarity - but a paying lawyer wants the SCOTUS case as the lead
authority. We multiply the rerank score by a tier-dependent factor so that,
for federal-law questions, SCOTUS and federal circuit cases float to the top
without crushing the underlying semantic signal.

Tiers (federal-law context)
---------------------------
- ``scotus``                  - Supreme Court of the United States    (×3.0)
- ``federal_circuit``         - Federal Circuit Court of Appeals      (×1.6)
- ``federal_district``        - Federal district court / bankruptcy   (×1.0  baseline)
- ``federal_specialty``       - Court of Federal Claims, Tax, etc.    (×1.1)
- ``state_supreme``           - State supreme court                   (×0.6)
- ``state_appellate``         - State intermediate appellate          (×0.3)
- ``state_trial``             - State trial / general jurisdiction    (×0.2)
- ``unknown``                 - Anything we can't classify            (×0.8)

For state-law questions, the federal-vs-state classifier flips the multiplier
table so that state supreme courts get the highest weight; that flip happens
in the caller, not here.

Source of truth: the public court_id taxonomy
covers the federal judiciary and every state's supreme court. State
intermediate-appellate courts are matched by an ``appellate`` / ``ctapp``
prefix heuristic, with explicit overrides for the few that don't fit the
pattern.
"""
# ruff: noqa: PIE810

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Tier:
    """Prestige tier for a US court."""

    name: str
    federal_multiplier: float
    state_multiplier: float
    is_federal: bool


# ---------------------------------------------------------------------------
# Tier definitions. ``federal_multiplier`` is used when the question is
# detected as federal-law; ``state_multiplier`` for state-law questions.
# ---------------------------------------------------------------------------

T_SCOTUS: Final = Tier("scotus", 3.0, 1.5, True)
T_FED_CIRCUIT: Final = Tier("federal_circuit", 1.6, 1.0, True)
T_FED_DISTRICT: Final = Tier("federal_district", 1.0, 0.7, True)
T_FED_SPECIALTY: Final = Tier("federal_specialty", 1.1, 0.7, True)
T_STATE_SUPREME: Final = Tier("state_supreme", 0.6, 2.4, False)
T_STATE_APPELLATE: Final = Tier("state_appellate", 0.3, 1.0, False)
T_STATE_TRIAL: Final = Tier("state_trial", 0.2, 0.6, False)
T_UNKNOWN: Final = Tier("unknown", 0.8, 0.8, False)


# the court registry uses short string identifiers. The full taxonomy has ~3,000
# courts; the curated map below covers everything paying customers will
# encounter on a US-legal Q&A widget.
_COURT_ID_MAP: Final[dict[str, Tier]] = {
    # -- Supreme Court --
    "scotus": T_SCOTUS,
    # -- Federal circuit courts of appeals --
    "ca1": T_FED_CIRCUIT,
    "ca2": T_FED_CIRCUIT,
    "ca3": T_FED_CIRCUIT,
    "ca4": T_FED_CIRCUIT,
    "ca5": T_FED_CIRCUIT,
    "ca6": T_FED_CIRCUIT,
    "ca7": T_FED_CIRCUIT,
    "ca8": T_FED_CIRCUIT,
    "ca9": T_FED_CIRCUIT,
    "ca10": T_FED_CIRCUIT,
    "ca11": T_FED_CIRCUIT,
    "cadc": T_FED_CIRCUIT,
    "cafc": T_FED_CIRCUIT,  # Federal Circuit (patents, claims)
    # -- Federal specialty courts --
    "uscfc": T_FED_SPECIALTY,  # Court of Federal Claims
    "tax": T_FED_SPECIALTY,
    "cit": T_FED_SPECIALTY,  # Court of International Trade
    "cavc": T_FED_SPECIALTY,  # Veterans Claims
    "armfor": T_FED_SPECIALTY,  # Armed Forces
    "cc": T_FED_SPECIALTY,  # Court of Claims (historical)
    "ccpa": T_FED_SPECIALTY,  # Customs & Patent Appeals (historical)
    # -- State supreme courts (US states + DC + territories) --
    # the court registry uses `<state>` for the state supreme court when the
    # state has a single high court.
    "ala": T_STATE_SUPREME,
    "alaska": T_STATE_SUPREME,
    "ariz": T_STATE_SUPREME,
    "ark": T_STATE_SUPREME,
    "cal": T_STATE_SUPREME,
    "colo": T_STATE_SUPREME,
    "conn": T_STATE_SUPREME,
    "del": T_STATE_SUPREME,
    "dc": T_STATE_SUPREME,  # DC Court of Appeals (functions as supreme)
    "fla": T_STATE_SUPREME,
    "ga": T_STATE_SUPREME,
    "haw": T_STATE_SUPREME,
    "idaho": T_STATE_SUPREME,
    "ill": T_STATE_SUPREME,
    "ind": T_STATE_SUPREME,
    "iowa": T_STATE_SUPREME,
    "kan": T_STATE_SUPREME,
    "ky": T_STATE_SUPREME,
    "la": T_STATE_SUPREME,
    "me": T_STATE_SUPREME,
    "md": T_STATE_SUPREME,
    "mass": T_STATE_SUPREME,
    "mich": T_STATE_SUPREME,
    "minn": T_STATE_SUPREME,
    "miss": T_STATE_SUPREME,
    "mo": T_STATE_SUPREME,
    "mont": T_STATE_SUPREME,
    "neb": T_STATE_SUPREME,
    "nev": T_STATE_SUPREME,
    "nh": T_STATE_SUPREME,
    "nj": T_STATE_SUPREME,
    "nm": T_STATE_SUPREME,
    "ny": T_STATE_SUPREME,  # Court of Appeals = NY's high court
    "nc": T_STATE_SUPREME,
    "nd": T_STATE_SUPREME,
    "ohio": T_STATE_SUPREME,
    "okla": T_STATE_SUPREME,
    "ore": T_STATE_SUPREME,
    "pa": T_STATE_SUPREME,
    "ri": T_STATE_SUPREME,
    "sc": T_STATE_SUPREME,
    "sd": T_STATE_SUPREME,
    "tenn": T_STATE_SUPREME,
    "tex": T_STATE_SUPREME,
    "utah": T_STATE_SUPREME,
    "vt": T_STATE_SUPREME,
    "va": T_STATE_SUPREME,
    "wash": T_STATE_SUPREME,
    "wva": T_STATE_SUPREME,
    "wis": T_STATE_SUPREME,
    "wyo": T_STATE_SUPREME,
    # Texas + Oklahoma split civil/criminal high courts
    "texcrimapp": T_STATE_SUPREME,
    "oklacrimapp": T_STATE_SUPREME,
    # Territories
    "pr": T_STATE_SUPREME,
    "guam": T_STATE_SUPREME,
    "vi": T_STATE_SUPREME,
    # the court registry's id for the Supreme Court of the Northern Mariana Islands.
    "nmariana": T_STATE_SUPREME,
}


# Heuristic prefixes / suffixes used when a court_id isn't in the curated map.
# Order matters - first match wins.
_PREFIX_RULES: Final[tuple[tuple[str, Tier], ...]] = (
    # Federal district courts use ISO-style state-suffixed IDs:
    # cand, casd, nyed, nysd, ilnd, txwd, etc. They almost always end in
    # one of these directional letter pairs.
    ("dcc", T_FED_DISTRICT),  # DC district variants
    # Bankruptcy courts:  bap1..bap9, ca[1-9]bk, etc.
    ("bap", T_FED_DISTRICT),
    ("bk", T_FED_DISTRICT),
)


def classify_court(court_id: str | None, court_name: str | None = None) -> Tier:
    """Return the prestige tier for a court.

    Lookup order:
      1. Exact match in :data:`_COURT_ID_MAP`.
      2. Federal-district pattern detection on the court_id.
      3. State-appellate / state-trial detection on the court_id.
      4. Court-name fallback for SCOTUS and circuit courts.
      5. ``T_UNKNOWN``.

    Parameters
    ----------
    court_id:
        The court registry short identifier (e.g., ``"scotus"``, ``"ca9"``,
        ``"cand"``, ``"ohioctapp"``).
    court_name:
        The human-readable court name. Used only as a fallback when the
        ``court_id`` is missing or empty.
    """
    cid = (court_id or "").strip().lower()
    name = (court_name or "").strip().lower()

    # --- 1. Exact ID match
    if cid in _COURT_ID_MAP:
        return _COURT_ID_MAP[cid]

    # --- 2. Federal district court pattern: 4-6 letters ending in
    # `c`/`d`/`e`/`m`/`n`/`s`/`w` (cand=Northern California, nysd=SDNY,
    # txwd=Western Texas, etc.). Also catches bankruptcy variants.
    if cid and 3 <= len(cid) <= 6:
        # Distinguish from state codes: state codes are usually 2-4 letters
        # of the state abbreviation (`cal`, `ny`, `tex`). District codes
        # contain a directional letter at the end.
        last = cid[-1]
        if last in {"d", "n", "s", "e", "w", "m", "c"} and any(ch.isdigit() is False for ch in cid):
            # Heuristic: if first 2-3 letters are a known state stub AND
            # it ends in a directional letter, treat as federal district.
            for stub in (
                "ca",
                "ny",
                "tx",
                "fl",
                "il",
                "pa",
                "oh",
                "ga",
                "nc",
                "mi",
                "nj",
                "va",
                "wa",
                "az",
                "ma",
                "tn",
                "in",
                "mo",
                "md",
                "wi",
                "co",
                "mn",
                "sc",
                "al",
                "la",
                "ky",
                "or",
                "ok",
                "ct",
                "ut",
                "nv",
                "ar",
                "ms",
                "ks",
                "nm",
                "ne",
                "wv",
                "id",
                "ha",
                "ri",
                "mt",
                "de",
                "sd",
                "nd",
                "ak",
                "vt",
                "wy",
                "me",
                "nh",
                "dc",
                "pr",
                "vi",
                "gu",
            ):
                if cid.startswith(stub):
                    return T_FED_DISTRICT

    # --- 3. State appellate / trial heuristics
    if cid:
        if "ctapp" in cid or "appct" in cid or "appellate" in cid:
            return T_STATE_APPELLATE
        if cid.endswith("super") or cid.endswith("dist") or cid.endswith("circuit"):
            return T_STATE_TRIAL

    # --- 4. Name-based fallback (when CL doesn't return a clean court_id)
    if name:
        if "supreme court of the united states" in name or "u.s. supreme court" in name:
            return T_SCOTUS
        if "court of appeals for the" in name and "circuit" in name:
            return T_FED_CIRCUIT
        if "district court" in name:
            # the court registry emits names like "District Court, S.D. New York"
            # for federal districts; reserve the "U.S./United States" check
            # for ambiguity but accept the bare "District Court" too -
            # state district courts use "Superior" or "Circuit" naming.
            return T_FED_DISTRICT
        if "bankruptcy court" in name:
            return T_FED_DISTRICT
        if "court of federal claims" in name:
            return T_FED_SPECIALTY
        if "supreme court" in name:
            return T_STATE_SUPREME
        if "court of appeals" in name or "appellate" in name:
            return T_STATE_APPELLATE

    return T_UNKNOWN


def boost_factor(tier: Tier, *, jurisdiction: str = "federal") -> float:
    """Return the rerank-score multiplier for a given tier + question jurisdiction.

    Parameters
    ----------
    tier:
        The :class:`Tier` returned by :func:`classify_court`.
    jurisdiction:
        ``"federal"`` (default) for federal-law questions or ``"state"`` for
        state-law questions. Anything else falls back to federal weights.
    """
    if jurisdiction == "state":
        return tier.state_multiplier
    return tier.federal_multiplier


__all__ = [
    "T_FED_CIRCUIT",
    "T_FED_DISTRICT",
    "T_FED_SPECIALTY",
    "T_SCOTUS",
    "T_STATE_APPELLATE",
    "T_STATE_SUPREME",
    "T_STATE_TRIAL",
    "T_UNKNOWN",
    "Tier",
    "boost_factor",
    "classify_court",
]
