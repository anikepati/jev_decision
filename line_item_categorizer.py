"""
line_item_categorizer.py
========================
Tiered categorizer for construction-loan budget line items.

    Hard Cost | Soft Cost | Acquisition | Others

Tiers run in order and the first confident answer wins:

    1. approved_budget  - inherit the category of the matching line on the
                          loan's previously approved budget (exact, then fuzzy)
    2. dictionary       - exact lookup of reviewer-confirmed descriptions
    3. rules            - abbreviation expansion + keyword/phrase rules
    4. ml               - TF-IDF + logistic regression trained on your history
    5. jev              - TypeSafe Jev Choice question (optional, cloud API)
    6. review           - nothing was confident: best guess + needs_review=True

Tiers 1-3 are fully deterministic. Tiers 4 and 5 are optional and are skipped
when no model / client is supplied. The tier-4 model lives in category_model.py:
build it once offline, then load it once at startup and pass it as ml_model=.

Quick start
-----------
    from line_item_categorizer import LineItemCategorizer

    cat = LineItemCategorizer()
    result = cat.categorize("Arch & Eng fees - bldg A")
    print(result.category, result.tier, result.confidence)

This module is stdlib-only. scikit-learn is needed only by category_model.py
(tier 4) and typesafe_sdk only when a Jev client is supplied (tier 5).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

log = logging.getLogger(__name__)

HARD, SOFT, ACQ, OTHER = "Hard Cost", "Soft Cost", "Acquisition", "Others"
CATEGORIES = (HARD, SOFT, ACQ, OTHER)

_CATEGORY_ALIASES = {
    "hard": HARD, "hard cost": HARD, "hard costs": HARD, "hardcost": HARD, "hc": HARD,
    "soft": SOFT, "soft cost": SOFT, "soft costs": SOFT, "softcost": SOFT, "sc": SOFT,
    "acquisition": ACQ, "acquisitions": ACQ, "acquisition cost": ACQ,
    "acquisition costs": ACQ, "acq": ACQ,
    "other": OTHER, "others": OTHER, "misc": OTHER, "miscellaneous": OTHER,
}


def canonical_category(value: Any) -> Optional[str]:
    """Map 'hard costs', 'SOFT', 'Acq' ... to the canonical label, else None."""
    key = " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())
    return _CATEGORY_ALIASES.get(key)


ALL_TIERS = ("approved_budget", "dictionary", "rules", "ml", "jev")

# --------------------------------------------------------------------------
# Default configuration. Override any key via LineItemCategorizer(config=...)
# or LineItemCategorizer.from_json("config.json") - no code change needed.
# --------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    # Which tiers run, in this fixed order. Remove a name to switch a tier off.
    "enabled_tiers": list(ALL_TIERS),
    "thresholds": {
        "budget_fuzzy": 0.88,   # min similarity to inherit from approved budget
        "budget_margin": 0.05,  # best match must beat a differently-categorized runner-up by this
        "typo_fuzzy": 0.84,     # min similarity to fix a typo against the rule vocabulary
        "ml": 0.80,             # min classifier probability
        "jev": 0.85,            # min Jev confidence
    },
    # token -> expansion, applied after lowercasing / punctuation stripping
    "abbreviations": {
        "bldg": "building", "bld": "building", "bldgs": "buildings",
        "arch": "architect", "archt": "architect", "a&e": "architect engineering",
        "eng": "engineering", "engr": "engineering", "engrg": "engineering",
        "mep": "mechanical electrical plumbing", "hvac": "hvac",
        "elec": "electrical", "plumb": "plumbing", "plbg": "plumbing",
        "mech": "mechanical", "conc": "concrete", "fdn": "foundation",
        "fndn": "foundation", "demo": "demolition", "excav": "excavation",
        "gc": "general contractor", "gen": "general", "cond": "conditions",
        "reqs": "requirements", "req": "requirements",
        "contr": "contractor", "constr": "construction", "const": "construction",
        "ins": "insurance", "insur": "insurance", "int": "interest",
        "res": "reserve", "resv": "reserve", "mgmt": "management",
        "mgt": "management", "dev": "developer", "devel": "developer",
        "acq": "acquisition", "acquis": "acquisition", "prop": "property",
        "re": "real estate", "ffe": "furniture fixtures equipment",
        "ff&e": "furniture fixtures equipment", "ti": "tenant improvements",
        "o&p": "overhead profit", "oh&p": "overhead profit",
        "env": "environmental", "enviro": "environmental",
        "geotech": "geotechnical", "appr": "appraisal", "misc": "miscellaneous",
        "mktg": "marketing", "mkt": "marketing", "admin": "administrative",
        "svcs": "services", "svc": "services", "prof": "professional",
        "equip": "equipment", "matl": "materials", "matls": "materials",
        "lndscp": "landscaping", "landscp": "landscaping",
        "util": "utilities", "utils": "utilities", "pmt": "payment",
        "perf": "performance", "contg": "contingency", "cont": "contingency",
        "contig": "contingency", "fin": "financing", "orig": "origination",
        "pmts": "payments", "ext": "exterior", "intr": "interior",
    },
    # Phrases are matched on whole words after normalization. Multi-word
    # phrases score higher than single words, so "construction interest"
    # (Soft) beats the bare word "construction" (Hard).
    "rules": {
        HARD: [
            "sitework", "site work", "site improvements", "grading", "excavation",
            "earthwork", "demolition", "foundation", "concrete", "masonry",
            "framing", "carpentry", "roofing", "roof", "drywall", "insulation",
            "plumbing", "electrical", "mechanical", "hvac", "elevator",
            "windows", "doors", "glazing", "flooring", "paint", "painting",
            "finishes", "cabinets", "countertops", "appliances", "landscaping",
            "paving", "asphalt", "structural steel", "steel", "fire sprinkler",
            "fire protection", "building", "construction", "general contractor",
            "general conditions", "general requirements", "overhead profit",
            "contractor fee", "hard cost contingency", "construction contingency",
            "tenant improvements", "furniture fixtures equipment", "materials",
            "labor", "utilities installation", "offsite improvements",
            "waterproofing", "siding", "stucco", "millwork", "rehab",
            "rehabilitation", "renovation",
        ],
        SOFT: [
            "architect", "architectural", "engineering", "engineer", "design fees",
            "survey", "geotechnical", "soils report", "environmental",
            "phase i", "phase ii", "appraisal", "legal", "attorney", "accounting",
            "audit", "permit", "permits", "impact fees", "plan check",
            "inspection", "inspections", "testing", "insurance", "builders risk",
            "property taxes", "real estate taxes", "taxes", "interest reserve",
            "construction interest", "loan interest", "loan fees",
            "origination fee", "financing fees", "lender fees", "bank fees",
            "construction loan fees", "construction management",
            "construction manager", "construction period interest",
            "construction period taxes", "construction period insurance",
            "developer fee", "development fee", "consultant", "consulting",
            "marketing", "leasing", "lease up", "market study", "syndication",
            "tax credit fees", "soft cost contingency", "operating reserve",
            "replacement reserve", "reserves", "relocation", "title and recording",
            "administrative", "project management", "professional services",
            "payment performance bond", "bond premium", "cost certification",
        ],
        ACQ: [
            "acquisition", "land", "land purchase", "land cost", "purchase price",
            "property purchase", "building purchase", "building acquisition",
            "land acquisition", "closing costs", "closing", "escrow",
            "title insurance", "title", "broker fee", "broker commission",
            "brokerage", "transfer tax", "earnest money", "due diligence",
            "option payment", "existing structure purchase", "real estate purchase",
        ],
        OTHER: ["miscellaneous", "other", "others", "unallocated", "sundry"],
    },
    # Category descriptions sent to Jev as Choice criteria.
    "jev_criteria": {
        HARD: "Physical construction of the project: labor, materials, sitework, "
              "building trades, general contractor costs, construction contingency",
        SOFT: "Non-physical project costs: architect, engineering, legal, permits, "
              "insurance, taxes, financing fees, interest reserve, developer fee",
        ACQ: "Buying the land or existing property: purchase price, closing, "
             "title, escrow, broker fees, transfer tax",
        OTHER: "Does not clearly fit hard cost, soft cost or acquisition",
    },
    "jev_model": "jev-1.13.0",   # pinned; do not use a moving alias in production
}


@dataclass
class CategoryResult:
    """Final output for one line item."""
    line_item: str                      # what you passed in
    normalized: str                     # after cleanup / abbreviation expansion
    category: str                       # one of CATEGORIES (best guess if needs_review)
    confidence: float                   # 0..1
    tier: str                           # which tier decided
    needs_review: bool = False
    reason: str = ""                    # human-readable why
    trace: list[dict] = field(default_factory=list)   # what every tier said

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------
_AMOUNT = re.compile(r"[$€£]?\s*\(?\d[\d,]*(?:\.\d+)?\)?\s*%?")
_KEEP_AMP = re.compile(r"\b(\w{1,3})\s*&\s*(\w{1,3})\b")
_NON_WORD = re.compile(r"[^a-z0-9&\s]")
_ROMAN = {"i", "ii", "iii", "iv"}


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else v
    return out


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class LineItemCategorizer:
    """
    Parameters
    ----------
    config : overrides merged on top of DEFAULT_CONFIG.
    confirmed : {description: category} already confirmed by reviewers (tier 2).
    ml_model : a loaded ``category_model.CategoryModel`` (or a path to a saved
        one). Tier 4 is skipped when None.
    jev_client : an open ``typesafe_sdk.TypeSafeClient`` (or anything with a
        compatible ``system_one``). Tier 5 is skipped when None.
    """

    def __init__(
        self,
        config: Optional[Mapping] = None,
        confirmed: Optional[Mapping[str, str]] = None,
        ml_model: Any = None,
        jev_client: Any = None,
    ):
        self.cfg = _deep_merge(DEFAULT_CONFIG, config or {})
        self.th = self.cfg["thresholds"]
        unknown_tiers = set(self.cfg["enabled_tiers"]) - set(ALL_TIERS)
        if unknown_tiers:
            raise ValueError(f"Unknown tier(s) {sorted(unknown_tiers)}; expected any of {ALL_TIERS}")
        self.enabled_tiers = [t for t in ALL_TIERS if t in self.cfg["enabled_tiers"]]
        self.abbr: dict[str, str] = {k.lower(): v.lower() for k, v in self.cfg["abbreviations"].items()}

        # phrase -> category, plus vocabulary for typo repair
        self._phrases: dict[str, str] = {}
        for cat, phrases in self.cfg["rules"].items():
            self._check_category(cat)
            for p in phrases:
                self._phrases[" ".join(p.lower().split())] = cat
        self._vocab = sorted({w for p in self._phrases for w in p.split() if len(w) > 3})

        self._confirmed: dict[str, str] = {}
        for desc, cat in (confirmed or {}).items():
            self.learn(desc, cat)

        self._model = None
        if ml_model is not None:
            self.set_ml_model(ml_model)
        self.jev_client = jev_client

    # ---- construction helpers -------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "LineItemCategorizer":
        """Load config overrides from a JSON file."""
        return cls(config=json.loads(Path(path).read_text()), **kwargs)

    @staticmethod
    def _check_category(cat: str) -> None:
        if cat not in CATEGORIES:
            raise ValueError(f"Unknown category {cat!r}; expected one of {CATEGORIES}")

    # ---- normalization --------------------------------------------------
    def normalize(self, text: str) -> str:
        """lowercase, drop amounts/punctuation, expand abbreviations, fix typos."""
        t = str(text or "").lower()
        t = _KEEP_AMP.sub(lambda m: f"{m.group(1)}&{m.group(2)}", t)   # "A & E" -> "a&e"
        t = _NON_WORD.sub(" ", t)
        out: list[str] = []
        for tok in t.split():
            if tok in self.abbr:
                out.extend(self.abbr[tok].split())
                continue
            if "&" in tok:                      # unknown x&y: split it
                out.extend(p for p in tok.split("&") if p)
                continue
            if _AMOUNT.fullmatch(tok):          # amounts / unit numbers add noise
                continue
            out.append(tok)
        out = [self.abbr.get(w, w) for w in out]
        out = [w for piece in out for w in piece.split()]
        return " ".join(self._fix_typo(w) for w in out)

    def _fix_typo(self, word: str) -> str:
        if len(word) < 5 or word in self._vocab or word in _ROMAN:
            return word
        best, score = word, 0.0
        for v in self._vocab:
            if abs(len(v) - len(word)) > 2 or v[0] != word[0]:
                continue
            s = _sim(word, v)
            if s > score:
                best, score = v, s
        return best if score >= self.th["typo_fuzzy"] else word

    # ---- tier 1: approved budget ---------------------------------------
    def _tier_budget(self, norm: str, approved_budget) -> Optional[tuple[str, float, str]]:
        if not approved_budget:
            return None
        items = approved_budget.items() if isinstance(approved_budget, Mapping) else approved_budget
        scored: list[tuple[float, str, str]] = []
        for desc, cat in items:
            cat = canonical_category(cat)
            if cat is None:
                continue
            n = self.normalize(desc)
            if not n:
                continue
            if n == norm:
                return cat, 1.0, f"exact match to approved budget line {desc!r}"
            scored.append((_sim(norm, n), cat, desc))
        if not scored:
            return None
        scored.sort(reverse=True)
        top_s, top_cat, top_desc = scored[0]
        if top_s < self.th["budget_fuzzy"]:
            return None
        rival = next((s for s, c, _ in scored[1:] if c != top_cat), 0.0)
        if top_s - rival < self.th["budget_margin"]:
            return None                          # two budget lines disagree; don't guess
        return top_cat, round(top_s, 3), f"fuzzy match ({top_s:.2f}) to approved budget line {top_desc!r}"

    # ---- tier 2: confirmed dictionary ----------------------------------
    def _tier_dictionary(self, norm: str) -> Optional[tuple[str, float, str]]:
        cat = self._confirmed.get(norm)
        return (cat, 1.0, "previously confirmed by a reviewer") if cat else None

    def learn(self, line_item: str, category: str) -> None:
        """Record a reviewer-confirmed answer so the same text is instant next time.
        Persist ``export_confirmed()`` to your own store (e.g. PostgreSQL)."""
        self._check_category(category)
        norm = self.normalize(line_item)
        if norm:
            self._confirmed[norm] = category

    def export_confirmed(self) -> dict[str, str]:
        return dict(self._confirmed)

    # ---- tier 3: rules --------------------------------------------------
    def _tier_rules(self, norm: str) -> Optional[tuple[str, float, str]]:
        words = norm.split()
        if not words:
            return None
        hits: list[tuple[int, int, int, str, str]] = []   # (start, end, nwords, phrase, cat)
        for n in range(min(4, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                cat = self._phrases.get(phrase)
                if cat:
                    hits.append((i, i + n, n, phrase, cat))
        # longest phrases win; drop shorter hits they overlap
        hits.sort(key=lambda h: (-h[2], h[0]))
        taken: set[int] = set()
        kept = []
        for start, end, n, phrase, cat in hits:
            span = set(range(start, end))
            if span & taken:
                continue
            taken |= span
            kept.append((n, phrase, cat))
        if not kept:
            return None

        score: dict[str, float] = {}
        for n, phrase, cat in kept:
            score[cat] = score.get(cat, 0.0) + n * n      # 2-word phrase = 4x a single word
        specific = {c: s for c, s in score.items() if c != OTHER}
        if not specific:
            return OTHER, 0.9, f"matched catch-all term {kept[0][1]!r}"
        ranked = sorted(specific.items(), key=lambda kv: -kv[1])
        best_cat, best = ranked[0]
        if len(ranked) > 1 and ranked[1][1] >= best * 0.6:
            return None        # mixed signals (likely a consolidated line) -> next tier
        matched = ", ".join(repr(p) for _, p, c in kept if c == best_cat)
        conf = 0.95 if len(ranked) == 1 else 0.85
        return best_cat, conf, f"rule match on {matched}"

    # ---- tier 4: ML classifier (see category_model.py) ------------------
    def normalization_fingerprint(self) -> str:
        """Hash of everything that affects normalize(). A model trained under a
        different fingerprint saw differently-cleaned text and should be rebuilt."""
        payload = json.dumps(
            {"abbr": self.abbr, "vocab": self._vocab, "typo": self.th["typo_fuzzy"]},
            sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def set_ml_model(self, ml_model: Any) -> "LineItemCategorizer":
        """Attach a CategoryModel (or a path to one). Call once at startup."""
        if isinstance(ml_model, (str, Path)):
            from category_model import CategoryModel
            ml_model = CategoryModel.load(ml_model)
        unknown = set(ml_model.classes) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"Model predicts unknown categories: {sorted(unknown)}")
        trained_fp = getattr(ml_model, "meta", {}).get("normalization_fingerprint")
        if trained_fp and trained_fp != self.normalization_fingerprint():
            log.warning(
                "ML model was trained with different abbreviations/rules than this "
                "pipeline is configured with; rebuild the model to keep them in step")
        self._model = ml_model
        return self

    def _tier_ml(self, norm: str) -> Optional[tuple[str, float, str]]:
        if self._model is None or not norm:
            return None
        cat, prob = self._model.predict(norm)
        return cat, round(prob, 3), "trained classifier"

    # ---- tier 5: Jev ----------------------------------------------------
    def _tier_jev(self, norm: str) -> Optional[tuple[str, float, str]]:
        if self.jev_client is None or not norm:
            return None
        try:
            from typesafe_sdk import Choice, Noul
            resp = self.jev_client.system_one(
                # normalized text only: no amounts, no raw borrower punctuation
                state={"line_item": norm},
                questions={
                    "category": Choice(
                        instructions="Which construction-loan budget category best fits `line_item`?",
                        criteria=self.cfg["jev_criteria"],
                    ),
                    "consolidated": Noul(
                        instructions="Does `line_item` combine more than one kind of cost?"
                    ),
                },
            )
            ans = resp.answers["category"]
            choice, conf = ans.choice, float(ans.confidence)
            if choice not in CATEGORIES:
                return None
            mixed = resp.answers["consolidated"]
            # Noul: accept either a boolean value or a P(true) probability
            p_mixed = getattr(mixed, "probability", None)
            if p_mixed is None:
                p_mixed = 1.0 if getattr(mixed, "value", False) else 0.0
            if float(p_mixed) >= 0.5:
                conf = min(conf, 0.5)            # consolidated lines always go to review
                return choice, round(conf, 3), "Jev: line looks consolidated"
            return choice, round(conf, 3), "Jev decision model"
        except Exception as exc:                 # network, auth, SDK shape: never break the pipeline
            log.warning("Jev tier failed, falling through: %s", exc)
            return None

    # ---- orchestration --------------------------------------------------
    def categorize(
        self,
        line_item: str,
        approved_budget: Optional[Mapping[str, str] | Sequence[tuple[str, str]]] = None,
    ) -> CategoryResult:
        """
        Run every tier and return the final category.

        line_item        : borrower's description, e.g. "Arch & Eng fees - bldg A"
        approved_budget  : optional {description: category} (or list of pairs)
                           for the SAME loan's previously approved budget.
        """
        norm = self.normalize(line_item)
        trace: list[dict] = []
        candidates: list[tuple[str, float, str, str]] = []   # below-threshold guesses

        def done(cat, conf, tier, reason, review=False) -> CategoryResult:
            return CategoryResult(line_item, norm, cat, conf, tier, review, reason, trace)

        if not norm:
            return done(OTHER, 0.0, "review", "empty or numeric-only description", True)

        tiers = (
            ("approved_budget", lambda: self._tier_budget(norm, approved_budget), 0.0),
            ("dictionary", lambda: self._tier_dictionary(norm), 0.0),
            ("rules", lambda: self._tier_rules(norm), 0.0),
            ("ml", lambda: self._tier_ml(norm), self.th["ml"]),
            ("jev", lambda: self._tier_jev(norm), self.th["jev"]),
        )
        for name, run, threshold in tiers:
            if name not in self.enabled_tiers:
                continue
            out = run()
            if out is None:
                trace.append({"tier": name, "result": None})
                continue
            cat, conf, reason = out
            trace.append({"tier": name, "result": cat, "confidence": conf, "reason": reason})
            if conf >= threshold:
                return done(cat, conf, name, reason)
            candidates.append((cat, conf, name, reason))

        if candidates:      # nothing cleared its threshold: surface the strongest guess
            cat, conf, name, reason = max(candidates, key=lambda c: c[1])
            return done(cat, conf, "review", f"low confidence; best guess from {name} ({reason})", True)
        return done(OTHER, 0.0, "review", "no tier could classify this line item", True)

    def categorize_many(self, line_items: Iterable[str], approved_budget=None) -> list[CategoryResult]:
        return [self.categorize(li, approved_budget) for li in line_items]


if __name__ == "__main__":
    import sys
    cat = LineItemCategorizer()
    for item in sys.argv[1:] or ["Arch & Eng fees - bldg A", "Land purchase", "Sitewrk & grading"]:
        r = cat.categorize(item)
        flag = "  [REVIEW]" if r.needs_review else ""
        print(f"{item!r:45} -> {r.category:12} ({r.tier}, {r.confidence:.2f}){flag}  {r.reason}")
