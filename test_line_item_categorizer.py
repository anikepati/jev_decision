"""Run: python -m pytest test_line_item_categorizer.py -q"""
import sys
import types

import pytest

from category_model import CategoryModel, main as model_cli
from line_item_categorizer import ACQ, HARD, OTHER, SOFT, LineItemCategorizer

BUDGET = {
    "Architect & Engineering": SOFT,
    "Building Construction": HARD,
    "Land Acquisition": ACQ,
    "Construction Interest Reserve": SOFT,
}


@pytest.fixture(scope="module")
def cat():
    return LineItemCategorizer()


@pytest.mark.parametrize("text,expected", [
    ("Arch & Eng fees - bldg A", SOFT),
    ("A & E", SOFT),
    ("Sitewrk & grading", HARD),
    ("Concrete fdn", HARD),
    ("HVAC / plumb / elec", HARD),
    ("Roofing $125,000.00", HARD),
    ("Land purchase", ACQ),
    ("Closing costs & title ins", ACQ),
    ("Const. interest", SOFT),            # phrase beats bare "construction"
    ("Construction mgmt fee", SOFT),
    ("Builders risk ins", SOFT),
    ("Permits and impact fees", SOFT),
    ("Developer fee", SOFT),
    ("Bldg acquisition", ACQ),            # phrase beats bare "building"
    ("Misc", OTHER),
    ("GC general conditions", HARD),
    ("Legal", SOFT),
    ("Enviromental phase I", SOFT),       # typo
    ("Demolition", HARD),
])
def test_rules(cat, text, expected):
    r = cat.categorize(text)
    assert (r.category, r.tier, r.needs_review) == (expected, "rules", False), r


def test_budget_exact_and_fuzzy(cat):
    r = cat.categorize("architect and engineering", BUDGET)
    assert (r.category, r.tier) == (SOFT, "approved_budget")
    r = cat.categorize("Bldg Constructon", BUDGET)       # abbreviation + typo
    assert (r.category, r.tier) == (HARD, "approved_budget")


def test_budget_overrides_rules(cat):
    # this lender books FF&E as soft; the approved budget wins over the default rule
    r = cat.categorize("FF&E", {"FF&E": SOFT})
    assert (r.category, r.tier) == (SOFT, "approved_budget")


def test_mixed_signals_go_to_review(cat):
    r = cat.categorize("Land and sitework")
    assert r.needs_review and r.tier == "review"


def test_unknown_and_empty(cat):
    assert cat.categorize("zzqx blorp").needs_review
    r = cat.categorize("  $1,000 ")
    assert r.needs_review and r.category == OTHER


def test_learn_feeds_dictionary():
    c = LineItemCategorizer()
    assert c.categorize("zzqx blorp").needs_review
    c.learn("ZZQX Blorp", HARD)
    r = c.categorize("zzqx blorp")
    assert (r.category, r.tier, r.needs_review) == (HARD, "dictionary", False)
    assert c.export_confirmed() == {"zzqx blorp": HARD}


def test_config_override():
    c = LineItemCategorizer(config={"rules": {SOFT: ["zzqx"]}})
    assert c.categorize("zzqx").category == SOFT


def test_bad_category_rejected():
    with pytest.raises(ValueError):
        LineItemCategorizer(confirmed={"x": "Medium Cost"})


TRAIN = (
    [(t, HARD) for t in ["scaffold rental", "crane hire", "rebar supply", "tower crane",
                         "scaffolding erect", "rebar install", "hoist rental"]] * 3
    + [(t, SOFT) for t in ["zoning counsel", "entitlement counsel", "lobbyist retainer",
                           "zoning lawyer retainer", "entitlement advisor"]] * 3
)


def _build(c=None):
    c = c or LineItemCategorizer()
    return CategoryModel.train(TRAIN, c.normalize, c.normalization_fingerprint())


def test_ml_build_save_load_reuse(tmp_path):
    path = _build().save(tmp_path / "m.joblib")
    model = CategoryModel.load(path)                    # load once
    assert model.meta["rows"] == len(TRAIN)
    assert model.meta["holdout_metrics"]["accuracy"] == 1.0
    for c in (LineItemCategorizer(ml_model=model), LineItemCategorizer(ml_model=path)):
        r = c.categorize("crane and scaffold hire")
        assert (r.category, r.tier) == (HARD, "ml") and r.confidence >= 0.8


def test_ml_fingerprint_mismatch_warns(caplog):
    model = _build()
    with caplog.at_level("WARNING"):
        LineItemCategorizer(ml_model=model)
    assert not caplog.records
    with caplog.at_level("WARNING"):
        LineItemCategorizer(config={"abbreviations": {"scaf": "scaffold"}}, ml_model=model)
    assert "rebuild the model" in caplog.text


def test_ml_sklearn_version_guard(tmp_path):
    m = _build()
    m.meta["sklearn_version"] = "0.0.1"
    path = m.save(tmp_path / "old.joblib")
    with pytest.raises(RuntimeError):
        CategoryModel.load(path)
    assert CategoryModel.load(path, strict_version=False).classes


def test_ml_cli(tmp_path, capsys):
    import csv
    data = tmp_path / "labeled.csv"
    with open(data, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["description", "category"])
        w.writerows(TRAIN)
    out = tmp_path / "cli.joblib"
    assert model_cli(["train", "--data", str(data), "--out", str(out)]) == 0
    assert model_cli(["info", str(out)]) == 0
    assert '"rows": 36' in capsys.readouterr().out
    with open(data, "a", newline="") as fh:
        csv.writer(fh).writerow(["thing", "Medium Cost"])
    with pytest.raises(SystemExit):
        model_cli(["train", "--data", str(data), "--out", str(out)])


# ---- Jev tier, exercised with a stand-in SDK (no network) ----------------
class _Ans:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeJev:
    def __init__(self, choice, confidence, mixed=False, boom=False):
        self.choice, self.confidence, self.mixed, self.boom = choice, confidence, mixed, boom
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append(state)
        if self.boom:
            raise RuntimeError("503")
        return _Ans(answers={"category": _Ans(choice=self.choice, confidence=self.confidence),
                             "consolidated": _Ans(value=self.mixed)})


@pytest.fixture
def fake_sdk(monkeypatch):
    mod = types.ModuleType("typesafe_sdk")
    mod.Choice = lambda **kw: ("choice", kw)
    mod.Noul = lambda **kw: ("noul", kw)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", mod)


def test_jev_confident(fake_sdk):
    jev = FakeJev(SOFT, 0.93)
    r = LineItemCategorizer(jev_client=jev).categorize("zzqx blorp $5,000")
    assert (r.category, r.tier, r.needs_review) == (SOFT, "jev", False)
    assert jev.calls == [{"line_item": "zzqx blorp"}]      # amount stripped before sending


def test_jev_not_called_when_rules_decide(fake_sdk):
    jev = FakeJev(SOFT, 0.99)
    assert LineItemCategorizer(jev_client=jev).categorize("Roofing").tier == "rules"
    assert jev.calls == []


def test_jev_low_confidence_mixed_and_failure(fake_sdk):
    r = LineItemCategorizer(jev_client=FakeJev(HARD, 0.6)).categorize("zzqx blorp")
    assert r.needs_review and r.category == HARD
    r = LineItemCategorizer(jev_client=FakeJev(HARD, 0.99, mixed=True)).categorize("zzqx blorp")
    assert r.needs_review
    r = LineItemCategorizer(jev_client=FakeJev(HARD, 0.99, boom=True)).categorize("zzqx blorp")
    assert r.needs_review and r.category == OTHER
