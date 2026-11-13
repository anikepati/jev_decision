"""Run: python -m pytest test_app.py -q"""
import json

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from category_model import CategoryModel, build_model, main as model_cli

LABELED = (
    [("Scaffold rental", "Hard"), ("Crane hire", "hard costs"), ("Rebar supply", "Hard Cost"),
     ("Tower crane", "Hard Cost"), ("Scaffolding erect", "Hard Cost"), ("Rebar install", "Hard Cost"),
     ("Hoist rental", "Hard Cost")] * 3
    + [("Zoning counsel", "Soft"), ("Entitlement counsel", "Soft Cost"), ("Lobbyist retainer", "soft costs"),
       ("Zoning lawyer retainer", "Soft Cost"), ("Entitlement advisor", "Soft Cost")] * 3
    + [("Parcel purchase", "Acquisition"), ("Parcel escrow deposit", "Acq"),
       ("Seller purchase price", "Acquisition"), ("Parcel transfer tax", "Acquisition")] * 3
)


def _excel(path, rows, headers=("Line Item", "Category")):
    wb = Workbook()
    ws = wb.active
    ws.append(list(headers))
    for r in rows:
        ws.append(list(r))
    wb.save(path)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    xlsx = _excel(tmp_path / "labeled.xlsx", LABELED)
    model_path = tmp_path / "model.joblib"
    assert model_cli(["train", "--data", str(xlsx), "--out", str(model_path)]) == 0

    loans = tmp_path / "loans"
    loans.mkdir()
    (loans / "L-1001.json").write_text(json.dumps({
        "Architect & Engineering": "Soft Cost",
        "Building Construction": "Hard Costs",
        "Crane hire": "Soft Cost",          # loan file must win over the model
    }))
    (loans / "L-2002.json").write_text(json.dumps({
        "loan_id": "L-2002",
        "line_items": [{"line_item": "Land Acquisition", "category": "acquisition", "amount": 5}],
    }))
    (loans / "BAD.json").write_text("{not json")

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("LOAN_CATEGORY_DIR", str(loans))
    monkeypatch.delenv("CATEGORIZER_TIERS", raising=False)
    return tmp_path


@pytest.fixture
def client(env):
    import app as app_module
    with TestClient(app_module.app) as c:
        yield c


def post(client, line_item, loan_id=None):
    r = client.post("/categorize", json={"line_item": line_item, "loan_id": loan_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_excel_build_metadata(env):
    m = CategoryModel.load(env / "model.joblib")
    assert m.meta["rows"] == len(LABELED)
    assert m.meta["source_file"] == "labeled.xlsx"
    assert set(m.classes) == {"Hard Cost", "Soft Cost", "Acquisition"}   # aliases canonicalised


def test_excel_custom_headers_and_bad_category(tmp_path):
    x = _excel(tmp_path / "a.xlsx", LABELED, headers=("Borrower Text", "Bucket"))
    with pytest.raises(ValueError, match="line item column"):
        build_model(x, tmp_path / "m.joblib")
    build_model(x, tmp_path / "m.joblib", line_item_col="Borrower Text", category_col="Bucket")
    y = _excel(tmp_path / "b.xlsx", LABELED + [("x", "Medium Cost")])
    with pytest.raises(ValueError, match="Medium Cost"):
        build_model(y, tmp_path / "m2.joblib")


def test_loan_file_exact_and_fuzzy(client):
    out = post(client, "Arch & Eng", "L-1001")
    assert (out["category"], out["source"], out["loan_file_found"]) == ("Soft Cost", "loan_file", True)
    out = post(client, "Bldg Constructon", "L-1001")
    assert (out["category"], out["source"]) == ("Hard Cost", "loan_file")


def test_loan_file_beats_model(client):
    assert post(client, "Crane hire", "L-1001")["category"] == "Soft Cost"
    assert post(client, "Crane hire")["category"] == "Hard Cost"


def test_falls_back_to_ml(client):
    out = post(client, "Tower crane and scaffold", "L-1001")     # not in the loan file
    assert (out["category"], out["source"], out["needs_review"]) == ("Hard Cost", "ml", False)
    out = post(client, "Zoning counsel retainer", "NO-SUCH-LOAN")
    assert (out["category"], out["source"], out["loan_file_found"]) == ("Soft Cost", "ml", False)


def test_nested_loan_shape(client):
    out = post(client, "Land acquisition", "L-2002")
    assert (out["category"], out["source"]) == ("Acquisition", "loan_file")


def test_low_confidence_is_flagged(client):
    out = post(client, "qqqq wwww")
    assert out["needs_review"] and out["source"] == "review"


def test_batch(client):
    r = client.post("/categorize/batch", json={
        "loan_id": "L-1001", "line_items": ["Architect and engineering", "Rebar supply"]})
    body = r.json()
    assert r.status_code == 200 and body["loan_file_found"]
    assert [x["source"] for x in body["results"]] == ["loan_file", "ml"]


def test_validation_and_errors(client):
    assert client.post("/categorize", json={"line_item": ""}).status_code == 422
    assert client.post("/categorize", json={"line_item": "x", "loan_id": "../../etc/passwd"}).status_code == 422
    assert client.post("/categorize", json={"line_item": "x", "loan_id": "BAD"}).status_code == 500


def test_loan_file_edit_is_picked_up(client, env):
    import os
    assert post(client, "Rebar supply", "L-1001")["source"] == "ml"
    p = env / "loans" / "L-1001.json"
    p.write_text(json.dumps({"Rebar supply": "Others"}))
    os.utime(p, (1, 1))                       # force a different mtime
    out = post(client, "Rebar supply", "L-1001")
    assert (out["category"], out["source"]) == ("Others", "loan_file")


def test_health(client):
    h = client.get("/health").json()
    assert h["ml_model_loaded"] and h["tiers"] == ["approved_budget", "ml"]
    assert h["ml_model"]["rows"] == len(LABELED)


def test_starts_without_model(env, monkeypatch):
    import app as app_module
    monkeypatch.setenv("MODEL_PATH", str(env / "missing.joblib"))
    with TestClient(app_module.app) as c:
        assert c.get("/health").json()["ml_model_loaded"] is False
        assert post(c, "Arch & Eng", "L-1001")["source"] == "loan_file"
        assert post(c, "Rebar supply")["needs_review"]


def test_rules_tier_can_be_enabled(env, monkeypatch):
    import app as app_module
    monkeypatch.setenv("CATEGORIZER_TIERS", "approved_budget,rules,ml")
    with TestClient(app_module.app) as c:
        assert post(c, "Roofing")["source"] == "rules"
