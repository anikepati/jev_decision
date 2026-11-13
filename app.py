"""
app.py - FastAPI service for line-item categorization.

Flow for every line item
------------------------
    1. Look in the loan's category JSON file. If the line item is there
       (exact or close match), return that category.
    2. Otherwise ask the ML model (built once from Excel, loaded once at startup).
    3. If the model is not confident, return its best guess with needs_review=true.

Build the model once:
    python category_model.py train --data labeled.xlsx --out models/category_model.joblib

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Settings (environment variables)
--------------------------------
    MODEL_PATH          models/category_model.joblib
    LOAN_CATEGORY_DIR   loan_categories        one file per loan: <loan_id>.json
    CATEGORIZER_TIERS   approved_budget,ml     add "rules", "dictionary", "jev" to enable them
    CATEGORIZER_CONFIG  (unset)                optional JSON file of pipeline overrides
    ML_THRESHOLD        (unset)                overrides thresholds.ml, e.g. 0.75

Loan category JSON - any of these shapes works:
    {"Architect & Engineering": "Soft Cost", "Building Construction": "Hard Cost"}
    [{"line_item": "Architect & Engineering", "category": "Soft Cost"}, ...]
    {"loan_id": "L-1001", "line_items": [{"line_item": "...", "category": "..."}]}
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from category_model import CategoryModel
from line_item_categorizer import CategoryResult, LineItemCategorizer, canonical_category

log = logging.getLogger("categorizer.api")

_LOAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_ITEM_KEYS = ("line_item", "lineitem", "line item", "description", "name", "item")
_LIST_KEYS = ("line_items", "lineitems", "items", "categories", "budget")


# --------------------------------------------------------------------------
# Loan category files
# --------------------------------------------------------------------------
def parse_loan_categories(data: Any) -> dict[str, str]:
    """Flatten any supported JSON shape into {line item: canonical category}."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower().replace(" ", "_") in _LIST_KEYS and isinstance(value, list):
                return parse_loan_categories(value)
        return {str(k): c for k, v in data.items()
                if isinstance(v, str) and (c := canonical_category(v))}
    if isinstance(data, list):
        out: dict[str, str] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            low = {str(k).lower().replace("_", " ").strip(): v for k, v in row.items()}
            desc = next((low[k.replace("_", " ")] for k in _ITEM_KEYS if low.get(k.replace("_", " "))), None)
            cat = canonical_category(low.get("category"))
            if desc and cat:
                out[str(desc)] = cat
        return out
    return {}


class LoanCategoryStore:
    """Reads <dir>/<loan_id>.json, cached and re-read only when the file changes."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self._cache: dict[str, tuple[float, dict[str, str]]] = {}
        self._lock = threading.Lock()

    def get(self, loan_id: Optional[str]) -> Optional[dict[str, str]]:
        """None means 'no file for this loan' (not an error: ML takes over)."""
        if not loan_id:
            return None
        if not _LOAN_ID.match(loan_id):
            raise HTTPException(422, "loan_id may contain only letters, digits, '.', '_' and '-'")
        path = self.dir / f"{loan_id}.json"
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            cached = self._cache.get(loan_id)
            if cached and cached[0] == mtime:
                return cached[1]
        try:
            parsed = parse_loan_categories(json.loads(path.read_text(encoding="utf-8-sig")))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.error("Loan category file %s is not valid JSON: %s", path, exc)
            raise HTTPException(500, f"Loan category file for {loan_id} is not valid JSON")
        with self._lock:
            self._cache[loan_id] = (mtime, parsed)
        return parsed


# --------------------------------------------------------------------------
# Startup: build everything once
# --------------------------------------------------------------------------
def create_state() -> dict[str, Any]:
    model_path = Path(os.getenv("MODEL_PATH", "models/category_model.joblib"))
    loan_dir = os.getenv("LOAN_CATEGORY_DIR", "loan_categories")
    tiers = [t.strip() for t in os.getenv("CATEGORIZER_TIERS", "approved_budget,ml").split(",") if t.strip()]

    config: dict[str, Any] = {}
    if cfg_path := os.getenv("CATEGORIZER_CONFIG"):
        config = json.loads(Path(cfg_path).read_text())
    config["enabled_tiers"] = tiers
    if th := os.getenv("ML_THRESHOLD"):
        config.setdefault("thresholds", {})["ml"] = float(th)

    model = None
    if model_path.exists():
        model = CategoryModel.load(model_path)            # loaded exactly once
        log.info("Loaded ML model %s (%s rows, trained %s)",
                 model_path, model.meta.get("rows"), model.meta.get("trained_at"))
    else:
        log.warning("No ML model at %s - only loan category files will be used. "
                    "Build one with: python category_model.py train --data <excel> --out %s",
                    model_path, model_path)

    return {
        "categorizer": LineItemCategorizer(config=config, ml_model=model),
        "model": model,
        "loans": LoanCategoryStore(loan_dir),
        "model_path": str(model_path),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = create_state()
    yield


app = FastAPI(title="Line Item Categorizer", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class CategorizeRequest(BaseModel):
    line_item: str = Field(..., min_length=1, max_length=500, examples=["Arch & Eng fees - bldg A"])
    loan_id: Optional[str] = Field(None, examples=["L-1001"])


class BatchRequest(BaseModel):
    line_items: list[str] = Field(..., min_length=1, max_length=2000)
    loan_id: Optional[str] = None


class CategorizeResponse(BaseModel):
    line_item: str
    category: str
    source: str            # "loan_file" | "ml" | "rules" | "dictionary" | "jev" | "review"
    confidence: float
    needs_review: bool
    reason: str
    loan_file_found: bool


class BatchResponse(BaseModel):
    loan_id: Optional[str]
    loan_file_found: bool
    results: list[CategorizeResponse]


def _to_response(r: CategoryResult, loan_file_found: bool) -> CategorizeResponse:
    return CategorizeResponse(
        line_item=r.line_item,
        category=r.category,
        source="loan_file" if r.tier == "approved_budget" else r.tier,
        confidence=r.confidence,
        needs_review=r.needs_review,
        reason=r.reason,
        loan_file_found=loan_file_found,
    )


# --------------------------------------------------------------------------
# Endpoints (plain `def`: FastAPI runs them in a worker thread)
# --------------------------------------------------------------------------
@app.post("/categorize", response_model=CategorizeResponse)
def categorize(req: CategorizeRequest, request: Request) -> CategorizeResponse:
    ctx = request.app.state.ctx
    loan = ctx["loans"].get(req.loan_id)
    result = ctx["categorizer"].categorize(req.line_item, loan)
    return _to_response(result, loan is not None)


@app.post("/categorize/batch", response_model=BatchResponse)
def categorize_batch(req: BatchRequest, request: Request) -> BatchResponse:
    ctx = request.app.state.ctx
    loan = ctx["loans"].get(req.loan_id)
    found = loan is not None
    results = [_to_response(ctx["categorizer"].categorize(li, loan), found) for li in req.line_items]
    return BatchResponse(loan_id=req.loan_id, loan_file_found=found, results=results)


@app.get("/health")
def health(request: Request) -> dict:
    ctx = request.app.state.ctx
    model = ctx["model"]
    return {
        "status": "ok",
        "tiers": ctx["categorizer"].enabled_tiers,
        "ml_model_loaded": model is not None,
        "ml_model_path": ctx["model_path"],
        "ml_model": model.meta if model else None,
    }
