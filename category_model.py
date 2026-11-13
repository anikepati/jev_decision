"""
category_model.py
=================
The ML tier of the line-item categorizer, as its own build-once / load-once
artifact.

Build once (offline job, rerun only when labeled data or normalization changes):

    python category_model.py train --data labeled.xlsx --out category_model.joblib

    The Excel (or CSV) file needs a header row with a "Line Item" column and
    a "Category" column.

Load once at application startup:

    from category_model import CategoryModel
    from line_item_categorizer import LineItemCategorizer

    MODEL = CategoryModel.load("category_model.joblib")
    CATEGORIZER = LineItemCategorizer(ml_model=MODEL)

    CATEGORIZER.categorize("Arch & Eng fees - bldg A").category

The saved file carries metadata (training date, row count, scikit-learn
version, hold-out metrics, normalization fingerprint) so a stale or
mismatched model is detected at load time instead of silently misbehaving.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

log = logging.getLogger(__name__)

FORMAT_VERSION = 1


class CategoryModel:
    """TF-IDF (word + character n-grams) + logistic regression.

    The model always sees *normalized* text. Pass the same normalizer at
    training time that the pipeline uses at prediction time
    (``LineItemCategorizer().normalize``); the fingerprint guards that.
    """

    def __init__(self, pipeline, meta: dict):
        self._pipeline = pipeline
        self.meta = meta

    # ---- build ---------------------------------------------------------
    @classmethod
    def train(
        cls,
        rows: Iterable[tuple[str, str]],
        normalizer: Callable[[str], str],
        fingerprint: str = "",
        holdout: float = 0.2,
        seed: int = 42,
    ) -> "CategoryModel":
        """
        rows        : (description, category) pairs from historical budgets
        normalizer  : text cleanup function, normally LineItemCategorizer().normalize
        fingerprint : LineItemCategorizer().normalization_fingerprint()
        holdout     : fraction held out to measure accuracy (0 disables).
                      The final model is refit on ALL rows afterwards.
        """
        import sklearn
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split

        X, y = [], []
        for desc, cat in rows:
            n = normalizer(desc)
            if n and cat:
                X.append(n)
                y.append(str(cat).strip())
        if len(set(y)) < 2:
            raise ValueError("Need training examples from at least two categories")

        metrics: Optional[dict] = None
        counts = {c: y.count(c) for c in set(y)}
        if holdout and min(counts.values()) >= 2 and len(X) * holdout >= len(counts):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=holdout, random_state=seed, stratify=y)
            probe = cls._new_pipeline().fit(Xtr, ytr)
            report = classification_report(yte, probe.predict(Xte), output_dict=True, zero_division=0)
            metrics = {
                "holdout_rows": len(Xte),
                "accuracy": round(report["accuracy"], 4),
                "per_category": {
                    c: {"precision": round(v["precision"], 4),
                        "recall": round(v["recall"], 4),
                        "support": int(v["support"])}
                    for c, v in report.items() if c in counts
                },
            }

        pipeline = cls._new_pipeline().fit(X, y)
        meta = {
            "format_version": FORMAT_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(X),
            "rows_per_category": counts,
            "sklearn_version": sklearn.__version__,
            "normalization_fingerprint": fingerprint,
            "holdout_metrics": metrics,
        }
        return cls(pipeline, meta)

    @staticmethod
    def _new_pipeline():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union

        return make_pipeline(
            make_union(
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
            ),
            LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"),
        )

    # ---- persist -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        joblib.dump({"pipeline": self._pipeline, "meta": self.meta}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, strict_version: bool = True) -> "CategoryModel":
        """Load a model file YOU produced (joblib uses pickle: never load an
        untrusted file). Raises if the scikit-learn version differs from the
        one used for training, unless strict_version=False."""
        import joblib
        import sklearn

        blob = joblib.load(path)
        if not isinstance(blob, dict) or "pipeline" not in blob or "meta" not in blob:
            raise ValueError(f"{path} is not a CategoryModel file")
        meta = blob["meta"]
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported model format {meta.get('format_version')!r}")
        trained_with = meta.get("sklearn_version")
        if trained_with != sklearn.__version__:
            msg = (f"Model was trained with scikit-learn {trained_with} but "
                   f"{sklearn.__version__} is installed; retrain or pin the version")
            if strict_version:
                raise RuntimeError(msg)
            log.warning(msg)
        return cls(blob["pipeline"], meta)

    # ---- use -----------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self._pipeline.classes_]

    def predict(self, normalized_text: str) -> tuple[str, float]:
        """Return (category, probability) for already-normalized text."""
        probs = self._pipeline.predict_proba([normalized_text])[0]
        i = int(probs.argmax())
        return self.classes[i], float(probs[i])

    def predict_proba(self, normalized_text: str) -> dict[str, float]:
        probs = self._pipeline.predict_proba([normalized_text])[0]
        return {c: float(p) for c, p in zip(self.classes, probs)}


# --------------------------------------------------------------------------
# CLI:  python category_model.py train --data labeled.csv --out model.joblib
#       python category_model.py info  model.joblib
# --------------------------------------------------------------------------
_DESC_HEADERS = ("line item", "lineitem", "line items", "description", "line item description", "item")
_CAT_HEADERS = ("category", "cost category", "line item category", "type")


def _norm_header(h) -> str:
    return " ".join(str(h or "").replace("_", " ").replace("-", " ").lower().split())


def _pick(headers: list[str], wanted: Optional[str], defaults: tuple[str, ...], what: str, src: str) -> int:
    normed = [_norm_header(h) for h in headers]
    for name in ([_norm_header(wanted)] if wanted else defaults):
        if name in normed:
            return normed.index(name)
    raise ValueError(
        f"{src}: could not find the {what} column. Headers found: {headers}. "
        f"Name it with --{what.replace(' ', '-')}-col.")


def read_labeled_file(
    path: str | Path,
    line_item_col: Optional[str] = None,
    category_col: Optional[str] = None,
    sheet: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Read (line item, category) rows from .xlsx/.xlsm or .csv.

    The first row must be a header. By default the columns are found by name
    (case-insensitive): "Line Item" / "Description" and "Category".
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb[sheet] if sheet else wb.worksheets[0]
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
        finally:
            wb.close()
    elif ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    else:
        raise ValueError(f"Unsupported file type {ext!r}; use .xlsx, .xlsm or .csv")

    rows = [r for r in rows if any(c not in (None, "") for c in r)]
    if len(rows) < 2:
        raise ValueError(f"{path}: needs a header row plus at least one data row")
    headers = ["" if h is None else str(h) for h in rows[0]]
    di = _pick(headers, line_item_col, _DESC_HEADERS, "line item", str(path))
    ci = _pick(headers, category_col, _CAT_HEADERS, "category", str(path))
    out = []
    for r in rows[1:]:
        desc = r[di] if di < len(r) else None
        cat = r[ci] if ci < len(r) else None
        if desc in (None, "") or cat in (None, ""):
            continue
        out.append((str(desc).strip(), str(cat).strip()))
    return out


def build_model(
    data_path: str | Path,
    out_path: str | Path,
    config: Optional[dict] = None,
    line_item_col: Optional[str] = None,
    category_col: Optional[str] = None,
    sheet: Optional[str] = None,
    holdout: float = 0.2,
) -> "CategoryModel":
    """Excel/CSV in, saved model out. This is the build-once entry point."""
    from line_item_categorizer import CATEGORIES, LineItemCategorizer, canonical_category

    pipeline = LineItemCategorizer(config=config)
    rows, bad = [], {}
    for desc, cat in read_labeled_file(data_path, line_item_col, category_col, sheet):
        canon = canonical_category(cat)
        if canon is None:
            bad[cat] = bad.get(cat, 0) + 1
        else:
            rows.append((desc, canon))
    if bad:
        raise ValueError(
            f"Unrecognised categories in {data_path}: {bad}. Expected {list(CATEGORIES)} "
            f"(variants like 'Hard', 'soft costs', 'Acq' are accepted).")
    model = CategoryModel.train(
        rows, pipeline.normalize, pipeline.normalization_fingerprint(), holdout=holdout)
    model.meta["source_file"] = Path(data_path).name
    model.save(out_path)
    return model


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build / inspect the line-item category model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train from a labeled Excel/CSV file and save")
    tr.add_argument("--data", required=True, help=".xlsx or .csv with line item and category columns")
    tr.add_argument("--out", required=True, help="output .joblib path")
    tr.add_argument("--config", help="same JSON config overrides the API/pipeline runs with")
    tr.add_argument("--line-item-col", help='header of the line item column (default: "Line Item" or "Description")')
    tr.add_argument("--category-col", help='header of the category column (default: "Category")')
    tr.add_argument("--sheet", help="worksheet name (default: first sheet)")
    tr.add_argument("--holdout", type=float, default=0.2)

    info = sub.add_parser("info", help="print a saved model's metadata")
    info.add_argument("model")

    args = ap.parse_args(argv)

    if args.cmd == "info":
        print(json.dumps(CategoryModel.load(args.model, strict_version=False).meta, indent=2))
        return 0

    cfg = json.loads(Path(args.config).read_text()) if args.config else None
    try:
        model = build_model(args.data, args.out, cfg, args.line_item_col,
                            args.category_col, args.sheet, args.holdout)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"Saved {args.out}")
    print(json.dumps(model.meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
