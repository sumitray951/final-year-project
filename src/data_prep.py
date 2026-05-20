import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing import clean_text, dedupe_clauses, keyword_risk_label
from src.seed_data import SEED_ROWS


HIGH_RISK_CUAD_CATEGORIES = {
    "anti-assignment",
    "audit rights",
    "covenant not to sue",
    "exclusivity",
    "governing law",
    "insurance",
    "ip ownership assignment",
    "irrevocable or perpetual license",
    "liquidated damages",
    "non-compete",
    "non-disparagement",
    "non-solicit of customers",
    "non-solicit of employees",
    "post-termination services",
    "termination for convenience",
    "uncapped liability",
    "warranty duration",
}

MEDIUM_RISK_CUAD_CATEGORIES = {
    "affiliate license-licensee",
    "affiliate license-licensor",
    "assignment",
    "change of control",
    "confidentiality obligation",
    "expiration date",
    "license grant",
    "minimum commitment",
    "notice period to terminate renewal",
    "renewal term",
    "revenue-profit sharing",
    "source code escrow",
}


def normalize_category(category: str) -> str:
    return category.replace("_", " ").replace("-", " ").strip().lower()


def cuad_category_to_label(category: str, answer: str) -> int:
    normalized = normalize_category(category)
    if normalized in HIGH_RISK_CUAD_CATEGORIES:
        return 2
    if normalized in MEDIUM_RISK_CUAD_CATEGORIES:
        return 1
    return keyword_risk_label(answer)


def load_cuad(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows: list[dict[str, Any]] = []
    for doc in payload.get("data", []):
        title = doc.get("title", "")
        for paragraph in doc.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                category = qa.get("question", qa.get("id", "unknown"))
                for answer in qa.get("answers", []):
                    text = clean_text(answer.get("text", ""))
                    if len(text) < 30:
                        continue
                    rows.append(
                        {
                            "text": text,
                            "label": cuad_category_to_label(category, text),
                            "source": "CUAD",
                            "category": normalize_category(category),
                            "document": title,
                        }
                    )

    return pd.DataFrame(rows)


def tosdr_rating_to_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"good", "good point", "positive", "low", "0", "a", "b"}:
        return 0
    if text in {"warn", "warning", "neutral", "medium", "1", "c"}:
        return 1
    if text in {"bad", "blocker", "negative", "high", "2", "d", "e"}:
        return 2
    return 1


def text_from_tosdr_record(record: dict[str, Any]) -> str:
    for key in ("quoteText", "quote", "clause", "text", "title", "point"):
        value = record.get(key)
        if value:
            return clean_text(str(value))
    return ""


def rating_from_tosdr_record(record: dict[str, Any]) -> Any:
    for key in ("caseClassification", "classification", "rating", "class", "label", "score"):
        if key in record and record[key] not in (None, ""):
            return record[key]
    return "medium"


def flatten_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        records: list[dict[str, Any]] = []
        for value in payload.values():
            records.extend(flatten_json_records(value))
        if any(key in payload for key in ("quoteText", "quote", "clause", "text", "title")):
            records.append(payload)
        return records
    return []


def load_tosdr(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted(path.glob("*"))
    rows: list[dict[str, Any]] = []

    for file in files:
        if file.suffix.lower() == ".json":
            with file.open("r", encoding="utf-8") as handle:
                records = flatten_json_records(json.load(handle))
            for record in records:
                text = text_from_tosdr_record(record)
                if len(text) < 30:
                    continue
                rows.append(
                    {
                        "text": text,
                        "label": tosdr_rating_to_label(rating_from_tosdr_record(record)),
                        "source": "ToS;DR",
                        "category": str(record.get("topic", record.get("case", ""))).lower(),
                        "document": str(record.get("service", record.get("serviceName", ""))),
                    }
                )
        elif file.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if file.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(file, sep=sep)
            text_col = next((col for col in df.columns if col.lower() in {"quotetext", "quote", "clause", "text", "title"}), None)
            label_col = next((col for col in df.columns if col.lower() in {"caseclassification", "classification", "rating", "class", "label", "score"}), None)
            if not text_col:
                continue
            for _, row in df.iterrows():
                text = clean_text(str(row[text_col]))
                if len(text) < 30:
                    continue
                rows.append(
                    {
                        "text": text,
                        "label": tosdr_rating_to_label(row[label_col] if label_col else "medium"),
                        "source": "ToS;DR",
                        "category": "",
                        "document": "",
                    }
                )

    return pd.DataFrame(rows)


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df["label"].value_counts()
    target = counts.max()
    balanced = [
        group.sample(target, replace=len(group) < target, random_state=42)
        for _, group in df.groupby("label")
    ]
    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)


def load_seed_data() -> pd.DataFrame:
    return pd.DataFrame(SEED_ROWS)


def build_dataset(
    cuad_path: Path,
    tosdr_path: Path,
    output_path: Path,
    balance: bool,
    include_seed: bool = True,
) -> pd.DataFrame:
    frames = []
    if include_seed:
        frames.append(load_seed_data())
    if cuad_path.exists():
        frames.append(load_cuad(cuad_path))
    if tosdr_path.exists():
        frames.append(load_tosdr(tosdr_path))
    if not frames:
        raise FileNotFoundError("No dataset files were found.")

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].isin(dedupe_clauses(df["text"]))]
    df = df.drop_duplicates(subset=["text"]).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int).clip(0, 2)
    if balance:
        df = balance_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LegalBERT risk dataset from CUAD and ToS;DR.")
    parser.add_argument("--cuad", type=Path, default=Path("data/raw/cuad/CUAD_v1.json"))
    parser.add_argument("--tosdr", type=Path, default=Path("data/raw/tosdr"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/clauses.csv"))
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--no-seed", action="store_true", help="Do not include the built-in starter training data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(
        args.cuad,
        args.tosdr,
        args.output,
        balance=not args.no_balance,
        include_seed=not args.no_seed,
    )
    print(f"Saved {len(df)} clauses to {args.output}")
    print(df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
