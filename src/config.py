from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA = DATA_DIR / "processed" / "clauses.csv"
DEFAULT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
FINETUNED_MODEL_DIR = BASE_DIR / "models" / "legalbert-risk"


def ensure_model_dirs() -> None:
    """Create the model directory structure expected by training/inference."""
    (BASE_DIR / "models").mkdir(parents=True, exist_ok=True)
    FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Create required directories on import so app startup never fails due to missing folders.
ensure_model_dirs()


LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
}

LABEL_TO_ID = {name.lower(): label_id for label_id, name in LABELS.items()}

RISK_COLORS = {
    0: "#1f9d55",
    1: "#b7791f",
    2: "#c53030",
}
