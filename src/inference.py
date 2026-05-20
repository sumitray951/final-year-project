from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import FINETUNED_MODEL_DIR, LABELS
from src.preprocessing import keyword_risk_analysis


class RiskClassifier:
    def __init__(self, model_dir: Path = FINETUNED_MODEL_DIR):
        self.model_dir = Path(model_dir)
        # Only treat the model directory as “available” if it contains real fine-tuned artifacts.
        # We consider it available when HuggingFace can load from it.
        self.available = self.model_dir.exists()
        self.tokenizer: Any | None = None
        self.model: Any | None = None

        if self.available:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.model.eval()
                self.available = True
            except Exception:
                # Fall back to keyword baseline if tokenizer/model artifacts are missing or incompatible.
                self.tokenizer = None
                self.model = None
                self.available = False


    def predict_one(self, text: str) -> dict[str, Any]:
        if not self.available or self.tokenizer is None or self.model is None:
            analysis = keyword_risk_analysis(text)
            label_id = int(analysis["label"])
            return {
                "text": text,
                "label": label_id,
                "risk": LABELS[label_id],
                "confidence": analysis["confidence"],
                "reason": analysis["reason"],
                "matches": analysis["matches"],
                "model": "smart-rule-baseline",
            }

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        label_id = int(torch.argmax(probs).item())
        return {
            "text": text,
            "label": label_id,
            "risk": LABELS[label_id],
            "confidence": float(probs[label_id].item()),
            "reason": "Fine-tuned LegalBERT prediction",
            "matches": [],
            "model": "fine-tuned-legalbert",
        }

    def predict(self, clauses: list[str]) -> list[dict[str, Any]]:
        return [self.predict_one(clause) for clause in clauses]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"score": 0, "decision": "No content", "pros": [], "cons": []}

    weighted = sum(item["label"] for item in results)
    score = round((weighted / (2 * len(results))) * 100)
    high_count = sum(item["label"] == 2 for item in results)

    if score >= 60 or high_count >= 3:
        decision = "Reject"
    elif score >= 30 or high_count:
        decision = "Accept with Caution"
    else:
        decision = "Accept"

    pros = [item["text"] for item in results if item["label"] == 0][:5]
    cons = [item["text"] for item in results if item["label"] == 2][:5]
    return {"score": score, "decision": decision, "pros": pros, "cons": cons}
