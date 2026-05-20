import argparse
import inspect
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.config import DEFAULT_MODEL_NAME, LABELS
from src.data_prep import build_dataset


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            device = logits.device
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "macro_f1": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }


def train(args: argparse.Namespace) -> None:
    if not args.data.exists():
        print(f"{args.data} not found. Creating starter dataset with built-in seed clauses.")
        build_dataset(
            cuad_path=Path("data/raw/cuad/CUAD_v1.json"),
            tosdr_path=Path("data/raw/tosdr"),
            output_path=args.data,
            balance=True,
            include_seed=True,
        )

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    # 1. Deduplicate by text to avoid data leakage between train/val splits
    initial_len = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Deduplicated dataset: reduced from {initial_len} to {len(df)} unique clauses.")

    if 0 < args.sample_frac < 1:
        # Safe stratified sampling ensuring at least 1 sample per class
        sampled_dfs = []
        for label, group in df.groupby("label"):
            n_samples = max(1, int(len(group) * args.sample_frac))
            sampled_dfs.append(group.sample(n=n_samples, random_state=42))
        df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Using sampled dataset: {len(df)} rows ({args.sample_frac:.0%} fraction).")

    train_df, eval_df = train_test_split(
        df[["text", "label"]],
        test_size=args.test_size,
        stratify=df["label"],
        random_state=42,
    )
    print(f"Original train size: {len(train_df)} | Eval size: {len(eval_df)}")

    # 2. Balance only the training set using hybrid resampling (undersample majority, mild oversample minority)
    def balance_train_dataset(train_data: pd.DataFrame, max_majority_size: int, min_minority_size: int = 300) -> pd.DataFrame:
        if train_data.empty:
            return train_data
        groups = {label: group for label, group in train_data.groupby("label")}
        
        majority_label = 0
        balanced_groups = []
        for label, group in groups.items():
            if label == majority_label:
                n_samples = min(len(group), max_majority_size)
                balanced_groups.append(group.sample(n=n_samples, random_state=42))
            else:
                if len(group) < min_minority_size:
                    balanced_groups.append(group.sample(n=min_minority_size, replace=True, random_state=42))
                else:
                    balanced_groups.append(group)
                    
        return pd.concat(balanced_groups).sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = balance_train_dataset(train_df, args.max_majority_size)
    print(f"Balanced train size (with mild minority oversampling): {len(train_df)}")
    print("Train label distribution:")
    print(train_df["label"].value_counts().to_string())
    print("Eval label distribution:")
    print(eval_df["label"].value_counts().to_string())

    # Compute class weights on the resampled training set
    y_train = train_df["label"].values
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    print(f"Computed class weights for WeightedLossTrainer: {class_weights}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=LABELS,
        label2id={name: idx for idx, name in LABELS.items()},
    )

    train_dataset = tokenize_dataset(Dataset.from_pandas(train_df), tokenizer, args.max_length)
    eval_dataset = tokenize_dataset(Dataset.from_pandas(eval_df), tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.1,  # Increased weight decay to prevent overfitting on oversampled training data
        "lr_scheduler_type": "cosine",  # Cosine schedule with learning rate decay
        "warmup_ratio": 0.1,  # Warm up learning rate for the first 10% of steps
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 25,
        "report_to": "none",
    }
    training_arg_names = inspect.signature(TrainingArguments).parameters
    if "evaluation_strategy" in training_arg_names:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],  # Add early stopping callback
    }
    trainer_arg_names = inspect.signature(Trainer).parameters
    if "tokenizer" in trainer_arg_names:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_arg_names:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = WeightedLossTrainer(class_weights=class_weights, **trainer_kwargs)
    trainer.train()
    trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LegalBERT for clause risk classification.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/clauses.csv"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("models/legalbert-risk"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)  # Set default to 256 for better legal clause coverage
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Use a stratified fraction of the dataset, e.g. 0.5 for half.",
    )
    parser.add_argument(
        "--max-majority-size",
        type=int,
        default=2000,
        help="Capping majority class size in the training set for hybrid resampling to prevent long training times.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
