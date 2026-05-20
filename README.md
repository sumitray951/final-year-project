# AI-Driven Terms Checker for Privacy Risk Detection

This project fine-tunes LegalBERT to classify Terms of Service and privacy-policy clauses as:

- `0` Low risk
- `1` Medium risk
- `2` High risk

It supports the project flow from the presentation: extract text from PDF/image/plain text, split into clauses, classify each clause, show pros/cons, and produce an Accept / Caution / Reject recommendation in Streamlit.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── train.py
└── data/
    ├── raw/
    │   ├── cuad/
    │   └── tosdr/
    └── processed/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Add Datasets

Place the datasets locally:

```text
data/raw/cuad/
  CUAD_v1.json

data/raw/tosdr/
  *.json, *.csv, or *.tsv
```

The CUAD loader expects SQuAD-style JSON. The ToS;DR loader accepts common exports containing clause/title/text and rating/class fields.

## Prepare Combined Dataset

```bash
python -m src.data_prep `
  --cuad data/raw/cuad/CUAD_v1.json `
  --tosdr data/raw/tosdr `
  --output data/processed/clauses.csv
```

If CUAD and ToS;DR are not added yet, the script still creates a balanced starter dataset from built-in seed clauses:

```bash
python -m src.data_prep --output data/processed/clauses.csv
```

## Fine-Tune LegalBERT

```bash
python -m src.train `
  --data data/processed/clauses.csv `
  --model-name nlpaueb/legal-bert-base-uncased `
  --output-dir models/legalbert-risk `
  --epochs 4 `
  --batch-size 8
```

## Run App

```bash
streamlit run app.py
```

If a fine-tuned model exists at `models/legalbert-risk`, the app uses it. Otherwise it falls back to a keyword baseline so the UI remains demoable.
