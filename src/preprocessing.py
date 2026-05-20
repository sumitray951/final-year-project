import re
from typing import Iterable


MIN_CLAUSE_CHARS = 12


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.strip()


def split_long_line(line: str) -> list[str]:
    parts = re.split(r"(?<=[.;:!?])\s+", line)
    return [clean_text(part) for part in parts if clean_text(part)]


def split_into_clauses(text: str, min_chars: int = MIN_CLAUSE_CHARS) -> list[str]:
    raw_text = text.replace("\u00a0", " ")
    if not raw_text.strip():
        return []

    raw_lines = [clean_text(line) for line in raw_text.splitlines()]
    raw_parts: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        if len(line) > 180:
            raw_parts.extend(split_long_line(line))
        else:
            raw_parts.append(line)

    clauses: list[str] = []
    buffer = ""

    for part in raw_parts:
        part = clean_text(part)
        if not part:
            continue
        if len(part) < min_chars:
            if clauses:
                clauses[-1] = f"{clauses[-1]} {part}".strip()
            else:
                buffer = f"{buffer} {part}".strip()
            continue
        if buffer:
            part = f"{buffer} {part}".strip()
            buffer = ""
        clauses.append(part)

    if buffer and clauses:
        clauses[-1] = f"{clauses[-1]} {buffer}".strip()
    elif buffer:
        clauses.append(buffer)

    return clauses


HIGH_RISK_PATTERNS = {
    "sell your data": 4,
    "sell personal information": 4,
    "share your personal information with advertisers": 4,
    "third parties for marketing": 4,
    "no liability": 4,
    "not liable": 3,
    "binding arbitration": 4,
    "waive your rights": 4,
    "class action waiver": 4,
    "irrevocable license": 4,
    "perpetual license": 4,
    "royalty-free license": 3,
    "without notice": 3,
    "at our sole discretion": 3,
    "terminate your account at any time": 3,
    "we may change these terms at any time": 3,
    "location data": 3,
    "precise location": 3,
    "biometric": 4,
    "sensitive personal information": 4,
    "16 hours": 4,
    "sixteen hours": 4,
    "working 16 hours": 4,
    "work 16 hours": 4,
    "working for 16 hours": 4,
}

MEDIUM_RISK_PATTERNS = {
    "cookies": 2,
    "tracking": 2,
    "third party": 2,
    "third-party": 2,
    "advertising": 2,
    "analytics": 2,
    "share information": 2,
    "service providers": 1,
    "collect information": 2,
    "usage data": 2,
    "device information": 2,
    "ip address": 2,
    "personalized ads": 2,
    "marketing communications": 2,
    "retain your information": 2,
    "legal requirements": 1,
}

LOW_RISK_PATTERNS = {
    "gdpr": 2,
    "right to delete": 3,
    "delete your data": 3,
    "opt-in consent": 3,
    "encryption": 2,
    "data minimization": 3,
    "access your data": 2,
    "withdraw consent": 3,
    "opt out": 2,
    "do not sell": 3,
    "we do not sell": 3,
    "two-factor authentication": 2,
    "limited retention": 2,
    "anonymized": 2,
    "privacy settings": 2,
}


def matched_patterns(text: str, patterns: dict[str, int]) -> list[tuple[str, int]]:
    lowered = text.lower()
    return [(pattern, weight) for pattern, weight in patterns.items() if pattern in lowered]


def keyword_risk_analysis(text: str) -> dict[str, object]:
    high_matches = matched_patterns(text, HIGH_RISK_PATTERNS)
    medium_matches = matched_patterns(text, MEDIUM_RISK_PATTERNS)
    low_matches = matched_patterns(text, LOW_RISK_PATTERNS)

    high_score = sum(weight for _, weight in high_matches)
    medium_score = sum(weight for _, weight in medium_matches)
    low_score = sum(weight for _, weight in low_matches)

    if high_score >= 3 or high_score > low_score + medium_score:
        label = 2
        matches = high_matches
        reason = "High-risk clause signal"
        confidence = min(0.95, 0.62 + high_score * 0.06)
    elif low_score >= 2 and high_score == 0 and medium_score <= 1:
        label = 0
        matches = low_matches
        reason = "User-protection or privacy-control signal"
        confidence = min(0.9, 0.58 + low_score * 0.06)
    elif medium_score > 0 or high_score > 0 or low_score > 0:
        label = 1
        matches = medium_matches or high_matches or low_matches
        reason = "Some data use or policy-risk signal"
        confidence = min(0.82, 0.55 + (medium_score + high_score + low_score) * 0.04)
    else:
        label = 0
        matches = []
        reason = "No risky keyword found"
        confidence = 0.55

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "reason": reason,
        "matches": [pattern for pattern, _ in matches[:3]],
    }


def keyword_risk_label(text: str) -> int:
    return int(keyword_risk_analysis(text)["label"])


def dedupe_clauses(clauses: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for clause in clauses:
        normalized = clean_text(clause).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(clean_text(clause))
    return unique
