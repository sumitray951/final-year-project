from dataclasses import dataclass
from html import escape
from io import BytesIO

import pdfplumber
import streamlit as st
import numpy as np
from PIL import Image

from src.config import LABELS, RISK_COLORS
from src.inference import RiskClassifier, summarize_results
from src.preprocessing import split_into_clauses


st.set_page_config(page_title="AI Terms Checker", page_icon="TC", layout="wide")


@dataclass
class InputText:
    text: str
    source: str
    extraction_method: str = "Plain text"
    warning: str = ""


@dataclass
class OcrText:
    text: str
    method: str
    warning: str = ""


@st.cache_resource
def get_classifier() -> RiskClassifier:
    return RiskClassifier()


def extract_pdf_text_with_pdfplumber(pdf_bytes: bytes) -> str:
    text_parts: list[str] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_image_text_from_pil(image: Image.Image) -> OcrText:
    easyocr_error = ""
    try:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        result = reader.readtext(np.array(image.convert("RGB")))
        text = "\n".join(item[1] for item in result)
        if text.strip():
            return OcrText(text=text, method="EasyOCR")
    except Exception as exc:
        easyocr_error = str(exc)

    try:
        import pytesseract

        text = pytesseract.image_to_string(image)
        if text.strip():
            return OcrText(text=text, method="Tesseract OCR")
        return OcrText(text="", method="OCR", warning="OCR ran, but no readable text was found in the image.")
    except Exception as exc:
        warning = (
            "Image OCR could not run. EasyOCR failed first, and Tesseract is not installed "
            "or not available in PATH. Install Tesseract OCR or use a clearer text/PDF input."
        )
        details = f" EasyOCR: {easyocr_error} Tesseract: {exc}"
        return OcrText(text="", method="OCR unavailable", warning=warning + details)


def extract_pdf_text(file) -> InputText:
    pdf_bytes = file.read()
    text = extract_pdf_text_with_pdfplumber(pdf_bytes)
    if len(text.strip()) >= 100:
        return InputText(text=text, source="PDF", extraction_method="PDF text layer")

    try:
        from pdf2image import convert_from_bytes

        pages = convert_from_bytes(pdf_bytes, dpi=250)
        page_results = [extract_image_text_from_pil(page) for page in pages]
        ocr_text = "\n".join(result.text for result in page_results)
        if ocr_text.strip():
            methods = sorted({result.method for result in page_results if result.text.strip()})
            method = " + ".join(methods) if methods else "OCR fallback"
            warnings = " ".join(result.warning for result in page_results if result.warning)
            return InputText(text=ocr_text, source="PDF", extraction_method=method, warning=warnings)
    except Exception as exc:
        warning = (
            "This PDF looks scanned or image-based, and OCR fallback could not run. "
            "Install Poppler for Windows and make sure Tesseract/EasyOCR is available."
        )
        return InputText(text=text, source="PDF", extraction_method="PDF text layer", warning=f"{warning} Details: {exc}")

    return InputText(
        text=text,
        source="PDF",
        extraction_method="PDF text layer",
        warning="Very little text was extracted. The PDF may be scanned or protected.",
    )


def extract_image_text(file) -> InputText:
    image = Image.open(file)
    result = extract_image_text_from_pil(image)
    return InputText(
        text=result.text,
        source="Image",
        extraction_method=result.method,
        warning=result.warning,
    )


def read_input_text() -> InputText:
    tab_text, tab_pdf, tab_image = st.tabs(["Text", "PDF", "Image"])
    with tab_text:
        pasted_text = st.text_area("Paste terms or privacy policy text", height=260)
    with tab_pdf:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        pdf_input = extract_pdf_text(pdf_file) if pdf_file else InputText("", "PDF")
    with tab_image:
        image_file = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"])
        image_input = extract_image_text(image_file) if image_file else InputText("", "Image")
    if pasted_text.strip():
        return InputText(text=pasted_text, source="Text")
    if pdf_input.text.strip():
        return pdf_input
    return image_input


def risk_sentence(item: dict) -> str:
    text = item["text"].strip()
    if text.endswith((".", "!", "?")):
        text = text[:-1]
    return f'Line {item["line_no"]}: "{text}" is {item["risk"]} Risk'


def risk_reason(item: dict) -> str:
    matches = item.get("matches") or []
    reason = str(item.get("reason", ""))
    if matches:
        return f"{reason}: {', '.join(matches)}"
    return reason


st.title("AI-Driven Terms Checker")
st.caption("LegalBERT clause risk detection for Terms of Service and privacy policies")

input_text = read_input_text()
text = input_text.text
if text.strip():
    st.info(
        f"Input source: {input_text.source} | Extraction: {input_text.extraction_method} | "
        f"Extracted characters: {len(text.strip())}"
    )
if input_text.warning:
    st.warning(input_text.warning)
classify = st.button("Analyze", type="primary", use_container_width=True)

if classify:
    clauses = split_into_clauses(text)
    if not clauses:
        st.warning("Add a longer document or clause to analyze.")
        st.stop()

    classifier = get_classifier()
    results = classifier.predict(clauses)
    for line_no, item in enumerate(results, start=1):
        item["line_no"] = line_no
    summary = summarize_results(results)

    low_count = sum(item["label"] == 0 for item in results)
    medium_count = sum(item["label"] == 1 for item in results)
    high_count = sum(item["label"] == 2 for item in results)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Decision", summary["decision"])
    metric_cols[1].metric("Risk Score", f"{summary['score']}/100")
    metric_cols[2].metric("Clauses", len(results))
    metric_cols[3].metric("Model", results[0]["model"])
    count_cols = st.columns(3)
    count_cols[0].metric("Low Risk", low_count)
    count_cols[1].metric("Medium Risk", medium_count)
    count_cols[2].metric("High Risk", high_count)

    st.divider()

    st.subheader("Low, Medium, and High Risk Lines")
    st.caption("These are the exact extracted lines with their risk level.")
    if results:
        for item in results:
            color = RISK_COLORS[item["label"]]
            safe_sentence = escape(risk_sentence(item))
            safe_reason = escape(risk_reason(item))
            st.markdown(
                f"""
                <div style="border-left: 6px solid {color}; padding: 0.9rem 1rem; margin: 0.6rem 0; background: #fffdf7;">
                    <p style="margin:0; font-size:1.05rem; font-weight:700; color:{color};">{safe_sentence}</p>
                    <p style="margin:0.35rem 0 0 0; color:#555;">Reason: {safe_reason} | Confidence: {item["confidence"]:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No lines found.")

    st.divider()

    filter_label = st.segmented_control(
        "Filter clauses",
        options=["All", "Low", "Medium", "High"],
        default="All",
    )

    for item in results:
        if filter_label != "All" and item["risk"] != filter_label:
            continue
        color = RISK_COLORS[item["label"]]
        safe_text = escape(item["text"])
        safe_reason = escape(risk_reason(item))
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding: 0.75rem 1rem; margin: 0.5rem 0; background: #fafafa;">
                <strong style="color:{color};">Line {item["line_no"]}: {item["risk"]} Risk</strong>
                <span style="color:#555;"> - confidence {item["confidence"]:.2f}</span>
                <p style="margin:0.5rem 0 0 0;">{safe_text}</p>
                <p style="margin:0.35rem 0 0 0; color:#555;">Reason: {safe_reason}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    col_pros, col_cons = st.columns(2)
    with col_pros:
        st.subheader("Pros")
        if summary["pros"]:
            for clause in summary["pros"]:
                st.success(clause)
        else:
            st.info("No clear low-risk clauses found.")
    with col_cons:
        st.subheader("Cons")
        if summary["cons"]:
            for clause in summary["cons"]:
                st.error(clause)
        else:
            st.info("No high-risk clauses found.")
