from dataclasses import dataclass
from html import escape
from io import BytesIO

import pdfplumber
import streamlit as st
import numpy as np
import altair as alt
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


# Custom premium CSS styling injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.8rem;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
    }
    
    .metric-card.low::before { background: #10b981; }
    .metric-card.medium::before { background: #f59e0b; }
    .metric-card.high::before { background: #ef4444; }
    .metric-card.info::before { background: #3b82f6; }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.2rem 0;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">AI-Driven Terms Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Fine-tuned LegalBERT risk analysis for Terms of Service and privacy policies</div>',
    unsafe_allow_html=True,
)

# Sidebar with configuration and legal guidance
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1450133064473-71024230f91b?auto=format&fit=crop&w=600&q=80",
        caption="AI-Driven Contract Intelligence",
        use_container_width=True
    )
    st.subheader("⚖️ Legal Risk Definition")
    st.markdown(
        """
        - 🟢 **Low Risk**: Standard protection clauses that align with data privacy best practices.
        - 🟡 **Medium Risk**: Clauses requiring caution, such as collection of usage data and third-party sharing.
        - 🔴 **High Risk**: Potentially harmful terms like liability waivers, binding arbitration, or permission to sell personal data.
        """
    )
    st.divider()
    st.subheader("🤖 Model Specification")
    st.info(
        "Utilizes a fine-tuned **LegalBERT** transformer optimized with **class-weighted cross-entropy loss** and **hybrid resampling** to accurately classify imbalanced legal datasets."
    )

# Input container
st.subheader("📄 Upload or Paste Terms")
input_text = read_input_text()
text = input_text.text

if text.strip():
    st.info(
        f"**Source:** {input_text.source} | **Extraction:** {input_text.extraction_method} | "
        f"**Characters:** {len(text.strip())}"
    )
if input_text.warning:
    st.warning(input_text.warning)

classify = st.button("Analyze Document", type="primary", use_container_width=True)

# Initialize Session State
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = None

if classify:
    clauses = split_into_clauses(text)
    if not clauses:
        st.warning("Add a longer document or clause to analyze.")
        st.stop()

    with st.spinner("Analyzing legal clauses with LegalBERT..."):
        classifier = get_classifier()
        results = classifier.predict(clauses)
        for line_no, item in enumerate(results, start=1):
            item["line_no"] = line_no
        summary = summarize_results(results)

        st.session_state.analysis_results = results
        st.session_state.analysis_summary = summary

# Render Results
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    summary = st.session_state.analysis_summary

    low_count = sum(item["label"] == 0 for item in results)
    medium_count = sum(item["label"] == 1 for item in results)
    high_count = sum(item["label"] == 2 for item in results)

    # Score Meter Column layout
    col_meter, col_chart = st.columns([1, 1])

    with col_meter:
        score = summary["score"]
        if score < 30:
            meter_color = "#10b981"
            status_text = "Safe / Low Risk"
        elif score < 60:
            meter_color = "#f59e0b"
            status_text = "Cautionary / Medium Risk"
        else:
            meter_color = "#ef4444"
            status_text = "High Risk / Critical"

        st.markdown(
            f"""
            <div style="background: white; border-radius: 16px; padding: 1.5rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); text-align: center; height: 100%;">
                <span style="font-size: 0.9rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Overall Risk Score</span>
                <div style="font-size: 3.5rem; font-weight: 800; color: {meter_color}; margin: 0.2rem 0;">{score}<span style="font-size: 1.5rem; font-weight: 600; color: #94a3b8;">/100</span></div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {meter_color}; margin-bottom: 1rem;">{status_text}</div>
                <div style="background: #e2e8f0; border-radius: 9999px; height: 12px; width: 100%; position: relative; overflow: hidden; margin-top: 1.2rem;">
                    <div style="background: {meter_color}; width: {score}%; height: 100%; border-radius: 9999px; transition: width 0.5s ease-in-out;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        # Donut Chart for breakdown
        chart_data = pd.DataFrame(
            {
                "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
                "Count": [low_count, medium_count, high_count],
            }
        )
        chart_data = chart_data[chart_data["Count"] > 0]
        
        if not chart_data.empty:
            donut_chart = (
                alt.Chart(chart_data)
                .mark_arc(innerRadius=50, outerRadius=75)
                .encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(
                        field="Risk Level",
                        type="nominal",
                        scale=alt.Scale(
                            domain=["Low Risk", "Medium Risk", "High Risk"],
                            range=["#10b981", "#f59e0b", "#ef4444"],
                        ),
                        legend=alt.Legend(orient="right", title="Risk Category"),
                    ),
                    tooltip=["Risk Level", "Count"],
                )
                .properties(height=180)
            )
            st.altair_chart(donut_chart, use_container_width=True)

    st.divider()

    # Metrics grid
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card info">
                <div class="metric-label">Decision</div>
                <div class="metric-value" style="color: {meter_color};">{summary["decision"]}</div>
            </div>
            <div class="metric-card info">
                <div class="metric-label">Clauses</div>
                <div class="metric-value">{len(results)}</div>
            </div>
            <div class="metric-card low">
                <div class="metric-label">Low Risk</div>
                <div class="metric-value" style="color: #10b981;">{low_count}</div>
            </div>
            <div class="metric-card medium">
                <div class="metric-label">Medium Risk</div>
                <div class="metric-value" style="color: #f59e0b;">{medium_count}</div>
            </div>
            <div class="metric-card high">
                <div class="metric-label">High Risk</div>
                <div class="metric-value" style="color: #ef4444;">{high_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clause breakdown / search & filter options
    col_filter, col_search = st.columns([1, 1])
    with col_filter:
        filter_label = st.segmented_control(
            "Filter Clauses",
            options=["All", "Low", "Medium", "High"],
            default="All",
        )
    with col_search:
        search_term = st.text_input("🔍 Search Clauses", placeholder="Type keywords to filter...")

    st.subheader("🔍 Clause Details")

    risk_badges = {
        0: {"bg": "#d1fae5", "fg": "#10b981", "icon": "✓", "lbl": "Low Risk"},
        1: {"bg": "#fef3c7", "fg": "#f59e0b", "icon": "⚠", "lbl": "Medium Risk"},
        2: {"bg": "#fee2e2", "fg": "#ef4444", "icon": "✖", "lbl": "High Risk"},
    }

    shown_any = False
    for item in results:
        if filter_label != "All" and item["risk"] != filter_label:
            continue
        if search_term and search_term.lower() not in item["text"].lower():
            continue

        badge = risk_badges[item["label"]]
        color = badge["fg"]
        bg = badge["bg"]
        icon = badge["icon"]
        lbl = badge["lbl"]

        safe_text = escape(item["text"])
        safe_reason = escape(risk_reason(item))
        shown_any = True

        st.markdown(
            f"""
            <div style="border-left: 5px solid {color}; padding: 1rem; margin: 0.6rem 0; background: white; border-radius: 8px; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #1e293b; font-size: 0.95rem;">Clause #{item["line_no"]}</span>
                    <span style="background: {bg}; color: {color}; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; display: inline-flex; align-items: center; gap: 4px;">
                        <span>{icon}</span> <span>{lbl}</span>
                    </span>
                </div>
                <p style="margin: 0; font-size: 0.925rem; line-height: 1.45; color: #334155;">"{safe_text}"</p>
                <div style="margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px dashed #e2e8f0; display: flex; flex-wrap: wrap; gap: 1.25rem; font-size: 0.8rem; color: #64748b;">
                    <div><strong>Reason:</strong> {safe_reason}</div>
                    <div><strong>Confidence:</strong> {item["confidence"]:.1%}</div>
                    <div><strong>Model:</strong> {item["model"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not shown_any:
        st.info("No clauses matched the selected filters.")

    st.divider()

    # Pros and cons
    col_pros, col_cons = st.columns(2)
    with col_pros:
        st.markdown("### 🟢 Positive Aspects (Pros)")
        if summary["pros"]:
            for clause in summary["pros"]:
                st.success(clause)
        else:
            st.info("No clear low-risk clauses found.")

    with col_cons:
        st.markdown("### 🔴 Risks Identified (Cons)")
        if summary["cons"]:
            for clause in summary["cons"]:
                st.error(clause)
        else:
            st.info("No high-risk clauses found.")

    st.divider()

    # Exporter / Downloader
    def generate_report_markdown(results: list[dict], summary: dict) -> str:
        md = f"# AI Terms Checker - Risk Assessment Report\n\n"
        md += f"## 📊 Executive Summary\n"
        md += f"- **Recommendation:** {summary['decision']}\n"
        md += f"- **Overall Risk Score:** {summary['score']}/100\n"
        md += f"- **Total Clauses Analyzed:** {len(results)}\n\n"
        
        md += f"## 🔍 Detailed Clause Risk Breakdown\n"
        for item in results:
            md += f"### Clause #{item['line_no']} - {item['risk']} Risk\n"
            md += f"- **Text:** \"{item['text']}\"\n"
            md += f"- **Reason:** {risk_reason(item)}\n"
            md += f"- **Confidence:** {item['confidence']:.1%}\n"
            md += f"- **Model Used:** {item['model']}\n\n"
        return md

    report_md = generate_report_markdown(results, summary)
    st.download_button(
        label="📥 Download Detailed Markdown Report",
        data=report_md,
        file_name="terms_risk_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
