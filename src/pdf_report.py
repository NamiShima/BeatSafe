from fpdf import FPDF          # PDF generation library
from fpdf.enums import XPos, YPos  # Updated position enums for fpdf2 v2.5+
from datetime import datetime  # Timestamp for the report
import os                      # File path operations

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE — Consistent with BeatSafe dark medical theme
# ─────────────────────────────────────────────────────────────────────────────
COLOR_RED    = (230, 57, 70)
COLOR_YELLOW = (255, 193, 7)
COLOR_GREEN  = (40, 167, 69)
COLOR_DARK   = (30, 30, 30)
COLOR_GRAY   = (100, 100, 100)
COLOR_WHITE  = (255, 255, 255)


class BeatSafeReport(FPDF):
    """
    Custom PDF class for BeatSafe clinical reports.
    Extends FPDF with a branded header and footer on every page.
    """

    def header(self):
        """Adds BeatSafe branded header to every page."""

        # Red header bar
        self.set_fill_color(*COLOR_RED)
        self.rect(0, 0, 210, 18, 'F')

        # BeatSafe title in white — using hyphen instead of em dash for latin-1 compatibility
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*COLOR_WHITE)
        self.set_xy(10, 4)
        self.cell(
            0, 10,
            "BEATSAFE - Cardiac Triage Report",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT  # Updated API for fpdf2 v2.5+
        )

        # Timestamp on the right side of header
        self.set_font("Helvetica", "", 8)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.set_xy(140, 6)
        self.cell(60, 6, timestamp, align="R")

        # Reset text color for body content
        self.set_text_color(*COLOR_DARK)
        self.ln(8)

    def footer(self):
        """Adds disclaimer footer to every page."""

        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*COLOR_GRAY)
        self.cell(
            0, 10,
            "BeatSafe supports health workers - it does not replace medical evaluation. "
            "When in doubt, always refer the patient for in-person assessment. | SAMU 192",
            align="C"
        )


def get_risk_color(risk_text: str) -> tuple:
    """
    Returns the appropriate color based on risk level in triage output.

    Args:
        risk_text: Full triage output text

    Returns:
        RGB color tuple matching the risk level
    """

    risk_upper = risk_text.upper()

    if "ALTO RISCO" in risk_upper or "HIGH RISK" in risk_upper:
        return COLOR_RED
    elif "MODERADO" in risk_upper or "MODERATE" in risk_upper:
        return COLOR_YELLOW
    else:
        return COLOR_GREEN


def clean_text(text: str) -> str:
    """
    Cleans AI output text for PDF compatibility.
    Removes markdown formatting and replaces special characters
    that are outside the latin-1 range supported by Helvetica.

    Args:
        text: Raw AI output text with markdown

    Returns:
        Clean text safe for PDF rendering
    """

    return (
        text
        .replace("\u2014", "-")   # Em dash to hyphen
        .replace("\u2013", "-")   # En dash to hyphen
        .replace("\u2019", "'")   # Smart quote to apostrophe
        .replace("\u2018", "'")   # Smart quote to apostrophe
        .replace("\u201c", '"')   # Smart double quote
        .replace("\u201d", '"')   # Smart double quote
        .replace("\u2022", "*")   # Bullet to asterisk
        .replace("**", "")        # Remove bold markdown
        .replace("##", "")        # Remove heading markdown
        .replace("* ", "- ")      # Markdown bullets to dashes
    )


def add_section(pdf: FPDF, title: str, content: str):
    """
    Adds a formatted section with dark header bar and clean body text.

    Args:
        pdf:     FPDF instance
        title:   Section header text
        content: Body text content
    """

    # Dark section header bar
    pdf.set_fill_color(*COLOR_DARK)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 7, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

    # Clean body text
    pdf.set_text_color(*COLOR_DARK)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_left_margin(12)
    pdf.multi_cell(0, 5, clean_text(content))
    pdf.set_left_margin(10)
    pdf.ln(4)


def generate_pdf(
    patient_name: str,
    age: int,
    sex: str,
    chief_complaint: str,
    symptoms: str,
    vitals: str,
    history: str,
    medications: str,
    symptom_triage: str,
    ecg_analysis: str = None,
    final_recommendation: str = None,
    output_path: str = "beatsafe_report.pdf"
) -> str:
    """
    Generates a complete BeatSafe PDF clinical report.

    Args:
        patient_name:         Patient name (optional)
        age:                  Patient age in years
        sex:                  Biological sex
        chief_complaint:      Main reason for the visit
        symptoms:             Associated symptoms
        vitals:               Vital signs string
        history:              Medical history and comorbidities
        medications:          Current medications
        symptom_triage:       Output from Gemma 3 triage function
        ecg_analysis:         Output from ECG analysis (optional)
        final_recommendation: Combined recommendation (optional)
        output_path:          Where to save the PDF file

    Returns:
        Path to the generated PDF file
    """

    # Initialize PDF with custom header/footer
    pdf = BeatSafeReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(10, 10, 10)

    # ── Patient Information Block ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_DARK)
    pdf.cell(0, 8, "PATIENT INFORMATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Light gray background for patient details
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Helvetica", "", 9)
    name_display = patient_name if patient_name else "Not provided"

    pdf.cell(0, 6, f"  Name: {name_display}    |    Age: {age} years    |    Sex: {sex}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Chief Complaint: {clean_text(chief_complaint)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Vital Signs: {clean_text(vitals)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Symptoms: {clean_text(symptoms)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Medical History: {clean_text(history)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Medications: {clean_text(medications)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(4)

    # ── Risk Level Indicator ──
    risk_color = get_risk_color(symptom_triage)
    pdf.set_fill_color(*risk_color)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 12)

    # Determine risk label from triage output
    risk_upper = symptom_triage.upper()
    if "ALTO RISCO" in risk_upper:
        risk_label = "HIGH RISK - IMMEDIATE ACTION REQUIRED"
    elif "MODERADO" in risk_upper:
        risk_label = "MODERATE RISK - MEDICAL EVALUATION NEEDED"
    else:
        risk_label = "LOW RISK - ROUTINE FOLLOW-UP"

    pdf.cell(0, 10, f"  {risk_label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(4)

    # ── Symptom Triage Section ──
    add_section(pdf, "SYMPTOM TRIAGE - Gemma 3 (Local, Offline)", symptom_triage)

    # ── ECG Analysis Section — only if provided ──
    if ecg_analysis and ecg_analysis != "Nenhuma imagem de ECG fornecida.":
        add_section(pdf, "ECG ANALYSIS - Gemini 2.5 Flash (Cloud)", ecg_analysis)

    # ── Final Combined Recommendation — only if different from symptom triage ──
    if final_recommendation and final_recommendation != symptom_triage:
        add_section(pdf, "FINAL COMBINED RECOMMENDATION", final_recommendation)

    # ── Emergency Reference Block ──
    pdf.set_fill_color(*COLOR_RED)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "  IN CASE OF EMERGENCY - CALL SAMU 192 IMMEDIATELY", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(2)

    # ── Generation timestamp ──
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(
        0, 6,
        f"Report generated by BeatSafe on {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    # Save PDF to output path
    pdf.output(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK — Generates a sample report when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("BeatSafe - PDF Report Generator")
    print("=" * 45)

    # Sample triage output for testing
    sample_triage = """ALTO RISCO

AVALIACAO:
O paciente apresenta quadro de dor toracica tipica, com caracteristicas sugestivas de sindrome coronariana aguda.

SINAIS DE ALERTA IDENTIFICADOS:
- Dor toracica tipica com irradiacao
- Falta de ar
- Sudorese fria
- Hipertensao grave (165/105 mmHg)

CONDUTA RECOMENDADA:
1. Acionar SAMU 192 imediatamente
2. Manter paciente em repouso semi-sentado
3. Administrar oxigenio se SatO2 menor que 94%

ENCAMINHAMENTO:
SAMU 192 - URGENCIA MAXIMA"""

    # Generate test report
    output = generate_pdf(
        patient_name="Joao Silva",
        age=58,
        sex="Male",
        chief_complaint="Chest pain for 2 hours, radiating to left arm",
        symptoms="Cold sweating, shortness of breath, nausea",
        vitals="BP 165/105 mmHg, HR 102 bpm, SpO2 91%",
        history="Hypertension 10 years, Type 2 Diabetes, former smoker",
        medications="Metformin, Losartan",
        symptom_triage=sample_triage,
        output_path="test_report.pdf"
    )

    print(f"\nPDF generated successfully: {output}")
    print("Open test_report.pdf to preview the report.")
