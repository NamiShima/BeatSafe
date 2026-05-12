from fpdf import FPDF          # PDF generation library
from datetime import datetime  # Timestamp for the report
import os                      # File path operations

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE — Consistent with BeatSafe dark medical theme
# Used for section headers and risk level indicators
# ─────────────────────────────────────────────────────────────────────────────
COLOR_RED    = (230, 57, 70)    # High risk — urgent
COLOR_YELLOW = (255, 193, 7)    # Moderate risk — attention
COLOR_GREEN  = (40, 167, 69)    # Low risk — routine
COLOR_DARK   = (30, 30, 30)     # Section headers
COLOR_GRAY   = (100, 100, 100)  # Subtle text
COLOR_WHITE  = (255, 255, 255)  # Text on dark backgrounds


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

        # BeatSafe title in white
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*COLOR_WHITE)
        self.set_xy(10, 4)
        self.cell(0, 10, "BEATSAFE — Cardiac Triage Report", ln=False)

        # Timestamp on the right
        self.set_font("Helvetica", "", 8)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.set_xy(140, 6)
        self.cell(60, 6, timestamp, align="R")

        # Reset text color for body
        self.set_text_color(*COLOR_DARK)
        self.ln(16)

    def footer(self):
        """Adds disclaimer footer to every page."""

        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*COLOR_GRAY)
        self.cell(
            0, 10,
            "BeatSafe supports health workers — it does not replace medical evaluation. "
            "When in doubt, always refer the patient for in-person assessment. | SAMU 192",
            align="C"
        )
        # Page number
        self.set_xy(170, -15)
        self.cell(0, 10, f"Page {self.page_no()}", align="R")


def get_risk_color(risk_text: str) -> tuple:
    """
    Returns the appropriate color tuple based on the risk level
    found in the triage output text.

    Args:
        risk_text: The full triage output text containing risk level

    Returns:
        RGB color tuple matching the risk level
    """

    risk_upper = risk_text.upper()

    # Check for risk keywords in the triage output
    if "ALTO RISCO" in risk_upper or "HIGH RISK" in risk_upper:
        return COLOR_RED
    elif "MODERADO" in risk_upper or "MODERATE" in risk_upper:
        return COLOR_YELLOW
    else:
        return COLOR_GREEN


def add_section(pdf: FPDF, title: str, content: str):
    """
    Adds a formatted section to the PDF with a dark header bar
    and clean body text.

    Args:
        pdf:     The FPDF instance to write to
        title:   Section header text (e.g. "SYMPTOM TRIAGE")
        content: Body text for the section
    """

    # Section header — dark background
    pdf.set_fill_color(*COLOR_DARK)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 7, f"  {title}", ln=True, fill=True)

    # Section body — clean text
    pdf.set_text_color(*COLOR_DARK)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_left_margin(12)

    # Clean up markdown formatting from AI output
    clean_content = (
        content
        .replace("**", "")   # Remove bold markdown
        .replace("##", "")   # Remove heading markdown
        .replace("* ", "• ") # Convert markdown bullets to unicode
    )

    # Write content as multi-line text
    pdf.multi_cell(0, 5, clean_content)
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

    The report includes:
        - Patient identification and vital signs
        - Risk level indicator (color coded)
        - Symptom triage output from Gemma 3
        - ECG analysis from Gemini (if available)
        - Final combined recommendation (if available)
        - Disclaimer and SAMU 192 reference

    Args:
        patient_name:        Patient name (optional)
        age:                 Patient age in years
        sex:                 Biological sex
        chief_complaint:     Main reason for the visit
        symptoms:            Associated symptoms
        vitals:              Vital signs string
        history:             Medical history and comorbidities
        medications:         Current medications
        symptom_triage:      Output from Gemma 3 triage function
        ecg_analysis:        Output from ECG analysis (optional)
        final_recommendation: Combined recommendation (optional)
        output_path:         Where to save the PDF file

    Returns:
        Path to the generated PDF file
    """

    # Initialize the PDF with custom header/footer
    pdf = BeatSafeReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(10, 10, 10)

    # ── Patient Information Block ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_DARK)
    pdf.cell(0, 8, "PATIENT INFORMATION", ln=True)

    # Patient details in a clean grid
    pdf.set_font("Helvetica", "", 9)
    name_display = patient_name if patient_name else "Not provided"

    pdf.set_fill_color(245, 245, 245)  # Light gray background for info block
    pdf.cell(0, 6, f"  Name: {name_display}    |    Age: {age} years    |    Sex: {sex}", ln=True, fill=True)
    pdf.cell(0, 6, f"  Chief Complaint: {chief_complaint}", ln=True, fill=True)
    pdf.cell(0, 6, f"  Vital Signs: {vitals}", ln=True, fill=True)
    pdf.cell(0, 6, f"  Associated Symptoms: {symptoms}", ln=True, fill=True)
    pdf.cell(0, 6, f"  Medical History: {history}", ln=True, fill=True)
    pdf.cell(0, 6, f"  Current Medications: {medications}", ln=True, fill=True)
    pdf.ln(4)

    # ── Risk Level Indicator ──
    risk_color = get_risk_color(symptom_triage)
    pdf.set_fill_color(*risk_color)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 12)

    # Determine risk label from triage output
    risk_upper = symptom_triage.upper()
    if "ALTO RISCO" in risk_upper:
        risk_label = "🔴  HIGH RISK — IMMEDIATE ACTION REQUIRED"
    elif "MODERADO" in risk_upper:
        risk_label = "🟡  MODERATE RISK — MEDICAL EVALUATION NEEDED"
    else:
        risk_label = "🟢  LOW RISK — ROUTINE FOLLOW-UP"

    pdf.cell(0, 10, f"  {risk_label}", ln=True, fill=True)
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(4)

    # ── Symptom Triage Section — Gemma 3 local ──
    add_section(pdf, "SYMPTOM TRIAGE — Gemma 3 (Local, Offline)", symptom_triage)

    # ── ECG Analysis Section — only if provided ──
    if ecg_analysis and ecg_analysis != "Nenhuma imagem de ECG fornecida.":
        add_section(pdf, "ECG ANALYSIS — Gemini 2.5 Flash (Cloud)", ecg_analysis)

    # ── Final Combined Recommendation — only if provided ──
    if final_recommendation and final_recommendation != symptom_triage:
        add_section(pdf, "FINAL COMBINED RECOMMENDATION", final_recommendation)

    # ── Emergency Reference Block ──
    pdf.set_fill_color(*COLOR_RED)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "  IN CASE OF EMERGENCY — CALL SAMU 192 IMMEDIATELY", ln=True, fill=True)
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(2)

    # ── Generation timestamp ──
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(
        0, 6,
        f"Report generated by BeatSafe on {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')}",
        ln=True
    )

    # Save the PDF to the specified output path
    pdf.output(output_path)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK — Generates a sample report when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("📋 BeatSafe — PDF Report Generator")
    print("=" * 45)

    # Sample triage output for testing
    sample_triage = """🔴 ALTO RISCO

📋 AVALIAÇÃO:
O paciente apresenta quadro de dor torácica típica, com características sugestivas de síndrome coronariana aguda. A irradiação para o braço esquerdo, a falta de ar, a sudorese fria e a náusea reforçam essa suspeita.

⚠️ SINAIS DE ALERTA IDENTIFICADOS:
• Dor torácica típica com irradiação
• Falta de ar
• Sudorese fria
• Hipertensão grave (165/105 mmHg)

✅ CONDUTA RECOMENDADA:
1. Acionar SAMU 192 imediatamente
2. Manter paciente em repouso semi-sentado
3. Administrar oxigênio se SatO2 < 94%

🏥 ENCAMINHAMENTO:
SAMU 192 — URGÊNCIA MÁXIMA"""

    # Generate test PDF
    output = generate_pdf(
        patient_name="João Silva",
        age=58,
        sex="Male",
        chief_complaint="Chest pain for 2 hours, pressure-like, radiating to left arm",
        symptoms="Cold sweating, shortness of breath, nausea",
        vitals="BP 165/105 mmHg, HR 102 bpm, SpO2 91%",
        history="Hypertension 10 years, Type 2 Diabetes, former smoker",
        medications="Metformin, Losartan",
        symptom_triage=sample_triage,
        output_path="test_report.pdf"
    )

    print(f"\n✅ PDF generated successfully: {output}")
    print("Open test_report.pdf to preview the report.")
