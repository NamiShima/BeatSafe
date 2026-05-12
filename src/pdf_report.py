from fpdf import FPDF                  # PDF generation library
from fpdf.enums import XPos, YPos      # Updated position enums for fpdf2 v2.5+
from datetime import datetime          # Timestamp for the report
import os                              # File path operations
import urllib.request                  # Download Unicode font if not present

# ─────────────────────────────────────────────────────────────────────────────
# UNICODE FONT SETUP — DejaVu supports full Portuguese characters with accents
# Downloads the font automatically if not present in the src/ folder
# ─────────────────────────────────────────────────────────────────────────────
FONT_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
FONT_BOLD_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
FONT_PATH = "DejaVuSans.ttf"
FONT_BOLD_PATH = "DejaVuSans-Bold.ttf"

def ensure_fonts():
    """Downloads DejaVu fonts if not already present locally."""
    if not os.path.exists(FONT_PATH):
        print("Downloading Unicode font for Portuguese support...")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    if not os.path.exists(FONT_BOLD_PATH):
        urllib.request.urlretrieve(FONT_BOLD_URL, FONT_BOLD_PATH)

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
    Uses DejaVu Unicode font for full Portuguese character support.
    """

    def header(self):
        """Adds BeatSafe branded header to every page."""

        # Red header bar
        self.set_fill_color(*COLOR_RED)
        self.rect(0, 0, 210, 18, 'F')

        # BeatSafe title in white using Unicode font
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(*COLOR_WHITE)
        self.set_xy(10, 4)
        self.cell(
            0, 10,
            "BEATSAFE - Relatório de Triagem Cardíaca",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )

        # Timestamp on the right
        self.set_font("DejaVu", "", 8)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.set_xy(140, 6)
        self.cell(60, 6, timestamp, align="R")

        # Reset text color for body
        self.set_text_color(*COLOR_DARK)
        self.ln(8)

    def footer(self):
        """Adds disclaimer footer to every page."""

        self.set_y(-15)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(*COLOR_GRAY)
        self.cell(
            0, 10,
            "BeatSafe apoia profissionais de saúde - não substitui avaliação médica. "
            "Em caso de dúvida, encaminhe o paciente para avaliação presencial. | SAMU 192",
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


def add_section(pdf: FPDF, title: str, content: str):
    """
    Adds a formatted section with dark header bar and clean body text.
    Uses DejaVu font for full Portuguese accent support.

    Args:
        pdf:     FPDF instance
        title:   Section header text
        content: Body text content
    """

    # Dark section header bar
    pdf.set_fill_color(*COLOR_DARK)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("DejaVu", "B", 9)
    pdf.cell(0, 7, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

    # Clean body text with full Portuguese support
    pdf.set_text_color(*COLOR_DARK)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_left_margin(12)

    # Remove markdown formatting from AI output
    clean = (
        content
        .replace("**", "")
        .replace("##", "")
        .replace("* ", "- ")
        .replace("•", "-")
    )

    pdf.multi_cell(0, 5, clean)
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
    Generates a complete BeatSafe PDF clinical report with full
    Portuguese character support via DejaVu Unicode font.

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

    # Ensure fonts are available before generating PDF
    ensure_fonts()

    # Initialize PDF with Unicode font support
    pdf = BeatSafeReport()
    pdf.add_font("DejaVu", "", FONT_PATH)        # Regular weight
    pdf.add_font("DejaVu", "B", FONT_BOLD_PATH)  # Bold weight
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(10, 10, 10)

    # ── Patient Information Block ──
    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 11)
    pdf.set_text_color(*COLOR_DARK)
    pdf.cell(0, 8, "INFORMAÇÕES DO PACIENTE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Light gray background for patient details
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("DejaVu", "", 9)
    name_display = patient_name if patient_name else "Não informado"

    pdf.cell(0, 6, f"  Nome: {name_display}    |    Idade: {age} anos    |    Sexo: {sex}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Queixa Principal: {chief_complaint}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Sinais Vitais: {vitals}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Sintomas: {symptoms}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Histórico Médico: {history}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Medicamentos: {medications}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(4)

    # ── Risk Level Indicator ──
    risk_color = get_risk_color(symptom_triage)
    pdf.set_fill_color(*risk_color)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("DejaVu", "B", 12)

    # Determine risk label from triage output
    risk_upper = symptom_triage.upper()
    if "ALTO RISCO" in risk_upper:
        risk_label = "ALTO RISCO - AÇÃO IMEDIATA NECESSÁRIA"
    elif "MODERADO" in risk_upper:
        risk_label = "RISCO MODERADO - AVALIAÇÃO MÉDICA NECESSÁRIA"
    else:
        risk_label = "BAIXO RISCO - ACOMPANHAMENTO DE ROTINA"

    pdf.cell(0, 10, f"  {risk_label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(4)

    # ── Symptom Triage Section ──
    add_section(pdf, "TRIAGEM POR SINTOMAS - Gemma 3 (Local, Offline)", symptom_triage)

    # ── ECG Analysis Section — only if provided ──
    if ecg_analysis and ecg_analysis != "Nenhuma imagem de ECG fornecida.":
        add_section(pdf, "ANÁLISE DO ECG - Gemini 2.5 Flash (Nuvem)", ecg_analysis)

    # ── Final Combined Recommendation — only if different ──
    if final_recommendation and final_recommendation != symptom_triage:
        add_section(pdf, "RECOMENDAÇÃO FINAL COMBINADA", final_recommendation)

    # ── Emergency Reference Block ──
    pdf.set_fill_color(*COLOR_RED)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("DejaVu", "B", 10)
    pdf.cell(0, 8, "  EM CASO DE EMERGÊNCIA - LIGUE SAMU 192 IMEDIATAMENTE", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(2)

    # ── Generation timestamp ──
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(
        0, 6,
        f"Relatório gerado pelo BeatSafe em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    # Save PDF
    pdf.output(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK — Generates a sample report when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("BeatSafe - Gerador de Relatório PDF")
    print("=" * 45)

    sample_triage = """🔴 ALTO RISCO

📋 AVALIAÇÃO:
O paciente apresenta quadro de dor torácica típica, com características sugestivas de síndrome coronariana aguda. A irradiação para o braço esquerdo, a falta de ar, a sudorese fria e a náusea reforçam essa suspeita.

⚠️ SINAIS DE ALERTA IDENTIFICADOS:
• Dor torácica típica com irradiação
• Falta de ar (dispneia)
• Sudorese fria
• Hipertensão grave (165/105 mmHg)

✅ CONDUTA RECOMENDADA:
1. Acionar SAMU 192 imediatamente
2. Manter paciente em repouso semi-sentado
3. Administrar oxigênio se SatO2 < 94%

🏥 ENCAMINHAMENTO:
SAMU 192 — URGÊNCIA MÁXIMA"""

    output = generate_pdf(
        patient_name="João Silva",
        age=58,
        sex="Male",
        chief_complaint="Dor no peito há 2 horas, tipo aperto, irradiando para braço esquerdo",
        symptoms="Sudorese fria, falta de ar, náusea",
        vitals="PA 165/105 mmHg, FC 102 bpm, SatO2 91%",
        history="Hipertensão há 10 anos, Diabetes tipo 2, ex-tabagista",
        medications="Metformina, Losartana",
        symptom_triage=sample_triage,
        output_path="test_report.pdf"
    )

    print(f"\n✅ PDF gerado com sucesso: {output}")
    print("Abra test_report.pdf para visualizar.")
