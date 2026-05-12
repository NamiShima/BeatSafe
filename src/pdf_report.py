# ─────────────────────────────────────────────────────────────────────────────
# UNICODE FONT SETUP — DejaVu supports full Portuguese characters with accents
# Downloads the font automatically if not present in the src/ folder
# ─────────────────────────────────────────────────────────────────────────────
FONT_URL      = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
FONT_BOLD_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
FONT_PATH      = "DejaVuSans.ttf"
FONT_BOLD_PATH = "DejaVuSans-Bold.ttf"

def ensure_fonts():
    """Downloads DejaVu fonts if not already present locally."""
    if not os.path.exists(FONT_PATH):
        print("Baixando fonte Unicode para suporte ao Português...")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    if not os.path.exists(FONT_BOLD_PATH):
        urllib.request.urlretrieve(FONT_BOLD_URL, FONT_BOLD_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE — Consistent with BeatSafe dark medical theme
# ─────────────────────────────────────────────────────────────────────────────
COLOR_RED     = (230, 57, 70)
COLOR_YELLOW  = (255, 193, 7)
COLOR_GREEN   = (40, 167, 69)
COLOR_DARK    = (30, 30, 30)
COLOR_GRAY    = (100, 100, 100)
COLOR_WHITE   = (255, 255, 255)
COLOR_LIGHT   = (245, 245, 245)
COLOR_ORANGE  = (255, 140, 0)

# ─────────────────────────────────────────────────────────────────────────────
# SEX TRANSLATION — Normalizes English input to Portuguese for the report
# ─────────────────────────────────────────────────────────────────────────────
SEX_TRANSLATION = {
    "male":      "Masculino",
    "female":    "Feminino",
    "masculino": "Masculino",
    "feminino":  "Feminino",
    "m":         "Masculino",
    "f":         "Feminino",
}

def translate_sex(sex: str) -> str:
    """Converts sex field to Portuguese regardless of input language."""
    return SEX_TRANSLATION.get(sex.strip().lower(), sex)


# ─────────────────────────────────────────────────────────────────────────────
# FRAMINGHAM SCORE — Estimates 10-year cardiovascular risk
# Simplified version using age, sex and key risk factors
# Reference: Caderno de Atenção Básica nº 14 — Ministério da Saúde
# ─────────────────────────────────────────────────────────────────────────────
def estimate_framingham(age: int, sex: str, history: str, vitals: str) -> dict:
    """
    Estimates 10-year cardiovascular risk using a simplified Framingham model.
    Considers age, sex, hypertension, diabetes and smoking from patient data.

    Args:
        age:     Patient age in years
        sex:     Biological sex (any language)
        history: Medical history string (looks for comorbidities)
        vitals:  Vital signs string (looks for systolic BP)

    Returns:
        dict with 'score', 'risk_label' and 'color' for use in the report
    """

    score = 0
    history_lower = history.lower()
    vitals_lower  = vitals.lower()

    # Age points — risk increases significantly with age
    if age >= 70:
        score += 14
    elif age >= 65:
        score += 12
    elif age >= 60:
        score += 10
    elif age >= 55:
        score += 8
    elif age >= 50:
        score += 6
    elif age >= 45:
        score += 4
    elif age >= 40:
        score += 2

    # Sex — men have higher baseline risk
    sex_normalized = translate_sex(sex)
    if sex_normalized == "Masculino":
        score += 3

    # Hypertension — adds significant risk
    if any(w in history_lower for w in ["hipertens", "pressão alta", "has"]):
        score += 3
    # Also check if systolic BP is elevated in vitals
    if "16" in vitals_lower or "17" in vitals_lower or "18" in vitals_lower:
        score += 2

    # Diabetes — major independent risk factor
    if any(w in history_lower for w in ["diabetes", "dm ", "dm2", "diabético"]):
        score += 3

    # Smoking — active smoker adds more risk than ex-smoker
    if "tabagista" in history_lower and "ex-tabagista" not in history_lower:
        score += 4
    elif "ex-tabagista" in history_lower:
        score += 2

    # Translate score to risk category (based on Framingham point system)
    if score >= 17:
        return {"score": score, "risk_label": "ALTO (>20% em 10 anos)",   "color": COLOR_RED}
    elif score >= 12:
        return {"score": score, "risk_label": "MODERADO (10–20% em 10 anos)", "color": COLOR_ORANGE}
    else:
        return {"score": score, "risk_label": "BAIXO (<10% em 10 anos)",   "color": COLOR_GREEN}


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

        # Timestamp on the right side of the header
        self.set_font("DejaVu", "", 8)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.set_xy(140, 6)
        self.cell(60, 6, timestamp, align="R")

        # Reset text color for body content
        self.set_text_color(*COLOR_DARK)
        self.ln(8)

    def footer(self):
        """Adds disclaimer and page number footer to every page."""

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
    Returns the appropriate color tuple based on risk level in triage output.

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

    # Remove markdown formatting symbols from AI output
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


def add_framingham_block(pdf: FPDF, framingham: dict):
    """
    Adds the Framingham cardiovascular risk block to the report.
    Uses colored label to match the risk category visually.

    Args:
        pdf:         FPDF instance
        framingham:  Dict returned by estimate_framingham()
    """

    # Section header
    pdf.set_fill_color(*COLOR_DARK)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("DejaVu", "B", 9)
    pdf.cell(
        0, 7,
        "  ESCORE DE FRAMINGHAM - Risco Cardiovascular em 10 Anos",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True
    )

    # Colored risk badge
    pdf.set_fill_color(*framingham["color"])
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_left_margin(12)
    pdf.cell(
        0, 8,
        f"  Risco: {framingham['risk_label']}   |   Pontuação: {framingham['score']} pontos",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True
    )

    # Explanatory note
    pdf.set_left_margin(12)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.set_font("DejaVu", "", 8)
    pdf.multi_cell(
        0, 5,
        "Calculado com base em idade, sexo, PA, diabetes e tabagismo (Caderno AB nº14 - MS).\n"
        "Este escore é uma estimativa de rastreio. Confirmação requer avaliação clínica completa."
    )
    pdf.set_left_margin(10)
    pdf.set_text_color(*COLOR_DARK)
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
    agent_name: str = None,
    agent_id: str = None,
    ecg_analysis: str = None,
    final_recommendation: str = None,
    output_path: str = "beatsafe_report.pdf"
) -> str:
    """
    Generates a complete BeatSafe PDF clinical report with full
    Portuguese character support via DejaVu Unicode font.

    Args:
        patient_name:         Patient name (optional, privacy-safe)
        age:                  Patient age in years
        sex:                  Biological sex (any language — auto-translated)
        chief_complaint:      Main reason for the visit
        symptoms:             Associated symptoms
        vitals:               Vital signs string
        history:              Medical history and comorbidities
        medications:          Current medications
        symptom_triage:       Output from Gemma 3 triage function
        agent_name:           Name of the community health agent (optional)
        agent_id:             Registration number of the agent (optional)
        ecg_analysis:         Output from ECG analysis (optional)
        final_recommendation: Combined recommendation (optional)
        output_path:          Where to save the PDF file

    Returns:
        Path to the generated PDF file
    """

    # Ensure fonts are available before generating PDF
    ensure_fonts()

    # Translate sex to Portuguese regardless of input language
    sex_pt = translate_sex(sex)

    # Calculate Framingham cardiovascular risk score
    framingham = estimate_framingham(age, sex, history, vitals)

    # Generate a unique attendance code for traceability
    attendance_code = str(uuid.uuid4()).upper()[:12]

    # Initialize PDF with Unicode font support
    pdf = BeatSafeReport()
    pdf.add_font("DejaVu", "",  FONT_PATH)       # Regular weight
    pdf.add_font("DejaVu", "B", FONT_BOLD_PATH)  # Bold weight
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(10, 10, 10)

    # ── Attendance Code — unique ID for this report ──
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(0, 5, f"Código de Atendimento: BS-{attendance_code}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(2)

    # ── Patient Information Block ──
    pdf.set_font("DejaVu", "B", 11)
    pdf.cell(0, 8, "INFORMAÇÕES DO PACIENTE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Light gray background for patient details rows
    pdf.set_fill_color(*COLOR_LIGHT)
    pdf.set_font("DejaVu", "", 9)
    name_display = patient_name if patient_name else "Não informado"

    pdf.cell(0, 6, f"  Nome: {name_display}    |    Idade: {age} anos    |    Sexo: {sex_pt}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Queixa Principal: {chief_complaint}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Sinais Vitais: {vitals}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Sintomas: {symptoms}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Histórico Médico: {history}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.cell(0, 6, f"  Medicamentos: {medications}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(4)

    # ── Risk Level Indicator — colored banner with icon ──
    risk_color = get_risk_color(symptom_triage)
    pdf.set_fill_color(*risk_color)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("DejaVu", "B", 12)

    # Select risk label and warning icon based on triage output
    risk_upper = symptom_triage.upper()
    if "ALTO RISCO" in risk_upper:
        risk_label = "⚠ ALTO RISCO - ACAO IMEDIATA NECESSARIA"
    elif "MODERADO" in risk_upper:
        risk_label = "⚠ RISCO MODERADO - AVALIACAO MEDICA NECESSARIA"
    else:
        risk_label = "✓ BAIXO RISCO - ACOMPANHAMENTO DE ROTINA"

    pdf.cell(0, 10, f"  {risk_label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(4)

    # ── Framingham Cardiovascular Risk Score Block ──
    add_framingham_block(pdf, framingham)

    # ── Symptom Triage Section — output from Gemma 3 ──
    add_section(pdf, "TRIAGEM POR SINTOMAS - Gemma 3 (Local, Offline)", symptom_triage)

    # ── ECG Analysis Section — only included when an ECG image was provided ──
    if ecg_analysis and ecg_analysis != "Nenhuma imagem de ECG fornecida.":
        add_section(pdf, "ANALISE DO ECG - Gemini Flash (Nuvem)", ecg_analysis)

    # ── Final Combined Recommendation — only shown when different from triage ──
    if final_recommendation and final_recommendation != symptom_triage:
        add_section(pdf, "RECOMENDACAO FINAL COMBINADA", final_recommendation)

    # ── Health Agent Signature Block ──
    pdf.set_fill_color(*COLOR_LIGHT)
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_text_color(*COLOR_DARK)
    pdf.cell(0, 7, "  IDENTIFICACAO DO AGENTE DE SAUDE", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

    pdf.set_font("DejaVu", "", 9)
    agent_display  = agent_name if agent_name else "___________________________________"
    agent_id_display = agent_id if agent_id else "_______________"
    pdf.set_left_margin(12)
    pdf.cell(0, 6, f"Nome: {agent_display}    |    Registro: {agent_id_display}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

    # Signature line
    pdf.ln(6)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(90, 5, "Assinatura do Agente:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(90, 5, f"Data: {datetime.now().strftime('%d/%m/%Y')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
    pdf.ln(8)
    pdf.set_left_margin(12)
    pdf.line(12, pdf.get_y(), 90, pdf.get_y())   # Signature underline
    pdf.set_left_margin(10)
    pdf.ln(6)

    # ── Emergency Reference Block ──
    pdf.set_fill_color(*COLOR_RED)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("DejaVu", "B", 10)
    pdf.cell(
        0, 8,
        "  EM CASO DE EMERGENCIA - LIGUE SAMU 192 IMEDIATAMENTE",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True
    )
    pdf.set_text_color(*COLOR_DARK)
    pdf.ln(2)

    # ── Generation timestamp and attendance code ──
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(
        0, 6,
        f"Relatorio gerado pelo BeatSafe em {datetime.now().strftime('%d/%m/%Y as %H:%M:%S')}  |  Codigo: BS-{attendance_code}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    # Save PDF to the specified path
    pdf.output(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK — Generates a sample report when run directly: python report_pdf.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("BeatSafe - Gerador de Relatorio PDF")
    print("=" * 45)

    sample_triage = """ALTO RISCO

AVALIACAO:
O paciente apresenta quadro de dor toracica tipica, com caracteristicas sugestivas de sindrome coronariana aguda. A irradiacao para o braco esquerdo, a falta de ar, a sudorese fria e a nausea reforcam essa suspeita.

SINAIS DE ALERTA IDENTIFICADOS:
- Dor toracica tipica com irradiacao
- Falta de ar (dispneia)
- Sudorese fria
- Hipertensao grave (165/105 mmHg)

CONDUTA RECOMENDADA:
1. Acionar SAMU 192 imediatamente
2. Manter paciente em repouso semi-sentado
3. Administrar oxigenio se SatO2 < 94%

ENCAMINHAMENTO:
SAMU 192 - URGENCIA MAXIMA"""

    output = generate_pdf(
        patient_name="Joao Silva",
        age=58,
        sex="Male",                          # Will be auto-translated to "Masculino"
        chief_complaint="Dor no peito ha 2 horas, tipo aperto, irradiando para braco esquerdo",
        symptoms="Sudorese fria, falta de ar, nausea",
        vitals="PA 165/105 mmHg, FC 102 bpm, SatO2 91%",
        history="Hipertensao ha 10 anos, Diabetes tipo 2, ex-tabagista",
        medications="Metformina, Losartana",
        symptom_triage=sample_triage,
        agent_name="Maria Souza",            # Community health agent name
        agent_id="ACS-2024-0381",            # Agent registration number
        output_path="test_report.pdf"
    )

    print(f"\nPDF gerado com sucesso: {output}")
    print("Abra test_report.pdf para visualizar.")
