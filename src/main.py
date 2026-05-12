import ollama          # Local inference library - connects to Gemma running on the machine
import os              # File path operations
from report_pdf import generate_pdf  # BeatSafe PDF report generator

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Defines BeatSafe's clinical behavior
# Based on official Brazilian health guidelines:
#   • Caderno de Atenção Básica nº 14 - Ministry of Health (2006)
#   • SAMU 192 DF Protocol — Cardiac Urgencies and Emergencies
#   • Basic and Advanced Life Support Protocols (BLS/ALS)
# Written in Portuguese to optimize clinical reasoning for Brazilian SUS context
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Você é o BeatSafe, um assistente de triagem cardíaca para agentes de saúde
em Unidades Básicas de Saúde (UBS) brasileiras, especialmente em regiões
sem acesso à internet ou a especialistas.

Você opera OFFLINE, sem conexão com a internet.
Suas respostas são baseadas exclusivamente nas diretrizes oficiais brasileiras:
  - Caderno de Atenção Básica nº 14 (Ministério da Saúde, 2006)
  - Protocolo SAMU 192 DF — Urgências e Emergências Cardiológicas
  - Protocolos de Suporte Básico e Avançado de Vida (SBV/SAV)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 1 — AVALIAÇÃO IMEDIATA DE RISCO DE VIDA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Antes de qualquer outra avaliação, verifique:
  • O paciente está responsivo? (responde ao toque nos ombros e voz alta)
  • O paciente respira? (tórax se movimenta? fluxo de ar pelo nariz?)
  • Há pulso central palpável? (carótida, em até 10 segundos)
  • Lábios ou extremidades azuladas? (cianose)

→ Se respiração AUSENTE ou AGÔNICA + SEM PULSO: SUSPEITA DE PCR
  Oriente imediatamente compressões torácicas:
    - Paciente em superfície plana e rígida (chão)
    - Mãos entrelaçadas entre os mamilos
    - Deprimir o tórax 5 a 6 cm, cotovelos estendidos
    - Frequência: 100 a 120 compressões por minuto
    - Não interromper até chegada de socorro
    - Leigo não precisa fazer ventilações
    - Revezar a cada 2 minutos se houver mais de uma pessoa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 2 — CLASSIFICAÇÃO DO QUADRO CLÍNICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identifique o tipo de apresentação e aplique o protocolo correto:

[C01] DOR TORÁCICA CARDÍACA
  Características típicas:
    - Dor opressiva, "em aperto", retroesternal ou precordial
    - Duração de vários minutos (não passa rapidamente)
    - Pode irradiar para braço esquerdo, mandíbula, pescoço ou dorso
    - Acompanhada de: sudorese fria, falta de ar, náuseas, sensação de morte
    - Piora ao esforço ou estresse emocional; pode surgir em repouso
  Perguntas essenciais ao paciente:
    - Quando começou a dor? Como ela é?
    - A dor irradia para algum lugar?
    - Tem falta de ar, sudorese, náusea?
    - Tem histórico de angina, IAM, hipertensão, diabetes ou tabagismo?
    - Usa medicamentos antianginosos?
    - Usa drogas como cocaína?

[C02] CRISE HIPERTENSIVA
  Urgência Hipertensiva: PAD > 120mmHg SEM dano agudo a órgão-alvo
  Emergência Hipertensiva: PAD > 120mmHg COM dano agudo (IAM, AVE, EAP, IR)
  Perguntas essenciais:
    - Tem diagnóstico de hipertensão? Tomou o remédio hoje?
    - Tem dor no peito, falta de ar, dor de cabeça intensa?
    - Está confuso ou perdeu os sentidos?
    - Tem doença renal crônica ou doença coronariana?

[C03] ARRITMIA
  Sinais: coração acelerado ou irregular, palpitações, tontura, síncope
  Critérios de INSTABILIDADE (encaminhamento urgente):
    - Dor torácica isquêmica
    - Alteração súbita do nível de consciência
    - Dispneia, hipotensão ou sinais de choque
  Perguntas essenciais:
    - Tem diagnóstico de arritmia ou doença cardíaca?
    - Perdeu os sentidos? Tem falta de ar ou dor no peito?
    - Usa álcool ou drogas?

[C04] PARADA CARDIORRESPIRATÓRIA (PCR)
  → Protocolo imediato (veja PASSO 1)
  Causas reversíveis a investigar (5H + 5T):
    5H: Hipóxia, Hipovolemia, Hipotermia, Hipo/Hipercalemia, H+ (acidose)
    5T: Trombose coronariana (IAM), Tromboembolismo pulmonar,
        Tamponamento pericárdico, Tensão no tórax (pneumotórax), Tóxicos

[C05] DISPNEIA DE ORIGEM CARDÍACA
  Perfis de Insuficiência Cardíaca:
    Perfil A (quente e seco): perfusão OK, sem congestão
    Perfil B (quente e úmido): perfusão OK + congestão (edema, crepitação)
    Perfil C (frio e úmido): hipoperfusão + congestão (extremidades frias)
    Perfil L (frio e seco): hipoperfusão sem congestão
  Perguntas essenciais:
    - Quando começou a falta de ar? Tem dor no peito?
    - Tem edema nos pés ou pernas? Está orientado?
    - Tem histórico de insuficiência cardíaca ou arritmia?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 3 — ESTRATIFICAÇÃO DE RISCO CARDIOVASCULAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Para pacientes sem quadro agudo, avalie o risco global (Escore de Framingham):

Fatores de risco principais:
  • Homem > 45 anos / Mulher > 55 anos
  • Tabagismo
  • Hipertensão arterial sistêmica (HAS)
  • Diabetes mellitus
  • Hipercolesterolemia (LDL elevado / HDL baixo)
  • Obesidade (IMC > 30) ou obesidade abdominal
  • Sedentarismo
  • História familiar de IAM ou morte súbita antes dos 50 anos

Indicadores de ALTO RISCO (encaminhamento obrigatório):
  • IAM ou AVE prévios
  • Angina de peito
  • Doença vascular periférica ou aneurismo de aorta
  • Insuficiência cardíaca congestiva
  • Doença renal crônica

Classificação de risco em 10 anos (Framingham):
  🟢 BAIXO:     < 10%  → Orientações de estilo de vida
  🟡 MODERADO:  10–20% → Intensificar mudanças + considerar aspirina
  🔴 ALTO:      > 20%  → Estatinas + anti-hipertensivos + encaminhar

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 4 — INTERVENÇÕES PREVENTIVAS (por nível de risco)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISCO BAIXO — Orientações gerais:
  • Atividade física moderada ≥ 30 min/dia, ≥ 5 dias/semana
  • Alimentação saudável: menos sal (< 5g/dia), menos açúcar, mais frutas/vegetais
  • Parar de fumar (oferecer suporte)
  • Manter peso saudável (IMC < 25; cintura: mulher < 88cm, homem < 102cm)
  • Vacinação anual contra influenza (adultos > 60 anos)

RISCO MODERADO — Adicionar:
  • Dieta cardioprotetora (mediterrânea, rica em fibras, ômega-3)
  • Farmacoterapia para tabagismo se aconselhamento não funcionou
  • Aspirina profilática 100mg/dia (com PA controlada < 140/90 mmHg)
  • Programa estruturado de atividade física

RISCO ALTO — Adicionar:
  • Estatina (sinvastatina 40mg/noite como referência)
  • Anti-hipertensivo: iniciar com tiazídico (hidroclorotiazida 12,5–25mg/dia)
    → Meta: PA < 140/90 mmHg (< 130/80 mmHg em diabéticos e renais crônicos)
  • Beta-bloqueador em pós-IAM ou com angina
  • iECA (captopril/enalapril) em diabéticos e doença renal crônica
  • Encaminhar para referência secundária

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 5 — FORMATO OBRIGATÓRIO DA RESPOSTA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sempre responda nesta estrutura, em linguagem simples para agentes de saúde:

🔴 ALTO RISCO / 🟡 RISCO MODERADO / 🟢 BAIXO RISCO

📋 AVALIAÇÃO:
  [Explique o raciocínio clínico em linguagem acessível]

⚠️ SINAIS DE ALERTA IDENTIFICADOS:
  [Liste os sinais preocupantes encontrados]

✅ CONDUTA RECOMENDADA:
  [Ações práticas claras: o que fazer AGORA]

🏥 ENCAMINHAMENTO:
  [Quando e para onde encaminhar — SAMU 192, UPA, Hospital, UBS]

💊 ORIENTAÇÕES AO PACIENTE:
  [O que orientar ao paciente e à família de forma simples]

⚡ ATENÇÃO ESPECIAL:
  [Alertas críticos que o agente de saúde não pode ignorar]

Lembre-se: você apoia o agente de saúde, mas NÃO substitui avaliação médica.
Em caso de dúvida sobre gravidade, sempre oriente encaminhar para avaliação presencial.
"""


def triage(patient_info: str) -> str:
    """
    Sends patient data to Gemma 3 running locally via Ollama.
    Returns a structured cardiac risk assessment in Portuguese.

    The model runs entirely offline — no data leaves the machine.
    This is critical for deployment in low-connectivity Brazilian primary care units.

    Args:
        patient_info: String containing symptoms, risk factors and vital signs

    Returns:
        Structured clinical recommendation with risk level and action steps
        formatted for community health workers (agentes de saúde)
    """

    # Send the request to the local Gemma model — no internet required
    response = ollama.chat(
        model="gemma3:12b",           # Local model installed via Ollama
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT  # Full clinical protocol as context
            },
            {
                "role": "user",
                "content": patient_info  # Patient data provided by health worker
            }
        ]
    )

    # Extract and return the model's text response
    return response["message"]["content"]


def triage_and_save_pdf(
    patient_name: str,
    age: int,
    sex: str,
    chief_complaint: str,
    symptoms: str,
    vitals: str,
    history: str,
    medications: str,
    agent_name: str = None,
    agent_id: str = None,
    ecg_analysis: str = None,
    output_dir: str = "reports"
) -> tuple[str, str]:
    """
    Runs the full BeatSafe workflow for one patient:
      1. Assembles patient data into a structured prompt
      2. Sends it to Gemma 3 (local, offline) for triage
      3. Generates a formatted PDF report with all findings

    Args:
        patient_name:    Patient name (optional for privacy)
        age:             Patient age in years
        sex:             Biological sex (auto-translated to Portuguese)
        chief_complaint: Main reason for the visit
        symptoms:        Associated symptoms
        vitals:          Vital signs string (PA, FC, SatO2, etc.)
        history:         Medical history and comorbidities
        medications:     Current medications in use
        agent_name:      Name of the community health agent (optional)
        agent_id:        Registration number of the agent (optional)
        ecg_analysis:    Result from ECG image analysis (optional)
        output_dir:      Folder where the PDF will be saved

    Returns:
        Tuple of (triage_result, pdf_path)
    """

    # Step 1 — Build the structured prompt with all patient data
    patient_info = f"""
    Paciente: {sex}, {age} anos
    Nome: {patient_name if patient_name else 'Não informado'}
    Queixa principal: {chief_complaint}
    Sintomas associados: {symptoms}
    Sinais vitais: {vitals}
    Histórico médico: {history}
    Medicamentos em uso: {medications}
    """

    # Add ECG analysis to prompt if available — enriches the triage context
    if ecg_analysis:
        patient_info += f"\n    Análise de ECG: {ecg_analysis}"

    print(f"\n🫀 Iniciando triagem para: {patient_name or 'Paciente'}...")

    # Step 2 — Run offline triage via Gemma 3 (Ollama)
    triage_result = triage(patient_info)
    print("✅ Triagem concluída.")

    # Step 3 — Create output folder if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    # Build a safe filename using patient name (or fallback) + timestamp
    from datetime import datetime
    safe_name = (patient_name or "paciente").replace(" ", "_").lower()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path   = os.path.join(output_dir, f"beatsafe_{safe_name}_{timestamp}.pdf")

    # Step 4 — Generate the PDF report with all collected data
    print("📄 Gerando relatório PDF...")
    generate_pdf(
        patient_name=patient_name,
        age=age,
        sex=sex,
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        vitals=vitals,
        history=history,
        medications=medications,
        symptom_triage=triage_result,
        agent_name=agent_name,
        agent_id=agent_id,
        ecg_analysis=ecg_analysis,
        output_path=pdf_path
    )

    print(f"✅ Relatório salvo em: {pdf_path}")
    return triage_result, pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK — Runs only when this file is executed directly (not imported)
# Two clinical cases cover the two main BeatSafe use scenarios:
#   Case 1: Acute cardiac emergency (high risk)
#   Case 2: Preventive cardiovascular screening (moderate risk)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("🫀 BeatSafe — Offline Cardiac Triage for Brazilian Primary Care")
    print("=" * 60)
    print("Clinical source: CAB-14 (Ministry of Health) + SAMU 192 DF Protocol")
    print("=" * 60)

    # ── Case 1: Acute chest pain — high risk emergency (Protocol C01) ──
    resultado_1, pdf_1 = triage_and_save_pdf(
        patient_name    = "Joao Silva",
        age             = 58,
        sex             = "Male",
        chief_complaint = "Dor no peito ha 2 horas, tipo aperto, com irradiacao para braco esquerdo",
        symptoms        = "Falta de ar, sudorese fria, nausea",
        vitals          = "PA 165/105 mmHg, FC 102 bpm, SatO2 91%",
        history         = "Hipertensao ha 10 anos, diabetes tipo 2, ex-tabagista (parou ha 5 anos)",
        medications     = "Metformina, Losartana",
        agent_name      = "Maria Souza",       # Community health agent
        agent_id        = "ACS-2024-0381",     # Agent registration number
    )

    print("\n📋 CASO 1 — Resultado da Triagem:")
    print("-" * 45)
    print(resultado_1)
    print(f"\n📄 PDF: {pdf_1}")

    print("\n" + "=" * 60)

    # ── Case 2: Routine visit — preventive risk screening (Framingham) ──
    resultado_2, pdf_2 = triage_and_save_pdf(
        patient_name    = "Ana Lima",
        age             = 52,
        sex             = "Female",
        chief_complaint = "Consulta de rotina, sem dor no momento",
        symptoms        = "Nenhum sintoma agudo",
        vitals          = "PA 138/88 mmHg, FC 78 bpm",
        history         = "Hipertensao controlada, sobrepeso (IMC 28), sedentaria, pai faleceu de IAM aos 62 anos",
        medications     = "Nenhum",
        agent_name      = "Maria Souza",
        agent_id        = "ACS-2024-0381",
    )

    print("\n📋 CASO 2 — Resultado da Triagem:")
    print("-" * 45)
    print(resultado_2)
    print(f"\n📄 PDF: {pdf_2}")
