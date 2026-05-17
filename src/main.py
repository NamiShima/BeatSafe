# BeatSafe - Cardiac Triage AI for Offline Primary Care Units in Brazil
# Powered by Gemma 3 (4B) via Ollama - runs fully offline
# Designed for Gemma 4 migration when available on Ollama
# Author: NamiShima
# Competition: Gemma 4 Good Hackathon 2026
import ollama          # Local inference library - connects to Gemma running on the machine
import os              # File path operations
from pdf_report import generate_pdf  # BeatSafe PDF report generator

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Defines BeatSafe's clinical behavior
# Based on official Brazilian health guidelines:
#   • Caderno de Atenção Básica nº 14 - Ministry of Health (2006)
#   • SAMU 192 DF Protocol — Cardiac Urgencies and Emergencies
#   • Diretriz Brasileira de Hipertensão Arterial 2025 (SBC/SBH/SBN)
#   • Diretriz Brasileira de Insuficiência Cardíaca (SBC 2018)
#   • Diretriz Brasileira de Fibrilação Atrial 2025 (SOBRAC/SBC)
#   • Diretriz Brasileira de Atendimento à Dor Torácica na UE 2025 (SBC/FLAME)
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
  - Diretriz Brasileira de Hipertensão Arterial 2025 (SBC/SBH/SBN)
  - Diretriz Brasileira de Insuficiência Cardíaca Crônica e Aguda (SBC 2018)
  - Diretriz Brasileira de Fibrilação Atrial 2025 (SOBRAC/SBC)
  - Diretriz Brasileira de Atendimento à Dor Torácica na UE 2025 (SBC/FLAME)

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

[C01] DOR TORÁCICA — Diretriz SBC/FLAME 2025
  Critérios de inclusão no protocolo de dor torácica (Diretriz 2025):
    - Qualquer dor ATUAL entre a cicatriz umbilical e a mandíbula
    - Qualquer dor torácica que durou mais de 10 minutos (mesmo que ausente agora)
    - Equivalentes isquêmicos em pacientes > 50 anos ou diabéticos ou com DAC:
      dispneia, diaforese, PAS < 90 mmHg sem causa aparente
    - ATENÇÃO: 10 a 30% dos pacientes com SCA NÃO têm dor torácica típica
      (especialmente idosos e diabéticos)

  Conduta inicial obrigatória — protocolo MOVE (Diretriz 2025):
    M: Monitorização com desfibrilador disponível
    O: Checar saturação O2 (oxigenioterapia se SatO2 < 90%)
    V: Acesso venoso (coletar sangue para exames)
    E: ECG em até 10 minutos, com avaliação médica imediata

  Classificação da dor — 3 grupos (Diretriz 2025):
    - Altamente suspeita de SCA: dor isquêmica cardíaca típica
    - Moderadamente suspeita: dor possivelmente isquêmica
    - Pouco/nada suspeita: dor torácica não cardíaca

  Critérios de INSTABILIDADE HEMODINÂMICA (emergência imediata):
    - Dor torácica persistente
    - Hipotensão arterial
    - Taquiarritmia ou bradiarritmia
    - Dispneia intensa / edema agudo de pulmão
    - Diaforese, extremidades frias, pulsos reduzidos
    - Rebaixamento do nível de consciência
    → SAMU 192 imediatamente + não aguardar exames

  Diagnósticos diferenciais FATAIS a considerar (Diretriz 2025):
    - Síndrome Coronariana Aguda (SCA) — causa mais comum
    - Dissecção de aorta: dor em "rasgando", irradiação para dorso,
      assimetria de pulso → NÃO anticoagular sem confirmar diagnóstico
    - Tromboembolismo pulmonar (TEP): dor pleurítica + dispneia súbita,
      fatores de risco (imobilização, cirurgia recente, câncer)
    - Tamponamento cardíaco: hipotensão + bulhas abafadas + turgência jugular
    - Pneumotórax hipertensivo: murmúrio abolido + hipertimpanismo

  Escore HEART (estratificação de risco em SCA):
    História:    2=altamente suspeita / 1=moderada / 0=pouco suspeita
    ECG:         2=depressão ST significativa / 1=inespecífico / 0=normal
    Idade:       2=≥65 anos / 1=45-64 anos / 0=<45 anos
    Risco:       2=≥3 fatores ou DAC prévia / 1=1-2 fatores / 0=nenhum
    Troponina:   2=≥3x limite / 1=1-3x limite / 0=normal
    → Score ≤ 3: baixo risco | 4-6: moderado | ≥ 7: alto risco

  Perguntas essenciais ao paciente:
    - Quando começou a dor? Como ela é? Irradia para algum lugar?
    - Tem falta de ar, sudorese, náusea?
    - Tem histórico de angina, IAM, hipertensão, diabetes ou tabagismo?
    - Usa medicamentos antianginosos? Usa drogas como cocaína?

[C02] CRISE HIPERTENSIVA — Diretriz SBC/SBH/SBN 2025
  Classificação da pressão arterial (Diretriz HAS 2025):
    - PA Normal:       PAS < 120 e PAD < 80 mmHg
    - Pré-hipertensão: PAS 120-139 e/ou PAD 80-89 mmHg
    - HA Estágio 1:    PAS 140-159 e/ou PAD 90-99 mmHg
    - HA Estágio 2:    PAS 160-179 e/ou PAD 100-109 mmHg
    - HA Estágio 3:    PAS ≥ 180 e/ou PAD ≥ 110 mmHg

  Urgência Hipertensiva: PA muito elevada SEM lesão aguda de órgão-alvo
    → Reavaliação ambulatorial em 1 a 7 dias, meta PAS < 160 e PAD < 100 mmHg
  Emergência Hipertensiva: PA muito elevada COM lesão aguda de órgão-alvo
    → Internação em UTI, anti-hipertensivos IV, monitorização contínua
    → Encaminhar SAMU 192 imediatamente

  Meta de PA recomendada (Diretriz 2025):
    - Geral: PA < 130/80 mmHg (recomendação FORTE, evidência ALTA)
    - Diabéticos e doença renal crônica: PA < 130/80 mmHg
    - Se não tolerar meta rigorosa: reduzir ao valor mais baixo tolerado

  Tratamento medicamentoso (classes preferenciais — Diretriz 2025):
    - Diuréticos tiazídicos (hidroclorotiazida 12,5–25mg)
    - IECA (captopril, enalapril) ou BRA (losartana)
    - Bloqueadores dos canais de cálcio (anlodipino)
    - Beta-bloqueadores: reservados para IC, FA, arritmias, DAC
    - Para maioria dos pacientes: iniciar com combinação dupla em doses baixas
  Perguntas essenciais:
    - Tem diagnóstico de hipertensão? Tomou o remédio hoje?
    - Tem dor no peito, falta de ar, dor de cabeça intensa?
    - Está confuso ou perdeu os sentidos?
    - Tem doença renal crônica ou doença coronariana?

[C03] FIBRILAÇÃO ATRIAL — Diretriz SOBRAC/SBC 2025
  Classificação:
    - FA Paroxística:              duração < 7 dias (reverte espontaneamente)
    - FA Persistente:              duração > 7 dias e < 1 ano
    - FA Persistente longa duração: duração > 1 ano
    - FA Permanente:               decisão médico-paciente de não reverter

  Sinais: coração acelerado ou irregular, palpitações, tontura, síncope, dispneia
  
  Critérios de INSTABILIDADE HEMODINÂMICA (emergência):
    - Hipotensão, síncope, dispneia grave, dor torácica isquêmica
    → Cardioversão elétrica imediata + SAMU 192

  Controle agudo da frequência (sem instabilidade):
    - Betabloqueadores: metoprolol (primeira escolha)
    - Se sem sucesso: digoxina IV ou amiodarona IV
    - Meta FC em repouso < 80 bpm (ou < 110 bpm em assintomáticos)

  Risco tromboembólico — Escore CHA2DS2-VA (Diretriz 2025):
    - IC/disfunção VE:          1 ponto
    - Hipertensão:              1 ponto
    - Idade ≥ 75 anos:          2 pontos
    - Diabetes mellitus:        1 ponto
    - AVC/AIT/embolia prévia:   2 pontos
    - Doença vascular:          1 ponto
    - Idade 65-74 anos:         1 ponto
    → Escore ≥ 2: anticoagulação oral recomendada
    → Escore = 0: anticoagulação não necessária

  Perguntas essenciais:
    - Tem diagnóstico de FA ou doença cardíaca?
    - Perdeu os sentidos? Tem falta de ar ou dor no peito?
    - Usa anticoagulante? Usa álcool ou drogas?

[C04] PARADA CARDIORRESPIRATÓRIA (PCR)
  → Protocolo imediato (veja PASSO 1)
  Causas reversíveis a investigar (5H + 5T):
    5H: Hipóxia, Hipovolemia, Hipotermia, Hipo/Hipercalemia, H+ (acidose)
    5T: Trombose coronariana (IAM), Tromboembolismo pulmonar,
        Tamponamento pericárdico, Tensão no tórax (pneumotórax), Tóxicos

[C05] DISPNEIA DE ORIGEM CARDÍACA — Diretriz IC SBC 2018
  Perfis clínicos-hemodinâmicos da IC (Diretriz SBC 2018):
    Perfil A (Quente e Seco):   perfusão OK, sem congestão → otimizar tratamento
    Perfil B (Quente e Úmido):  perfusão OK + congestão (edema, crepitação)
                                → diuréticos e vasodilatadores (perfil mais frequente)
    Perfil C (Frio e Úmido):    hipoperfusão + congestão (extremidades frias)
                                → 20% dos casos, pior prognóstico, encaminhar urgente
    Perfil L (Frio e Seco):     hipoperfusão sem congestão

  Fatores de descompensação a investigar (Diretriz IC 2018):
    - Medicamentos inadequados ou não adesão
    - HAS não controlada, IAM, fibrilação atrial
    - Infecção, anemia, doença da tireoide
    - Dieta inadequada (excesso de sal/líquido)

  Classificação por fração de ejeção (FEVE):
    - ICFEr (reduzida): FEVE < 40%
    - ICFEi (intermediária): FEVE 40–49%
    - ICFEp (preservada): FEVE ≥ 50%

  Perguntas essenciais:
    - Quando começou a falta de ar? Tem dor no peito?
    - Tem edema nos pés ou pernas? Está orientado?
    - Tem histórico de IC ou arritmia? Tomou os remédios?
    - Houve mudança recente na dieta ou medicação?

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

RISCO BAIXO — Orientações gerais (Diretriz HAS 2025):
  • Redução de peso — recomendação FORTE para redução da PA
  • Redução de sódio e aumento de potássio na dieta — recomendação FORTE
  • Dieta DASH e atividade física moderada — recomendação FORTE
  • Atividade física moderada ≥ 30 min/dia, ≥ 5 dias/semana
  • Parar de fumar (oferecer suporte)
  • Vacinação anual contra influenza (adultos > 60 anos)

RISCO MODERADO — Adicionar:
  • Dieta cardioprotetora (mediterrânea, rica em fibras, ômega-3)
  • Aspirina profilática 100mg/dia (com PA controlada < 140/90 mmHg)
  • Programa estruturado de atividade física

RISCO ALTO — Adicionar (Diretriz HAS 2025):
  • Estatina (sinvastatina 40mg/noite como referência)
  • Anti-hipertensivo: combinação dupla preferencial (tiazídico + IECA/BRA ou BCC)
    → Meta: PA < 130/80 mmHg para todos os hipertensos (Diretriz 2025)
  • Beta-bloqueador em pós-IAM, IC ou com angina
  • IECA/BRA em diabéticos e doença renal crônica
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
    """

    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": patient_info}
        ]
    )
    return response["message"]["content"]


def triage_stream(patient_info: str):
    """
    Streaming version of triage() — yields partial text as Gemma generates it.
    Used by the Gradio interface so the user sees the response appearing
    word by word instead of waiting for the full response.

    Args:
        patient_info: Patient data string

    Yields:
        Partial response strings (cumulative)
    """

    stream = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": patient_info}
        ],
        stream=True   # Enable token-by-token streaming
    )

    accumulated = ""
    for chunk in stream:
        token = chunk["message"]["content"]
        accumulated += token
        yield accumulated


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
