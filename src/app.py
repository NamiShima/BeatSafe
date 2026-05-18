# BeatSafe - Cardiac Triage AI for Offline Primary Care Units in Brazil
# Powered by Gemma 4 (E4B) via Ollama - runs fully offline
# Author: NamiShima
# Competition: Gemma 4 Good Hackathon 2026
import gradio as gr              # Web interface library
import os                        # File path operations
import tempfile                  # Temporary file handling for PDF download

# Import core BeatSafe modules
from main import triage, triage_stream
from ecg import analyze_ecg, combined_analysis
from pdf_report import generate_pdf
from history import save_triage, get_history, get_stats   # SQLite history module

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark medical theme with red accent
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    box-sizing: border-box;
}

.gradio-container {
    background-color: #0d0d0d !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}

.header-block {
    text-align: center;
    padding: 32px 0 16px 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 24px;
}

.header-block h1 {
    color: #e63946;
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: 2px;
    margin: 0;
}

.header-block p {
    color: #888;
    font-size: 0.85rem;
    margin: 8px 0 0 0;
    letter-spacing: 1px;
}

label, .label-wrap {
    color: #aaa !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}

textarea, input[type="text"], input[type="number"] {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e0e0e0 !important;
    font-size: 0.9rem !important;
}

textarea:focus, input:focus {
    border-color: #e63946 !important;
    outline: none !important;
}

.triage-btn {
    background-color: #e63946 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: background-color 0.2s !important;
}

.triage-btn:hover {
    background-color: #c1121f !important;
}

.output-box textarea {
    background-color: #111 !important;
    border: 1px solid #2a2a2a !important;
    color: #e0e0e0 !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    min-height: 300px !important;
}

.section-title {
    color: #e63946;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 8px;
    margin: 16px 0 12px 0;
}

.footer-note {
    text-align: center;
    color: #555;
    font-size: 0.72rem;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #1a1a1a;
}

.stat-box {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
"""


def run_full_analysis(
    patient_name, age, sex,
    chief_complaint, symptoms, vitals, history, medications,
    ecg_image
):
    """
    Runs the complete BeatSafe pipeline with streaming for the triage step.
    Yields partial results so the user sees text appearing in real time.
    PDF and history are saved after the full triage is complete.
    """

    # Assemble patient info string
    patient_info = f"""
    Paciente: {sex}, {age} anos {"— " + patient_name if patient_name else ""}
    Queixa principal: {chief_complaint}
    Sintomas associados: {symptoms}
    Sinais vitais: {vitals}
    Histórico médico / Comorbidades: {history}
    Medicamentos em uso: {medications}
    """

    # Step 1 — Stream triage response token by token
    symptom_result = ""
    for partial in triage_stream(patient_info):
        symptom_result = partial
        yield partial, "Aguardando triagem...", "", None

    # Step 2 — ECG analysis (only if image provided)
    if ecg_image is not None:
        yield symptom_result, "Analisando ECG...", "", None
        ecg_result   = analyze_ecg(ecg_image)
        final_result = combined_analysis(symptom_result, ecg_result)
    else:
        ecg_result   = "Nenhuma imagem de ECG fornecida."
        final_result = symptom_result

    yield symptom_result, ecg_result, final_result, None

    # Step 3 — Generate PDF
    pdf_path = os.path.join(tempfile.gettempdir(), "beatsafe_report.pdf")
    generate_pdf(
        patient_name=patient_name,
        age=int(age),
        sex=sex,
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        vitals=vitals,
        history=history,
        medications=medications,
        symptom_triage=symptom_result,
        ecg_analysis=ecg_result if ecg_image else None,
        final_recommendation=final_result if ecg_image else None,
        output_path=pdf_path
    )

    # Step 4 — Save to history
    save_triage(
        patient_name=patient_name,
        age=int(age),
        sex=sex,
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        vitals=vitals,
        history=history,
        medications=medications,
        triage_result=symptom_result,
        ecg_analysis=ecg_result if ecg_image else None,
        pdf_path=pdf_path
    )

    yield symptom_result, ecg_result, final_result, pdf_path


def load_history(risk_filter: str) -> list:
    """Fetches triage history from SQLite filtered by risk level."""
    return get_history(risk_filter=risk_filter)


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT — Gemma 4 offline answering clinical questions from health agents
# Focused on SUS protocols, CAB-14, SAMU 192 and primary care procedures
# ─────────────────────────────────────────────────────────────────────────────
CHATBOT_SYSTEM_PROMPT = """
Você é o BeatSafe Assistente, um apoio clínico offline para agentes comunitários
de saúde e profissionais de Unidades Básicas de Saúde (UBS) brasileiras.

Responda dúvidas sobre:
  - Protocolos cardíacos do SUS (Caderno de Atenção Básica nº 14)
  - Sinais e sintomas cardiovasculares
  - Quando acionar o SAMU 192
  - Medicamentos do protocolo SUS (hipertensão, diabetes, colesterol)
  - Procedimentos de suporte básico de vida (SBV)
  - Escore de Framingham e fatores de risco
  - Orientações para pacientes e familiares

Regras:
  - Linguagem simples e acessível para agentes de saúde não médicos
  - Respostas objetivas e práticas — o agente está no campo
  - Sempre reforce: em emergências, acionar SAMU 192 imediatamente
  - Nunca substitua avaliação médica presencial
  - Se a dúvida estiver fora do escopo cardíaco/SUS, redirecione gentilmente
"""


def chat_with_beatsafe(message: str, chat_history: list):
    """
    Streaming chatbot — yields partial responses token by token.
    """

    import ollama

    messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    stream = ollama.chat(model="gemma4:e4b", messages=messages, stream=True)

    chat_history = chat_history + [(message, "")]
    for chunk in stream:
        token = chunk["message"]["content"]
        chat_history[-1] = (message, chat_history[-1][1] + token)
        yield "", chat_history


MEDICATION_SYSTEM_PROMPT = """
Você é um auxiliar farmacológico do BeatSafe, baseado exclusivamente no
Caderno de Atenção Básica nº 14 (Ministério da Saúde, 2006) e na RENAME
(Relação Nacional de Medicamentos Essenciais).

Dado o quadro clínico informado, sugira medicamentos do protocolo SUS
seguindo esta estrutura:

MEDICAMENTOS SUGERIDOS (Protocolo SUS):
  [Liste cada medicamento com: nome genérico, dose, via, frequência e indicação]

CONTRAINDICAÇÕES IMPORTANTES:
  [Liste as principais contraindicações para o quadro descrito]

INTERAÇÕES A VERIFICAR:
  [Interações relevantes com medicamentos comuns]

ORIENTAÇÕES AO AGENTE:
  [O que verificar antes de administrar ou orientar]

Regras absolutas:
  - Apenas medicamentos da RENAME/protocolo SUS
  - Nunca sugira medicamentos sem indicação clara no protocolo
  - Sempre reforce que prescrição é ato médico exclusivo
  - Em emergências: SAMU 192 antes de qualquer medicação
"""


def suggest_medications(clinical_info: str):
    """
    Streaming medication suggestion based on SUS protocols.
    Always includes disclaimer that prescription is a medical act.
    """

    import ollama

    if not clinical_info.strip():
        yield "Por favor, descreva o quadro clínico do paciente."
        return

    stream = ollama.chat(
        model="gemma4:e4b",
        messages=[
            {"role": "system", "content": MEDICATION_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Quadro clínico: {clinical_info}"}
        ],
        stream=True
    )

    accumulated = ""
    for chunk in stream:
        accumulated += chunk["message"]["content"]
        yield accumulated


def load_stats() -> str:
    """Returns an HTML summary of triage statistics."""
    s = get_stats()
    return f"""
    <div style="display: flex; gap: 12px; margin: 8px 0;">
        <div style="flex:1; background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px; padding:16px; text-align:center;">
            <div style="color:#e0e0e0; font-size:1.8rem; font-weight:600;">{s['total']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; margin-top:4px;">Total</div>
        </div>
        <div style="flex:1; background:#1a1a1a; border:1px solid #e63946; border-radius:8px; padding:16px; text-align:center;">
            <div style="color:#e63946; font-size:1.8rem; font-weight:600;">{s['alto_risco']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; margin-top:4px;">Alto Risco</div>
        </div>
        <div style="flex:1; background:#1a1a1a; border:1px solid #ffc107; border-radius:8px; padding:16px; text-align:center;">
            <div style="color:#ffc107; font-size:1.8rem; font-weight:600;">{s['moderado']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; margin-top:4px;">Moderado</div>
        </div>
        <div style="flex:1; background:#1a1a1a; border:1px solid #28a745; border-radius:8px; padding:16px; text-align:center;">
            <div style="color:#28a745; font-size:1.8rem; font-weight:600;">{s['baixo_risco']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; margin-top:4px;">Baixo Risco</div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI — Five tabs: Nova Triagem | Histórico | Assistente | Unidades | Medicamentos
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="BeatSafe") as app:

    # ── Header ──
    gr.HTML("""
        <div class="header-block">
            <h1>BEATSAFE</h1>
            <p>Triagem Cardíaca Offline · Atenção Básica · Gemma 4 + Gemini Flash</p>
        </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════
        # TAB 1 — NOVA TRIAGEM
        # ══════════════════════════════════════════════════════════
        with gr.Tab("⚡ Nova Triagem"):

            with gr.Row():

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">Identificação do Paciente</div>')
                    patient_name = gr.Textbox(
                        label="Nome do Paciente (opcional)",
                        placeholder="Ex: João Silva",
                        lines=1
                    )
                    age = gr.Number(
                        label="Idade (anos)",
                        value=50,
                        minimum=1,
                        maximum=120
                    )
                    sex = gr.Radio(
                        label="Sexo Biológico",
                        choices=["Masculino", "Feminino"],
                        value="Masculino"
                    )
                    vitals = gr.Textbox(
                        label="Sinais Vitais",
                        placeholder="Ex: PA 160/100 mmHg, FC 98 bpm, SatO2 94%",
                        lines=2
                    )

                    gr.HTML('<div class="section-title">Imagem de ECG (opcional)</div>')
                    ecg_image = gr.Image(
                        label="Enviar ECG — Gemini Flash irá analisar",
                        type="filepath",
                        sources=["upload"]
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-title">Dados Clínicos</div>')
                    chief_complaint = gr.Textbox(
                        label="Queixa Principal",
                        placeholder="Ex: Dor no peito há 2 horas, tipo aperto, irradiando para o braço esquerdo",
                        lines=2
                    )
                    symptoms = gr.Textbox(
                        label="Sintomas Associados",
                        placeholder="Ex: Falta de ar, sudorese fria, náusea, tontura",
                        lines=3
                    )
                    history = gr.Textbox(
                        label="Histórico Médico / Comorbidades",
                        placeholder="Ex: Hipertensão há 10 anos, Diabetes tipo 2, ex-tabagista",
                        lines=3
                    )
                    medications = gr.Textbox(
                        label="Medicamentos em Uso",
                        placeholder="Ex: Metformina 850mg, Losartana 50mg, Aspirina 100mg",
                        lines=2
                    )

            triage_btn = gr.Button(
                "Iniciar Triagem Cardíaca",
                elem_classes=["triage-btn"]
            )

            gr.HTML('<div class="section-title">Resultado da Avaliação</div>')

            with gr.Row():
                with gr.Column():
                    symptom_output = gr.Textbox(
                        label="Triagem por Sintomas — Gemma 4 (Local)",
                        lines=15,
                        interactive=False,
                        elem_classes=["output-box"]
                    )
                with gr.Column():
                    ecg_output = gr.Textbox(
                        label="Análise do ECG — Gemini Flash (Nuvem)",
                        lines=15,
                        interactive=False,
                        elem_classes=["output-box"]
                    )

            final_output = gr.Textbox(
                label="Recomendação Final Combinada",
                lines=12,
                interactive=False,
                elem_classes=["output-box"]
            )

            gr.HTML('<div class="section-title">Relatório Clínico</div>')
            pdf_download = gr.File(
                label="Baixar Relatório PDF — compartilhe via WhatsApp com o médico",
                interactive=False
            )

            gr.Examples(
                label="Casos Demonstrativos",
                examples=[
                    [
                        "João Silva", 58, "Masculino",
                        "Dor no peito há 2 horas, tipo aperto, irradiando para o braço esquerdo",
                        "Sudorese fria, falta de ar, náusea",
                        "PA 165/105 mmHg, FC 102 bpm, SatO2 91%",
                        "Hipertensão há 10 anos, Diabetes tipo 2, ex-tabagista",
                        "Metformina, Losartana",
                        None
                    ],
                    [
                        "Maria Oliveira", 52, "Feminino",
                        "Consulta de rotina, sem dor no momento",
                        "Cansaço leve ocasional",
                        "PA 138/88 mmHg, FC 78 bpm, SatO2 98%",
                        "Hipertensão controlada, sobrepeso IMC 28. Pai faleceu de IAM aos 62 anos",
                        "Hidroclorotiazida 25mg",
                        None
                    ],
                ],
                inputs=[patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications, ecg_image]
            )

            gr.HTML("""
                <div class="footer-note">
                    O BeatSafe apoia agentes de saúde — não substitui avaliação médica.
                    Triagem por sintomas roda offline via Gemma 4. Análise de ECG requer internet via Gemini Flash.
                    Em caso de dúvida, encaminhe o paciente para avaliação presencial. · SAMU 192
                </div>
            """)

            triage_btn.click(
                fn=run_full_analysis,
                inputs=[
                    patient_name, age, sex,
                    chief_complaint, symptoms, vitals, history, medications,
                    ecg_image
                ],
                outputs=[symptom_output, ecg_output, final_output, pdf_download]
            )

        # ══════════════════════════════════════════════════════════
        # TAB 2 — HISTÓRICO DE TRIAGENS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("📋 Histórico"):

            gr.HTML('<div class="section-title">Resumo de Atendimentos</div>')
            stats_html = gr.HTML(load_stats)

            gr.HTML('<div class="section-title">Triagens Realizadas</div>')
            with gr.Row():
                risk_filter = gr.Dropdown(
                    label="Filtrar por Risco",
                    choices=["Todos", "🔴 Alto Risco", "🟡 Moderado", "🟢 Baixo Risco"],
                    value="Todos",
                    scale=2
                )
                refresh_btn = gr.Button("🔄 Atualizar", scale=1)

            history_table = gr.Dataframe(
                headers=["Data/Hora", "Paciente", "Idade", "Risco", "Queixa Principal", "Sinais Vitais"],
                datatype=["str", "str", "number", "str", "str", "str"],
                value=load_history("Todos"),
                interactive=False,
                wrap=True,
                column_widths=["130px", "120px", "60px", "110px", "280px", "200px"]
            )

            risk_filter.change(fn=load_history, inputs=[risk_filter], outputs=[history_table])
            refresh_btn.click(
                fn=lambda f: (load_history(f), load_stats()),
                inputs=[risk_filter],
                outputs=[history_table, stats_html]
            )

            gr.HTML("""
                <div class="footer-note">
                    Histórico armazenado localmente em beatsafe_history.db — nenhum dado sai do dispositivo.
                </div>
            """)

        # ══════════════════════════════════════════════════════════
        # TAB 3 — ASSISTENTE DE DÚVIDAS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("💬 Assistente"):

            gr.HTML('<div class="section-title">Assistente Clínico — Gemma 4 (Offline)</div>')
            gr.HTML("""
                <div style="color:#888; font-size:0.78rem; margin-bottom:12px;">
                    Tire dúvidas sobre protocolos do SUS, sinais de alerta e medicamentos.
                    Funciona 100% offline via Gemma 4.
                </div>
            """)

            chatbot = gr.Chatbot(label="", height=420, show_label=False)

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ex: Quando devo acionar o SAMU? O que é fibrilação atrial?",
                    lines=1, scale=5, show_label=False, container=False
                )
                chat_btn = gr.Button("Enviar", scale=1, variant="primary")

            gr.HTML("""
                <div class="footer-note">
                    Opera offline via Gemma 4 · Não substitui avaliação médica · SAMU 192
                </div>
            """)

            chat_btn.click(fn=chat_with_beatsafe, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
            chat_input.submit(fn=chat_with_beatsafe, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])

        # ══════════════════════════════════════════════════════════
        # TAB 4 — UNIDADES DE SAÚDE PRÓXIMAS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🏥 Unidades Próximas"):

            gr.HTML('<div class="section-title">Hospitais e Postos de Saúde Próximos</div>')
            gr.HTML("""
                <div style="color:#888; font-size:0.78rem; margin-bottom:12px;">
                    Localiza UBS, UPA, hospitais e SAMU próximos à sua localização atual.
                    Requer conexão com internet e permissão de localização no navegador.
                </div>
            """)

            gr.HTML("""
                <div style="background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px; padding:20px;">
                    <div style="color:#aaa; font-size:0.78rem; margin-bottom:20px;">
                        Clique no botão desejado para abrir o Google Maps com as unidades mais próximas de você.
                    </div>
                    <div style="display:flex; flex-direction:column; gap:12px;">
                        <a href="https://www.google.com/maps/search/hospital+próximo" target="_blank"
                           style="display:block; background:#e63946; color:white; border-radius:4px;
                                  padding:14px 16px; font-size:0.85rem; text-decoration:none; text-align:center;">
                            HOSPITAIS
                        </a>
                        <a href="https://www.google.com/maps/search/UBS+posto+de+saúde+próximo" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  text-decoration:none; text-align:center;">
                            POSTO DE SAÚDE
                        </a>
                        <a href="https://www.google.com/maps/search/farmácia+próxima" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  text-decoration:none; text-align:center;">
                            FARMÁCIAS
                        </a>
                        <a href="https://www.google.com/maps/search/UPA+próxima" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  text-decoration:none; text-align:center;">
                            UPA
                        </a>
                        <a href="https://www.google.com/maps/search/pronto+socorro+próximo" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  text-decoration:none; text-align:center;">
                            PRONTO-SOCORRO
                        </a>
                        <a href="https://www.google.com/maps/search/SAMU+192" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  text-decoration:none; text-align:center;">
                            SAMU 192
                        </a>
                    </div>
                    <div style="margin-top:20px; padding:12px; background:#111;
                                border-radius:4px; border-left:3px solid #e63946;">
                        <div style="color:#e63946; font-size:0.75rem; letter-spacing:2px; font-weight:700;">
                            EM CASO DE EMERGÊNCIA
                        </div>
                        <div style="color:#e0e0e0; font-size:0.85rem; margin-top:4px;">
                            Ligue <strong>SAMU 192</strong> imediatamente — não espere pelo mapa.
                        </div>
                    </div>
                </div>
                <div class="footer-note" style="margin-top:16px;">
                    Abre o Google Maps · Nenhum dado é enviado ao BeatSafe.
                </div>
            """)

        # ══════════════════════════════════════════════════════════
        # TAB 5 — SUGESTÃO DE MEDICAMENTOS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("💊 Medicamentos"):

            gr.HTML('<div class="section-title">Sugestão de Medicamentos — Protocolo SUS</div>')

            gr.HTML("""
                <div style="background:#1a1a1a; border-left:4px solid #e63946;
                            border-radius:4px; padding:12px 16px; margin-bottom:16px;">
                    <div style="color:#e63946; font-size:0.75rem; font-weight:600;
                                letter-spacing:1px; margin-bottom:4px;">
                        AVISO IMPORTANTE
                    </div>
                    <div style="color:#aaa; font-size:0.82rem; line-height:1.6;">
                        As sugestões abaixo são baseadas exclusivamente no
                        <strong style="color:#e0e0e0;">Caderno de Atenção Básica nº 14</strong>
                        e na <strong style="color:#e0e0e0;">RENAME (Ministério da Saúde)</strong>.
                        A prescrição é ato médico exclusivo. Em emergências, acione o
                        <strong style="color:#e63946;">SAMU 192</strong> antes de qualquer medicação.
                    </div>
                </div>
            """)

            med_input = gr.Textbox(
                label="Descreva o quadro clínico",
                placeholder="Ex: Paciente hipertenso, 60 anos, PA 170/110 mmHg, sem dor no peito, tomou losartana hoje",
                lines=3
            )

            med_btn = gr.Button("Consultar Protocolo SUS", elem_classes=["triage-btn"])

            med_output = gr.Textbox(
                label="Sugestão baseada no Protocolo SUS",
                lines=18,
                interactive=False,
                elem_classes=["output-box"]
            )

            gr.HTML("""
                <div class="footer-note">
                    Baseado no Caderno AB nº14 e RENAME · Prescrição é ato médico exclusivo · SAMU 192
                </div>
            """)

            med_btn.click(fn=suggest_medications, inputs=[med_input], outputs=[med_output])

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        share=True,
        show_error=True
    )
