import gradio as gr              # Web interface library
import os                        # File path operations
import tempfile                  # Temporary file handling for PDF download

# Import core BeatSafe modules
from main import triage
from ecg import analyze_ecg, combined_analysis
from pdf_report import generate_pdf
from history import save_triage, get_history, get_stats   # SQLite history module

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark medical theme with red accent
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
* {
    font-family: 'Courier New', monospace;
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
    font-weight: 700;
    letter-spacing: 4px;
    margin: 0;
}

.header-block p {
    color: #888;
    font-size: 0.85rem;
    margin: 8px 0 0 0;
    letter-spacing: 2px;
    text-transform: uppercase;
}

label, .label-wrap {
    color: #aaa !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

textarea, input[type="text"] {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e0e0e0 !important;
    font-family: 'Courier New', monospace !important;
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
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
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
    font-family: 'Courier New', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    min-height: 300px !important;
}

.section-title {
    color: #e63946;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 8px;
    margin: 16px 0 12px 0;
}

.footer-note {
    text-align: center;
    color: #555;
    font-size: 0.72rem;
    letter-spacing: 1px;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #1a1a1a;
}

.stat-box {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 16px;
    text-align: center;
}

.stat-box h2 {
    color: #e63946;
    font-size: 2rem;
    margin: 0;
}

.stat-box p {
    color: #888;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 4px 0 0 0;
}
"""


def run_full_analysis(
    patient_name, age, sex,
    chief_complaint, symptoms, vitals, history, medications,
    ecg_image
):
    """
    Runs the complete BeatSafe pipeline and generates a PDF report.
    Automatically saves the triage to the SQLite history database.

    Step 1 — Symptom triage via Gemma 3 local (offline)
    Step 2 — ECG image analysis via Gemini 2.5 Flash (cloud, if image provided)
    Step 3 — Combined final recommendation (if ECG provided)
    Step 4 — PDF report generation for download and WhatsApp sharing
    Step 5 — Save record to history database

    Returns:
        Tuple: symptom_output, ecg_output, final_output, pdf_file
    """

    # Assemble patient report string from form fields
    patient_info = f"""
    Paciente: {sex}, {age} anos {"— " + patient_name if patient_name else ""}
    Queixa principal: {chief_complaint}
    Sintomas associados: {symptoms}
    Sinais vitais: {vitals}
    Histórico médico / Comorbidades: {history}
    Medicamentos em uso: {medications}
    """

    # Step 1 — Offline symptom triage using local Gemma 3
    symptom_result = triage(patient_info)

    # Step 2 — ECG analysis using Gemini (only if image was uploaded)
    if ecg_image is not None:
        ecg_result = analyze_ecg(ecg_image)
        final_result = combined_analysis(symptom_result, ecg_result)
    else:
        ecg_result = "Nenhuma imagem de ECG fornecida."
        final_result = symptom_result

    # Step 3 — Generate PDF report for download
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

    # Step 4 — Save triage to SQLite history database
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

    return symptom_result, ecg_result, final_result, pdf_path


def load_history(risk_filter: str) -> list:
    """Fetches triage history from SQLite filtered by risk level."""
    return get_history(risk_filter=risk_filter)


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT — Gemma 3 offline answering clinical questions from health agents
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


def chat_with_beatsafe(message: str, chat_history: list) -> tuple:
    """
    Sends a question to Gemma 3 (local, offline) and returns the answer.
    Maintains full conversation history for context-aware follow-up questions.

    Args:
        message:      Current question from the health agent
        chat_history: List of previous [user, assistant] message pairs

    Returns:
        Tuple of (empty string to clear input, updated chat history)
    """

    import ollama

    # Build messages list with full conversation history for context
    messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Send to local Gemma 3 — no internet required
    response = ollama.chat(model="gemma3:4b", messages=messages)
    answer = response["message"]["content"]

    # Append new exchange to history and clear input
    chat_history.append((message, answer))
    return "", chat_history


def load_stats() -> str:
    """Returns an HTML summary of triage statistics."""
    s = get_stats()
    return f"""
    <div style="display: flex; gap: 12px; margin: 8px 0;">
        <div class="stat-box" style="flex:1; background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px; padding:16px; text-align:center;">
            <div style="color:#e0e0e0; font-size:1.8rem; font-weight:700;">{s['total']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">Total</div>
        </div>
        <div class="stat-box" style="flex:1; background:#1a1a1a; border:1px solid #e63946; border-radius:6px; padding:16px; text-align:center;">
            <div style="color:#e63946; font-size:1.8rem; font-weight:700;">{s['alto_risco']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">Alto Risco</div>
        </div>
        <div class="stat-box" style="flex:1; background:#1a1a1a; border:1px solid #ffc107; border-radius:6px; padding:16px; text-align:center;">
            <div style="color:#ffc107; font-size:1.8rem; font-weight:700;">{s['moderado']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">Moderado</div>
        </div>
        <div class="stat-box" style="flex:1; background:#1a1a1a; border:1px solid #28a745; border-radius:6px; padding:16px; text-align:center;">
            <div style="color:#28a745; font-size:1.8rem; font-weight:700;">{s['baixo_risco']}</div>
            <div style="color:#888; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">Baixo Risco</div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI — Two tabs: Nova Triagem | Histórico
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="BeatSafe") as app:

    # ── Header — shared across both tabs ──
    gr.HTML("""
        <div class="header-block">
            <h1>🫀 BEATSAFE</h1>
            <p>Offline Cardiac Triage AI · Brazilian Primary Care · Gemma 3 + Gemini Flash</p>
        </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════
        # TAB 1 — NOVA TRIAGEM
        # ══════════════════════════════════════════════════════════
        with gr.Tab("⚡ Nova Triagem"):

            # ── Input form — two columns ──
            with gr.Row():

                # Left column — patient identity and ECG upload
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">Patient Identity</div>')
                    patient_name = gr.Textbox(
                        label="Patient Name (optional)",
                        placeholder="e.g. João Silva",
                        lines=1
                    )
                    age = gr.Number(
                        label="Age (years)",
                        value=50,
                        minimum=1,
                        maximum=120
                    )
                    sex = gr.Radio(
                        label="Biological Sex",
                        choices=["Male", "Female"],
                        value="Male"
                    )
                    vitals = gr.Textbox(
                        label="Vital Signs",
                        placeholder="e.g. BP 160/100 mmHg, HR 98 bpm, SpO2 94%",
                        lines=2
                    )

                    gr.HTML('<div class="section-title">ECG Image (optional)</div>')
                    ecg_image = gr.Image(
                        label="Upload ECG — Gemini Flash will analyze it",
                        type="filepath",
                        sources=["upload"]
                    )

                # Right column — clinical data
                with gr.Column(scale=2):
                    gr.HTML('<div class="section-title">Clinical Data</div>')
                    chief_complaint = gr.Textbox(
                        label="Chief Complaint",
                        placeholder="e.g. Chest pain for 2 hours, pressure-like, radiating to left arm",
                        lines=2
                    )
                    symptoms = gr.Textbox(
                        label="Associated Symptoms",
                        placeholder="e.g. Shortness of breath, cold sweating, nausea, dizziness",
                        lines=3
                    )
                    history = gr.Textbox(
                        label="Medical History / Comorbidities",
                        placeholder="e.g. Hypertension for 10 years, Type 2 Diabetes, former smoker",
                        lines=3
                    )
                    medications = gr.Textbox(
                        label="Current Medications",
                        placeholder="e.g. Metformin 850mg, Losartan 50mg, Aspirin 100mg",
                        lines=2
                    )

            # ── Triage button ──
            triage_btn = gr.Button(
                "⚡ RUN CARDIAC TRIAGE",
                elem_classes=["triage-btn"]
            )

            # ── Output panels ──
            gr.HTML('<div class="section-title">Clinical Assessment</div>')

            with gr.Row():
                with gr.Column():
                    symptom_output = gr.Textbox(
                        label="🫀 Symptom Triage — Gemma 3 (Local)",
                        lines=15,
                        interactive=False,
                        elem_classes=["output-box"]
                    )
                with gr.Column():
                    ecg_output = gr.Textbox(
                        label="📊 ECG Analysis — Gemini Flash (Cloud)",
                        lines=15,
                        interactive=False,
                        elem_classes=["output-box"]
                    )

            final_output = gr.Textbox(
                label="🎯 Final Combined Recommendation",
                lines=12,
                interactive=False,
                elem_classes=["output-box"]
            )

            gr.HTML('<div class="section-title">Clinical Report</div>')
            pdf_download = gr.File(
                label="📋 Download PDF Report — share via WhatsApp with the doctor",
                interactive=False
            )

            # ── Demo cases ──
            gr.Examples(
                label="Quick Demo Cases",
                examples=[
                    [
                        "João Silva", 58, "Male",
                        "Chest pain for 2 hours, pressure-like, radiating to left arm",
                        "Cold sweating, shortness of breath, nausea",
                        "BP 165/105 mmHg, HR 102 bpm, SpO2 91%",
                        "Hypertension 10 years, Type 2 Diabetes, former smoker",
                        "Metformin, Losartan",
                        None
                    ],
                    [
                        "Maria Oliveira", 52, "Female",
                        "Routine visit, no pain currently",
                        "Occasional mild fatigue",
                        "BP 138/88 mmHg, HR 78 bpm, SpO2 98%",
                        "Controlled hypertension, overweight BMI 28. Father died of MI at 62",
                        "Hydrochlorothiazide 25mg",
                        None
                    ],
                ],
                inputs=[patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications, ecg_image]
            )

            # ── Footer ──
            gr.HTML("""
                <div class="footer-note">
                    BeatSafe supports health workers — it does not replace medical evaluation.
                    Symptom triage runs offline via Gemma 3. ECG analysis requires internet via Gemini Flash.
                    When in doubt, always refer the patient for in-person assessment. · SAMU 192
                </div>
            """)

            # ── Wire up triage button ──
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

            # ── Stats dashboard ──
            gr.HTML('<div class="section-title">Resumo de Atendimentos</div>')
            stats_html = gr.HTML(load_stats)

            # ── Filter + refresh controls ──
            gr.HTML('<div class="section-title">Triagens Realizadas</div>')
            with gr.Row():
                risk_filter = gr.Dropdown(
                    label="Filtrar por Risco",
                    choices=["Todos", "🔴 Alto Risco", "🟡 Moderado", "🟢 Baixo Risco"],
                    value="Todos",
                    scale=2
                )
                refresh_btn = gr.Button("🔄 Atualizar", scale=1)

            # ── History table — colunas ajustadas para evitar scroll horizontal ──
            history_table = gr.Dataframe(
                headers=["Data/Hora", "Paciente", "Idade", "Risco", "Queixa Principal", "Sinais Vitais"],
                datatype=["str", "str", "number", "str", "str", "str"],
                value=load_history("Todos"),
                interactive=False,
                wrap=True,
                column_widths=["130px", "120px", "60px", "110px", "280px", "200px"]
            )

            # ── Wire up filter and refresh ──
            risk_filter.change(
                fn=load_history,
                inputs=[risk_filter],
                outputs=[history_table]
            )

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

            gr.HTML('<div class="section-title">Assistente Clínico — Gemma 3 (Offline)</div>')
            gr.HTML("""
                <div style="color:#888; font-size:0.78rem; letter-spacing:1px; margin-bottom:12px;">
                    Tire dúvidas clínicas sobre protocolos do SUS, sinais de alerta, medicamentos
                    e procedimentos. Funciona 100% offline via Gemma 3.
                </div>
            """)

            # ── Chat window ──
            chatbot = gr.Chatbot(
                label="",
                height=420,
                show_label=False,
                placeholder="💬 Faça sua pergunta clínica...",
            )

            # ── Input row ──
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ex: Quando devo acionar o SAMU? O que é fibrilação atrial?",
                    lines=1,
                    scale=5,
                    show_label=False,
                    container=False
                )
                chat_btn = gr.Button("Enviar", scale=1, variant="primary")

            # ── Quick question suggestions ──
            gr.HTML('<div class="section-title" style="margin-top:16px;">Perguntas Frequentes</div>')
            with gr.Row():
                q1 = gr.Button("🚨 Quando acionar o SAMU?",        scale=1)
                q2 = gr.Button("💊 Medicamentos para hipertensão",  scale=1)
                q3 = gr.Button("❤️ Como fazer RCP corretamente?",   scale=1)
            with gr.Row():
                q4 = gr.Button("📊 O que é o Escore de Framingham?", scale=1)
                q5 = gr.Button("🩺 Sinais de infarto no ECG",         scale=1)
                q6 = gr.Button("🧂 Meta de pressão arterial no SUS",  scale=1)

            gr.HTML("""
                <div class="footer-note">
                    O Assistente BeatSafe opera offline via Gemma 3 — nenhuma pergunta sai do dispositivo.
                    Não substitui avaliação médica presencial. · SAMU 192
                </div>
            """)

            # ── Wire up send button and Enter key ──
            chat_btn.click(
                fn=chat_with_beatsafe,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )
            chat_input.submit(
                fn=chat_with_beatsafe,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )

            # ── Wire up quick question buttons — preenche o input e envia ──
            def make_quick_question(q):
                def fn(history):
                    return chat_with_beatsafe(q, history)
                return fn

            q1.click(fn=make_quick_question("Quando devo acionar o SAMU 192? Quais são os critérios?"),         inputs=[chatbot], outputs=[chat_input, chatbot])
            q2.click(fn=make_quick_question("Quais medicamentos são usados para hipertensão no protocolo SUS?"), inputs=[chatbot], outputs=[chat_input, chatbot])
            q3.click(fn=make_quick_question("Como fazer RCP (reanimação cardiopulmonar) corretamente?"),         inputs=[chatbot], outputs=[chat_input, chatbot])
            q4.click(fn=make_quick_question("O que é o Escore de Framingham e como ele é calculado?"),           inputs=[chatbot], outputs=[chat_input, chatbot])
            q5.click(fn=make_quick_question("Quais são os sinais de infarto no ECG que devo reconhecer?"),       inputs=[chatbot], outputs=[chat_input, chatbot])
            q6.click(fn=make_quick_question("Qual é a meta de pressão arterial recomendada pelo SUS?"),          inputs=[chatbot], outputs=[chat_input, chatbot])
        # ══════════════════════════════════════════════════════════
        # TAB 4 — UNIDADES DE SAÚDE PRÓXIMAS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🏥 Unidades Próximas"):

            gr.HTML('<div class="section-title">Hospitais e Postos de Saúde Próximos</div>')
            gr.HTML("""
                <div style="color:#888; font-size:0.78rem; letter-spacing:1px; margin-bottom:12px;">
                    Localiza UBS, UPA, hospitais e SAMU próximos à sua localização atual.
                    Requer conexão com internet e permissão de localização no navegador.
                </div>
            """)

            # ── Busca via links diretos Google Maps — sem JavaScript ──
            gr.HTML("""
                <div style="background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px; padding:20px;">

                    <div style="color:#aaa; font-size:0.78rem; margin-bottom:20px; letter-spacing:1px;">
                        Clique no botão desejado para abrir o Google Maps com as unidades mais próximas de você.
                        O navegador vai pedir permissão de localização automaticamente.
                    </div>

                    <div style="display:flex; flex-direction:column; gap:12px;">

                        <a href="https://www.google.com/maps/search/hospital+próximo" target="_blank"
                           style="display:block; background:#e63946; color:white; border-radius:4px;
                                  padding:14px 16px; font-family:monospace; font-size:0.85rem;
                                  letter-spacing:2px; text-decoration:none; text-align:center;">
                            🏥 HOSPITAIS PRÓXIMOS
                        </a>

                        <a href="https://www.google.com/maps/search/UBS+posto+de+saúde+próximo" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  font-family:monospace; font-size:0.85rem; letter-spacing:2px;
                                  text-decoration:none; text-align:center;">
                            🩺 UBS / POSTOS DE SAÚDE
                        </a>

                        <a href="https://www.google.com/maps/search/farmácia+próxima" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  font-family:monospace; font-size:0.85rem; letter-spacing:2px;
                                  text-decoration:none; text-align:center;">
                            💊 FARMÁCIAS
                        </a>

                        <a href="https://www.google.com/maps/search/UPA+pronto+socorro+próximo" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  font-family:monospace; font-size:0.85rem; letter-spacing:2px;
                                  text-decoration:none; text-align:center;">
                            🚑 UPA / PRONTO-SOCORRO
                        </a>

                        <a href="https://www.google.com/maps/search/SAMU+192" target="_blank"
                           style="display:block; background:#1a1a1a; color:#e0e0e0;
                                  border:1px solid #e63946; border-radius:4px; padding:14px 16px;
                                  font-family:monospace; font-size:0.85rem; letter-spacing:2px;
                                  text-decoration:none; text-align:center;">
                            🚨 SAMU 192
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
                    Abre o Google Maps com sua localização atual · Nenhum dado é enviado ao BeatSafe.
                </div>
            """)


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        share=True,
        show_error=True
    )
