import gradio as gr  # Web interface library - creates the browser UI
import ollama        # Local inference library - connects to Gemma running on the machine

# Import the clinical system prompt from the core triage engine
from main import SYSTEM_PROMPT, triage

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Visual styling for the BeatSafe interface
# Dark medical theme with red accent — clean and professional
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global reset and base font ── */
* {
    font-family: 'Courier New', monospace;
    box-sizing: border-box;
}

/* ── Main app background ── */
.gradio-container {
    background-color: #0d0d0d !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* ── Header block ── */
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

/* ── Section labels ── */
label, .label-wrap {
    color: #aaa !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* ── Input fields ── */
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

/* ── Triage button ── */
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

/* ── Output result box ── */
.output-box textarea {
    background-color: #111 !important;
    border: 1px solid #2a2a2a !important;
    color: #e0e0e0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    min-height: 380px !important;
}

/* ── Example cases row ── */
.examples-row {
    margin-top: 16px;
}

/* ── Footer disclaimer ── */
.footer-note {
    text-align: center;
    color: #555;
    font-size: 0.72rem;
    letter-spacing: 1px;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #1a1a1a;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# TRIAGE FUNCTION FOR GRADIO
# Wraps the core triage() function to work with Gradio's UI inputs
# ─────────────────────────────────────────────────────────────────────────────
def run_triage(patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications):
    """
    Receives structured form fields from the Gradio UI,
    assembles them into a patient report string,
    and sends it to the Gemma model for triage.

    Args:
        patient_name:     Patient's name (optional, for reference only)
        age:              Patient's age in years
        sex:              Biological sex (Male/Female)
        chief_complaint:  Main reason for the visit
        symptoms:         Associated symptoms reported by patient
        vitals:           Vital signs: BP, HR, SpO2
        history:          Medical history, comorbidities, prior events
        medications:      Current medications in use

    Returns:
        Formatted clinical triage assessment from Gemma
    """

    # Build the patient report string from individual form fields
    patient_info = f"""
    Paciente: {sex}, {age} anos {"— " + patient_name if patient_name else ""}
    Queixa principal: {chief_complaint}
    Sintomas associados: {symptoms}
    Sinais vitais: {vitals}
    Histórico médico / Comorbidades: {history}
    Medicamentos em uso: {medications}
    """

    # Send assembled report to Gemma via Ollama and return the response
    return triage(patient_info)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI — Builds the browser interface
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="BeatSafe") as app:

    # ── Header ──
    gr.HTML("""
        <div class="header-block">
            <h1>🫀 BEATSAFE</h1>
            <p>Offline Cardiac Triage AI · Brazilian Primary Care · Powered by Gemma 3</p>
        </div>
    """)

    # ── Input form — two columns ──
    with gr.Row():

        # Left column — patient identity
        with gr.Column(scale=1):
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

        # Right column — clinical data
        with gr.Column(scale=2):
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
                placeholder="e.g. Hypertension for 10 years, Type 2 Diabetes, former smoker, father had MI at 55",
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

    # ── Output result ──
    result = gr.Textbox(
        label="Clinical Assessment",
        lines=18,
        interactive=False,
        elem_classes=["output-box"]
    )

    # ── Pre-loaded example cases for quick demo ──
    gr.Examples(
        label="Quick Demo Cases",
        examples=[
            [
                "João Silva",
                58,
                "Male",
                "Chest pain for 2 hours, pressure-like, radiating to left arm",
                "Cold sweating, shortness of breath, nausea",
                "BP 165/105 mmHg, HR 102 bpm, SpO2 91%",
                "Hypertension 10 years, Type 2 Diabetes, former smoker",
                "Metformin, Losartan"
            ],
            [
                "Maria Oliveira",
                52,
                "Female",
                "Routine visit, no pain currently",
                "Occasional mild fatigue",
                "BP 138/88 mmHg, HR 78 bpm, SpO2 98%",
                "Controlled hypertension, overweight BMI 28, sedentary. Father died of MI at 62",
                "Hydrochlorothiazide 25mg"
            ],
            [
                "",
                70,
                "Male",
                "Unresponsive, not breathing",
                "No pulse detected, cyanosis",
                "BP undetectable, HR absent, SpO2 undetectable",
                "Unknown",
                "Unknown"
            ]
        ],
        inputs=[patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications]
    )

    # ── Footer disclaimer ──
    gr.HTML("""
        <div class="footer-note">
            BeatSafe supports health workers — it does not replace medical evaluation.
            When in doubt, always refer the patient for in-person assessment. · SAMU 192
        </div>
    """)

    # ── Connect button to triage function ──
    triage_btn.click(
        fn=run_triage,
        inputs=[patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications],
        outputs=result
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH — Starts the local web server when this file is executed directly
# Opens automatically in the browser at http://localhost:7860
# share=True generates a public Hugging Face Spaces link for the demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        share=True,   # Generates a public link — needed for Kaggle submission demo
        show_error=True
    )
