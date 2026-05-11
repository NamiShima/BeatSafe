import gradio as gr              # Web interface library - creates the browser UI
import tempfile                  # Handle temporary files for ECG image upload
import os                        # File path operations

# Import core triage and ECG analysis functions
from main import triage
from ecg import analyze_ecg, combined_analysis

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Visual styling for the BeatSafe interface
# Dark medical theme with red accent — clean and professional
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
"""


def run_triage_only(patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications):
    """
    Runs symptom-only triage using Gemma 3 local model (offline).
    Called when no ECG image is provided.

    Args:
        patient_name, age, sex: Patient identity fields
        chief_complaint, symptoms, vitals, history, medications: Clinical data

    Returns:
        Structured triage assessment from Gemma 3 local
    """

    # Assemble patient report from form fields
    patient_info = f"""
    Paciente: {sex}, {age} anos {"— " + patient_name if patient_name else ""}
    Queixa principal: {chief_complaint}
    Sintomas associados: {symptoms}
    Sinais vitais: {vitals}
    Histórico médico / Comorbidades: {history}
    Medicamentos em uso: {medications}
    """

    # Run offline symptom triage with local Gemma 3
    return triage(patient_info)


def run_full_analysis(patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications, ecg_image):
    """
    Runs the complete BeatSafe pipeline when an ECG image is provided:
        Step 1 — Symptom triage via Gemma 3 local (offline)
        Step 2 — ECG analysis via Gemma 4 API (cloud)
        Step 3 — Combined final recommendation

    Args:
        patient_name, age, sex: Patient identity fields
        chief_complaint, symptoms, vitals, history, medications: Clinical data
        ecg_image: Uploaded ECG image file path from Gradio

    Returns:
        Tuple of three strings:
            - symptom result
            - ECG analysis result
            - combined final recommendation
    """

    # Assemble patient report from form fields
    patient_info = f"""
    Paciente: {sex}, {age} anos {"— " + patient_name if patient_name else ""}
    Queixa principal: {chief_complaint}
    Sintomas associados: {symptoms}
    Sinais vitais: {vitals}
    Histórico médico / Comorbidades: {history}
    Medicamentos em uso: {medications}
    """

    # Step 1 — Offline symptom triage with Gemma 3 local
    symptom_result = triage(patient_info)

    # Step 2 — ECG image analysis with Gemma 4 cloud (only if image provided)
    if ecg_image is not None:
        ecg_result = analyze_ecg(ecg_image)
        # Step 3 — Combine both results into unified recommendation
        final_result = combined_analysis(symptom_result, ecg_result)
    else:
        # No ECG provided — skip image analysis
        ecg_result = "Nenhuma imagem de ECG fornecida."
        final_result = symptom_result

    return symptom_result, ecg_result, final_result


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI — Builds the browser interface with two modes:
#   Mode 1: Symptom triage only (Gemma 3 local, offline)
#   Mode 2: Full analysis with ECG image (Gemma 3 + Gemma 4)
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="BeatSafe") as app:

    # ── Header ──
    gr.HTML("""
        <div class="header-block">
            <h1>🫀 BEATSAFE</h1>
            <p>Offline Cardiac Triage AI · Brazilian Primary Care · Gemma 3 + Gemma 4</p>
        </div>
    """)

    # ── Patient form — two columns ──
    with gr.Row():

        # Left column — patient identity and vitals
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

            # ECG image upload — triggers full analysis with Gemma 4
            gr.HTML('<div class="section-title">ECG Image (optional)</div>')
            ecg_image = gr.Image(
                label="Upload ECG Image — Gemma 4 will analyze it",
                type="filepath",    # Returns file path for the analyze_ecg() function
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

    # ── Output section — three panels ──
    gr.HTML('<div class="section-title">Clinical Assessment</div>')

    with gr.Row():
        # Panel 1 — Symptom triage result (Gemma 3 local)
        with gr.Column():
            symptom_output = gr.Textbox(
                label="🫀 Symptom Triage — Gemma 3 (Local)",
                lines=15,
                interactive=False,
                elem_classes=["output-box"]
            )

        # Panel 2 — ECG analysis result (Gemma 4 cloud)
        with gr.Column():
            ecg_output = gr.Textbox(
                label="📊 ECG Analysis — Gemma 4 (Cloud)",
                lines=15,
                interactive=False,
                elem_classes=["output-box"]
            )

    # Panel 3 — Combined final recommendation (full width)
    final_output = gr.Textbox(
        label="🎯 Final Combined Recommendation — Gemma 4",
        lines=12,
        interactive=False,
        elem_classes=["output-box"]
    )

    # ── Pre-loaded example cases ──
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
                None  # No ECG image for this demo case
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

    # ── Footer disclaimer ──
    gr.HTML("""
        <div class="footer-note">
            BeatSafe supports health workers — it does not replace medical evaluation.
            Symptom triage runs offline via Gemma 3. ECG analysis requires internet via Gemma 4.
            When in doubt, always refer the patient for in-person assessment. · SAMU 192
        </div>
    """)

    # ── Connect button to analysis function ──
    triage_btn.click(
        fn=run_full_analysis,
        inputs=[patient_name, age, sex, chief_complaint, symptoms, vitals, history, medications, ecg_image],
        outputs=[symptom_output, ecg_output, final_output]
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH — Starts the local web server
# share=True generates a public link for the Kaggle submission demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        share=True,      # Generates public Hugging Face link for demo
        show_error=True
    )
