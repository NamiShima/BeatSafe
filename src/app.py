import gradio as gr              # Web interface library
import os                        # File path operations
import tempfile                  # Temporary file handling for PDF download

# Import core BeatSafe modules
from main import triage
from ecg import analyze_ecg, combined_analysis
from pdf_report import generate_pdf

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
"""


def run_full_analysis(
    patient_name, age, sex,
    chief_complaint, symptoms, vitals, history, medications,
    ecg_image
):
    """
    Runs the complete BeatSafe pipeline and generates a PDF report.

    Step 1 — Symptom triage via Gemma 3 local (offline)
    Step 2 — ECG image analysis via Gemini 2.5 Flash (cloud, if image provided)
    Step 3 — Combined final recommendation (if ECG provided)
    Step 4 — PDF report generation for download and WhatsApp sharing

    Args:
        patient_name, age, sex: Patient identity fields
        chief_complaint, symptoms, vitals, history, medications: Clinical data
        ecg_image: Uploaded ECG image file path (optional)

    Returns:
        Tuple of four values:
            - symptom_output: Gemma 3 triage result
            - ecg_output:     ECG analysis result
            - final_output:   Combined recommendation
            - pdf_file:       Path to generated PDF for download
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
        # Step 3 — Combine both results into final recommendation
        final_result = combined_analysis(symptom_result, ecg_result)
    else:
        # No ECG provided — symptom triage is the final result
        ecg_result = "Nenhuma imagem de ECG fornecida."
        final_result = symptom_result

    # Step 4 — Generate PDF report for download
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

    return symptom_result, ecg_result, final_result, pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI — Browser interface with three output panels and PDF download
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="BeatSafe") as app:

    # ── Header ──
    gr.HTML("""
        <div class="header-block">
            <h1>🫀 BEATSAFE</h1>
            <p>Offline Cardiac Triage AI · Brazilian Primary Care · Gemma 3 + Gemma 4</p>
        </div>
    """)

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

            # ECG upload — triggers Gemini analysis
            gr.HTML('<div class="section-title">ECG Image (optional)</div>')
            ecg_image = gr.Image(
                label="Upload ECG — Gemini 2.5 Flash will analyze it",
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
        # Panel 1 — Symptom triage (Gemma 3 local)
        with gr.Column():
            symptom_output = gr.Textbox(
                label="🫀 Symptom Triage — Gemma 3 (Local)",
                lines=15,
                interactive=False,
                elem_classes=["output-box"]
            )

        # Panel 2 — ECG analysis (Gemini cloud)
        with gr.Column():
            ecg_output = gr.Textbox(
                label="📊 ECG Analysis — Gemini 2.5 Flash (Cloud)",
                lines=15,
                interactive=False,
                elem_classes=["output-box"]
            )

    # Panel 3 — Combined recommendation (full width)
    final_output = gr.Textbox(
        label="🎯 Final Combined Recommendation",
        lines=12,
        interactive=False,
        elem_classes=["output-box"]
    )

    # ── PDF download button — appears after triage runs ──
    gr.HTML('<div class="section-title">Clinical Report</div>')
    pdf_download = gr.File(
        label="📋 Download PDF Report — share via WhatsApp with the doctor",
        interactive=False
    )

    # ── Pre-loaded demo cases ──
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
            Symptom triage runs offline via Gemma 3. ECG analysis requires internet via Gemini 2.5 Flash.
            When in doubt, always refer the patient for in-person assessment. · SAMU 192
        </div>
    """)

    # ── Connect button to analysis function ──
    triage_btn.click(
        fn=run_full_analysis,
        inputs=[
            patient_name, age, sex,
            chief_complaint, symptoms, vitals, history, medications,
            ecg_image
        ],
        outputs=[symptom_output, ecg_output, final_output, pdf_download]
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH — Starts the local web server
# share=True generates a public Hugging Face link for the Kaggle demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        share=True,
        show_error=True
    )
