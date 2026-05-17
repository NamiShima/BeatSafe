# BeatSafe - Cardiac Triage AI for Offline Primary Care Units in Brazil
# Powered by Gemma 3 (4B) via Ollama - runs fully offline
# Designed for Gemma 4 migration when available on Ollama
# Author: NamiShima
# Competition: Gemma 4 Good Hackathon 2026

import os                        # Access environment variables (API key)
from google import genai         # Google AI API library (google-genai)
from google.genai import types   # Content types for multimodal requests
from dotenv import load_dotenv   # Load .env file securely
from PIL import Image            # Image processing library
import io                        # Handle image bytes in memory

# ─────────────────────────────────────────────────────────────────────────────
# LOAD API KEY — Reads the Gemini API key from the .env file
# This keeps the key out of the source code (safe for GitHub)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv("../.env")  # Load from BeatSafe root folder
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Google AI client with the API key
client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION — Gemini 2.5 Flash handles ECG image analysis (cloud, optional)
# This is the multimodal component of BeatSafe's hybrid architecture:
#   - Core triage: Gemma 3:4b via Ollama (100% offline, no internet required)
#   - ECG analysis: Gemini 2.5 Flash via API (cloud, only when internet available)
# ECG analysis is optional — BeatSafe degrades gracefully without it.
# Future goal: migrate to Gemma 4 multimodal via Ollama when available locally.
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "models/gemini-2.5-flash"

# ─────────────────────────────────────────────────────────────────────────────
# ECG ANALYSIS PROMPT — Instructs the model on how to read the ECG image
# Written in Portuguese to match the clinical context of Brazilian health workers
# ─────────────────────────────────────────────────────────────────────────────
ECG_PROMPT = """
Você é um especialista em eletrocardiografia clínica.
Analise esta imagem de ECG e forneça uma interpretação estruturada seguindo
as diretrizes da Sociedade Brasileira de Cardiologia (SBC).

Avalie obrigatoriamente os seguintes itens:

1. RITMO
   - Sinusal, fibrilação atrial, flutter, taquicardia, bradicardia?
   - Frequência cardíaca estimada (bpm)

2. EIXO ELÉTRICO
   - Normal, desvio para esquerda, desvio para direita?

3. INTERVALO PR
   - Normal (120-200ms)? Prolongado? Curto?

4. COMPLEXO QRS
   - Duração normal (<120ms)? Alargado?
   - Morfologia: bloqueio de ramo esquerdo (BRE)? Bloqueio de ramo direito (BRD)?

5. SEGMENTO ST
   - Supradesnivelamento? Em quais derivações?
   - Infradesnivelamento? Em quais derivações?
   - Padrão normal?

6. ONDA T
   - Normal? Invertida? Apiculada?
   - Em quais derivações?

7. INTERVALO QT
   - Normal? Prolongado? (risco de Torsades de Pointes)

8. ACHADOS RELEVANTES
   - Sinais de isquemia aguda?
   - Sinais de IAM (infarto agudo do miocárdio)?
   - Hipertrofia ventricular?
   - Outras alterações significativas?

9. INTERPRETACAO FINAL
   - ALTERADO - URGENTE: risco imediato a vida
   - ALTERADO - ATENCAO: requer avaliacao medica
   - NORMAL: sem alteracoes significativas

10. RECOMENDACAO PARA O AGENTE DE SAUDE
    - O que fazer AGORA com base neste ECG?

Seja claro, direto e use linguagem acessivel para agentes de saude nao especialistas.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ECG COMBINATION PROMPT — Merges ECG findings with symptom triage
# This is the core of BeatSafe's hybrid architecture:
#   Gemma 3 (offline) handles symptom reasoning
#   Gemini Flash (cloud) handles image analysis
#   Together they produce a more complete clinical picture
# ─────────────────────────────────────────────────────────────────────────────
ECG_COMBINATION_PROMPT = """
Você é um cardiologista de plantão em uma Unidade Básica de Saúde brasileira.

Você recebeu dois relatórios sobre o mesmo paciente:

RELATÓRIO 1 — TRIAGEM POR SINTOMAS:
{symptom_triage}

RELATÓRIO 2 — ANÁLISE DO ECG:
{ecg_analysis}

Com base nos DOIS relatórios combinados, forneça uma recomendação clínica final:

DIAGNOSTICO MAIS PROVAVEL:
[Qual é a hipótese diagnóstica principal considerando sintomas + ECG?]

NIVEL DE RISCO FINAL:
[Confirma ou altera o risco identificado na triagem por sintomas?]

CONDUTA IMEDIATA:
[O que o agente de saúde deve fazer AGORA, considerando os dois relatórios?]

ENCAMINHAMENTO:
[SAMU 192 / UPA / Hospital / UBS — com qual urgência?]

ATENCAO ESPECIAL:
[Algo que o ECG revelou que muda ou reforça a conduta inicial?]

Seja objetivo e use linguagem simples para agentes de saúde.
"""


def analyze_ecg(image_path: str) -> str:
    """
    Sends an ECG image to Gemini 2.5 Flash for multimodal analysis.
    Returns a structured clinical interpretation in Portuguese.

    This is the optional cloud component of BeatSafe — only runs when
    the health agent has internet access and uploads an ECG image.
    Core triage (main.py) always runs offline via Gemma 3.

    Args:
        image_path: File path to the ECG image (JPG or PNG)

    Returns:
        Structured ECG interpretation with rhythm, ST segment,
        QRS complex and clinical recommendation for health workers
    """

    # Open and convert the ECG image to bytes for the API
    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # Build the multimodal request — image + clinical prompt
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            ECG_PROMPT
        ]
    )

    return response.text


def combined_analysis(symptom_triage: str, ecg_analysis: str) -> str:
    """
    Combines symptom triage output (from Gemma 3 local) with ECG analysis
    (from Gemini Flash cloud) into a single unified clinical recommendation.

    BeatSafe's hybrid design:
      - Gemma 3 via Ollama: offline symptom triage (always available)
      - Gemini 2.5 Flash via API: ECG image analysis (when internet available)
      - This function: merges both into one final recommendation

    Args:
        symptom_triage: Output from the triage() function in main.py
        ecg_analysis:   Output from the analyze_ecg() function above

    Returns:
        Unified clinical recommendation combining both data sources
    """

    prompt = ECG_COMBINATION_PROMPT.format(
        symptom_triage=symptom_triage,
        ecg_analysis=ecg_analysis
    )

    # Text-only request to Gemini Flash — no image needed at this stage
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


def full_beatsafe_analysis(patient_info: str, image_path: str) -> dict:
    """
    Runs the complete BeatSafe pipeline for a patient with an ECG image:
        Step 1 — Symptom triage via Gemma 3 local (offline, always runs)
        Step 2 — ECG image analysis via Gemini 2.5 Flash (cloud, optional)
        Step 3 — Combined final recommendation via Gemini 2.5 Flash

    Args:
        patient_info: Patient symptoms, vitals and history (text)
        image_path:   Path to the ECG image file

    Returns:
        Dictionary with three keys:
            - symptom_triage: result from Gemma 3 (offline)
            - ecg_analysis:   result from Gemini Flash (multimodal)
            - final:          unified combined recommendation
    """

    # Import here to avoid circular imports between ecg.py and main.py
    from main import triage

    # Step 1 — Offline symptom triage using local Gemma 3 model
    print("Step 1/3 — Running symptom triage with Gemma 3 (local, offline)...")
    symptom_result = triage(patient_info)

    # Step 2 — Cloud ECG analysis using Gemini 2.5 Flash multimodal
    print("Step 2/3 — Analyzing ECG image with Gemini 2.5 Flash (cloud)...")
    ecg_result = analyze_ecg(image_path)

    # Step 3 — Combine both analyses into one unified recommendation
    print("Step 3/3 — Combining both analyses into final recommendation...")
    final_result = combined_analysis(symptom_result, ecg_result)

    return {
        "symptom_triage": symptom_result,
        "ecg_analysis": ecg_result,
        "final": final_result
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK — Tests ECG analysis with a local image
# Place any ECG image named "test_ecg.jpg" in the src/ folder to test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("BeatSafe — ECG Multimodal Analysis Module")
    print("=" * 50)
    print("Cloud component: Gemini 2.5 Flash via Google AI API")
    print("Offline component: Gemma 3:4b via Ollama (see main.py)")
    print("=" * 50)

    test_image_path = "test_ecg.jpg"

    if not os.path.exists(test_image_path):
        print(f"\nNo test image found.")
        print(f"Please place an ECG image named '{test_image_path}' in the src/ folder.")
    else:
        print(f"\nTest image found: {test_image_path}")
        print("\nAnalyzing ECG with Gemini 2.5 Flash...")
        print("-" * 50)

        result = analyze_ecg(test_image_path)
        print(result)
