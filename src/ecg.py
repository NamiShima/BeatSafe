import os                        # Access environment variables (API key)
from google import genai         # Updated Google Gemini API library (google-genai)
from google.genai import types   # Content types for multimodal requests
from dotenv import load_dotenv   # Load .env file securely
from PIL import Image            # Image processing library
import io                        # Handle image bytes in memory

# ─────────────────────────────────────────────────────────────────────────────
# LOAD API KEY — Reads the Gemini API key from the .env file
# This keeps the key out of the source code (safe for GitHub)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini client with the API key
client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini model to use — Flash is fast and supports image input
GEMINI_MODEL = "gemini-1.5-flash-latest"

# ─────────────────────────────────────────────────────────────────────────────
# ECG ANALYSIS PROMPT — Instructs Gemini on how to read the ECG image
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

9. INTERPRETAÇÃO FINAL
   - ALTERADO — URGENTE: risco imediato à vida
   - ALTERADO — ATENÇÃO: requer avaliação médica
   - NORMAL: sem alterações significativas

10. RECOMENDAÇÃO PARA O AGENTE DE SAÚDE
    - O que fazer AGORA com base neste ECG?

Seja claro, direto e use linguagem acessível para agentes de saúde não especialistas.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ECG COMBINATION PROMPT — Merges ECG findings with symptom triage
# This is the core innovation: two AI models working together
# ─────────────────────────────────────────────────────────────────────────────
ECG_COMBINATION_PROMPT = """
Você é um cardiologista de plantão em uma Unidade Básica de Saúde brasileira.

Você recebeu dois relatórios sobre o mesmo paciente:

RELATÓRIO 1 — TRIAGEM POR SINTOMAS:
{symptom_triage}

RELATÓRIO 2 — ANÁLISE DO ECG:
{ecg_analysis}

Com base nos DOIS relatórios combinados, forneça uma recomendação clínica final:

DIAGNÓSTICO MAIS PROVÁVEL:
[Qual é a hipótese diagnóstica principal considerando sintomas + ECG?]

NÍVEL DE RISCO FINAL:
[Confirma ou altera o risco identificado na triagem por sintomas?]

CONDUTA IMEDIATA:
[O que o agente de saúde deve fazer AGORA, considerando os dois relatórios?]

ENCAMINHAMENTO:
[SAMU 192 / UPA / Hospital / UBS — com qual urgência?]

ATENÇÃO ESPECIAL:
[Algo que o ECG revelou que muda ou reforça a conduta inicial?]

Seja objetivo e use linguagem simples para agentes de saúde.
"""


def analyze_ecg(image_path: str) -> str:
    """
    Sends an ECG image to Gemini for multimodal analysis.
    Returns a structured clinical interpretation in Portuguese.

    Requires internet connection and a valid GEMINI_API_KEY in .env

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

    # Build the multimodal request with image + clinical prompt
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            ECG_PROMPT  # Clinical instructions for ECG interpretation
        ]
    )

    return response.text


def combined_analysis(symptom_triage: str, ecg_analysis: str) -> str:
    """
    Combines symptom triage output (from Gemma local) with ECG analysis
    (from Gemini cloud) into a single unified clinical recommendation.

    This is the key innovation of BeatSafe:
    Gemma handles text reasoning offline,
    Gemini handles image analysis in the cloud,
    together they produce a more accurate triage.

    Args:
        symptom_triage: Output from the triage() function in main.py
        ecg_analysis:   Output from the analyze_ecg() function above

    Returns:
        Unified clinical recommendation combining both data sources
    """

    # Fill in the combination prompt with both reports
    prompt = ECG_COMBINATION_PROMPT.format(
        symptom_triage=symptom_triage,
        ecg_analysis=ecg_analysis
    )

    # Send to Gemini for final unified analysis (text only, no image)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


def full_beatsafe_analysis(patient_info: str, image_path: str) -> dict:
    """
    Runs the complete BeatSafe pipeline for a patient with an ECG image:
        Step 1 — Symptom triage via Gemma 3 local (offline)
        Step 2 — ECG image analysis via Gemini API (cloud)
        Step 3 — Combined final recommendation

    Args:
        patient_info: Patient symptoms, vitals and history (text)
        image_path:   Path to the ECG image file

    Returns:
        Dictionary with three keys:
            - symptom_triage: result from Gemma local
            - ecg_analysis:   result from Gemini multimodal
            - final:          unified combined recommendation
    """

    # Import here to avoid circular imports between ecg.py and main.py
    from main import triage

    # Step 1 — Offline symptom triage using local Gemma model
    print("Step 1/3 — Running symptom triage with Gemma (local, offline)...")
    symptom_result = triage(patient_info)

    # Step 2 — Cloud ECG analysis using Gemini multimodal model
    print("Step 2/3 — Analyzing ECG image with Gemini (cloud)...")
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
# TEST BLOCK — Tests only the ECG analysis with a local image
# Place any ECG image named "test_ecg.jpg" in the src/ folder to test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("BeatSafe — ECG Multimodal Analysis Module")
    print("=" * 50)
    print("Powered by Gemini 2.0 Flash (Google AI)")
    print("=" * 50)

    # Look for a local test ECG image in the current folder
    test_image_path = "test_ecg.jpg"

    if not os.path.exists(test_image_path):
        # No test image found — guide the user
        print(f"\nNo test image found.")
        print(f"Please place an ECG image named '{test_image_path}' in the src/ folder.")
        print("You can use any ECG image in JPG or PNG format.")
        print("\nTip: Search Google Images for 'ECG normal sinus rhythm'")
        print("     and save it as 'test_ecg.jpg' in the src/ folder.")
    else:
        # Test image found — run ECG analysis only
        print(f"\nTest image found: {test_image_path}")
        print("\nAnalyzing ECG with Gemini...")
        print("-" * 50)

        result = analyze_ecg(test_image_path)
        print(result)
