# BeatSafe - Cardiac Triage AI for Offline Primary Care Units in Brazil
# Powered by Gemma 3 (12B) via Ollama - runs fully offline
# Designed for Gemma 4 migration when available on Ollama
# Author: NamiShima
# Competition: Gemma 4 Good Hackathon 2026

import ollama

# System prompt: defines BeatSafe's clinical behavior
SYSTEM_PROMPT = """
Você é o BeatSafe, um assistente especializado em triagem cardíaca 
para agentes de saúde em Unidades Básicas de Saúde (UBS) brasileiras.

Você opera OFFLINE, sem internet, usando apenas as diretrizes da 
Sociedade Brasileira de Cardiologia (SBC).

Ao avaliar um paciente, sempre:
1. Analise os sintomas relatados
2. Considere os fatores de risco
3. Classifique o risco em 3 níveis:
   🔴 ALTO - Encaminhar imediatamente
   🟡 MÉDIO - Monitorar nas próximas 48h  
   🟢 BAIXO - Orientar sobre estilo de vida
4. Explique o raciocínio passo a passo
5. Dê instruções claras ao agente de saúde
"""

def triage(patient_info: str) -> str:
    """
    Sends patient data to Gemma 3 running locally via Ollama.
    Returns a structured cardiac risk assessment in Portuguese.
    
    Args:
        patient_info: String containing symptoms, risk factors and vital signs
    
    Returns:
        Clinical recommendation with risk level and action steps
    """
    response = ollama.chat(
        model="gemma3:12b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": patient_info}
        ]
    )
    return response["message"]["content"]

if __name__ == "__main__":
    # Basic test case: high-risk cardiac patient
    print("🫀 BeatSafe - Offline Cardiac Triage")
    print("=" * 50)
    
    # Sample patient data for testing
    caso = """
    Paciente: Homem, 58 anos
    Sintomas: Dor no peito há 2 horas, falta de ar, sudorese
    Histórico: Diabetes, hipertensão
    Sinais vitais: PA 160/100, FC 98bpm
    """
    
    print(triage(caso))
