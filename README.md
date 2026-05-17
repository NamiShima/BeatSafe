# BeatSafe — Offline Cardiac Triage AI for Brazilian Primary Care

> **Gemma 4 Good Hackathon** · Health & Sciences Track · Ollama Special Technology Track

**BeatSafe** is an offline-first cardiac triage assistant designed for community health agents working in Brazilian public health units (UBS — *Unidades Básicas de Saúde*), especially in underserved regions with limited or no internet access. It runs entirely on local hardware using **Gemma 3 (via Ollama)** for symptom triage and **Gemini Flash** for optional ECG image analysis.

---

## The Problem

Brazil's public health system (SUS) serves over 215 million people. The vast majority of first contact with the healthcare system happens at UBS units — often staffed by community health agents who are **not doctors**, working in regions far from hospitals and specialists.

Every year, thousands of patients arrive at these units with chest pain, hypertension crises, or signs of heart failure. The agents have minimal clinical tools, no decision support, and no access to specialists. They must decide: *is this a SAMU 192 emergency or can this wait?*

A wrong decision costs lives. Getting it right consistently requires clinical knowledge most agents simply don't have.

---

## The Solution

BeatSafe gives community health agents a clinical decision support system that:

- **Runs 100% offline** — no internet required for core triage (Gemma 3 via Ollama)
- **Speaks the language of the user** — fully in Portuguese, designed for non-medical professionals
- **Is grounded in official Brazilian guidelines** — not general AI knowledge
- **Generates a PDF report** shareable via WhatsApp with the supervising doctor
- **Logs all triage sessions** locally in SQLite — no data leaves the device

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  BeatSafe (Gradio UI)                │
├──────────────┬──────────────┬────────────────────────┤
│  Triage Tab  │  Chatbot Tab │  Medication Tab        │
│  (offline)   │  (offline)   │  (offline)             │
├──────────────┴──────────────┴────────────────────────┤
│           Gemma 3:4b via Ollama (local)              │
│           runs on CPU/GPU — no internet              │
├─────────────────────────────────────────────────────┤
│  ECG Analysis (optional, requires internet)         │
│           Gemini 2.5 Flash (multimodal)             │
├─────────────────────────────────────────────────────┤
│  PDF Report  │  SQLite History  │  Maps (links)     │
└─────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI — 5 tabs, all in Portuguese |
| `main.py` | Triage engine — system prompt + Ollama streaming |
| `ecg.py` | ECG image analysis via Gemini Flash |
| `pdf_report.py` | One-page clinical PDF with risk banner |
| `history.py` | SQLite triage log with stats dashboard |

---

## How Gemma 3 is Used

Gemma 3:4b runs locally via Ollama and powers three features:

**1. Cardiac Triage (streaming)**
The core feature. The agent fills in patient data (symptoms, vitals, history, medications) and Gemma generates a structured clinical assessment token-by-token, following a strict protocol format:
- Risk classification (High / Moderate / Low)
- Clinical reasoning in plain Portuguese
- Recommended actions
- Referral guidance (SAMU 192, UPA, hospital, UBS)
- Patient instructions

**2. Clinical Chatbot**
An offline assistant that answers questions about SUS protocols, signs of cardiac emergency, medications, and when to call SAMU 192 — all in language accessible to non-medical agents.

**3. Medication Suggestions**
Based on the clinical scenario, Gemma suggests medications from the official Brazilian essential medicines list (RENAME), with mandatory disclaimer that prescription is a medical act.

---

## Clinical Knowledge Base

BeatSafe's system prompt is grounded in **6 official Brazilian clinical guidelines** — not generic AI training data:

| Guideline | Year | Key Contribution |
|---|---|---|
| Caderno de Atenção Básica nº 14 (MS) | 2006 | Primary care cardiac protocols |
| Protocolo SAMU 192 DF | — | Emergency referral criteria |
| Protocolos SBV/SAV | — | CPR and life support |
| Diretriz HAS — SBC/SBH/SBN | 2025 | Hypertension staging, PA targets, drug classes |
| Diretriz IC — SBC | 2018 | Heart failure profiles A/B/C/L, LVEF classification |
| Diretriz FA — SOBRAC/SBC | 2025 | AF classification, CHA2DS2-VA score |
| Diretriz Dor Torácica na UE — SBC/FLAME | 2025 | HEART score, MOVE protocol, instability criteria |

This is what separates BeatSafe from a generic LLM wrapper: the model is instructed to reason within specific Brazilian clinical protocols, not global medical knowledge.

---

## Features

### ⚡ Nova Triagem (New Triage)
- Patient identity form (name, age, sex, vitals)
- Associated symptoms, medical history, current medications
- Optional ECG image upload (analyzed by Gemini Flash)
- Streaming response — text appears word by word
- Automatic PDF generation after each triage
- Demo cases pre-loaded in Portuguese

### 📋 Histórico (History)
- All triage sessions stored locally in SQLite
- Stats dashboard: total, high risk, moderate, low risk
- Filter by risk level
- No data ever sent to external servers

### 💬 Assistente (Chatbot)
- Offline clinical Q&A powered by Gemma 3
- Focused on SUS protocols, SAMU criteria, cardiac symptoms

### 🏥 Unidades Próximas (Nearby Units)
- One-click Google Maps links for hospitals, UBS, UPA, pharmacies, SAMU
- Emergency alert: "Call SAMU 192 — don't wait for the map"

### 💊 Medicamentos (Medications)
- Protocol-based medication suggestions (RENAME/CAB-14)
- Mandatory disclaimer displayed before any suggestion
- Streaming response via Gemma 3

---

## Setup

### Requirements
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Gemma 3:4b pulled: `ollama pull gemma3:4b`
- Google API key (for ECG analysis only)

### Installation

```bash
git clone https://github.com/NamiShima/BeatSafe
cd BeatSafe/src
pip install -r requirements.txt
ollama serve   # if not already running
python app.py
```

### Environment Variables
```bash
GOOGLE_API_KEY=your_key_here   # optional — only for ECG analysis
```

---

## Design Decisions

**Why Gemma 3:4b instead of a larger model?**
The target hardware is a standard laptop at a UBS unit. Gemma 3:4b runs on 2.5GB RAM, responds in seconds, and produces clinically coherent output when properly prompted. We tested Gemma 3:12b but it caused 99% RAM usage on typical field hardware — a dealbreaker.

**Why Gradio instead of a native app?**
Gradio 5.x runs in any browser, requires no installation by the end user, and supports streaming natively. The agent just opens a URL. We downgraded from 6.x to 5.29.0 after discovering critical rendering issues with tabs and CSS in the newer version.

**Why SQLite instead of a cloud database?**
Privacy and offline resilience. Patient data never leaves the device. The agent doesn't need internet to review history or generate statistics.

**Why a hybrid model (Gemma + Gemini)?**
ECG analysis requires multimodal vision capability. Gemma 3:4b does not support image input. Gemini Flash handles this cloud-side when internet is available, while core triage stays fully offline. The system degrades gracefully — if there's no internet, ECG analysis is simply skipped.

---

## Impact

BeatSafe directly addresses the **Health & Sciences** and **Ollama Special Technology** tracks:

- **Health & Sciences**: democratizes clinical decision support for the 300,000+ community health agents in Brazil who currently operate without AI tools
- **Ollama**: demonstrates Gemma 3 running locally on consumer hardware, with a real production use case where offline capability is not optional — it's a medical necessity

The PDF report feature enables continuity of care: the agent can share the triage result with a supervising doctor via WhatsApp, creating a documented clinical handoff even in areas with no EHR systems.

---

## Repository

**GitHub:** https://github.com/NamiShima/BeatSafe

---

## Disclaimer

BeatSafe is a clinical decision **support** tool. It does not replace medical evaluation. All triage results are recommendations based on official Brazilian protocols and must be validated by qualified healthcare professionals. In emergencies, always call **SAMU 192** first.
