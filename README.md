# NEXUS AI — Data Analyst Team

A Streamlit dashboard powered by CrewAI and Google Gemini. Upload a CSV, inspect data quality, explore charts, and generate a multi-agent business analysis report.

## Features
- Dark purple RGB/neon dashboard
- CSV preview and dataset metrics
- Missing-value and duplicate checks
- Histogram, bar, scatter, box, and correlation charts
- Three sequential CrewAI agents: Data Analyst, Insight Analyst, Report Writer
- Downloadable Markdown report

## Setup (Windows / VS Code)

1. Install Python 3.10–3.13 and open this folder in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell activation is blocked, use Command Prompt:
   `\.venv\Scripts\activate.bat`

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your Google AI Studio API key. Keep `.env` private.
5. Run:

   ```bash
   python -m streamlit run app.py
   ```

6. Open http://localhost:8501 and upload a CSV.

## Gemini API
Create/manage your API key at https://aistudio.google.com/apikey. Model availability and quota depend on your Google account. If you receive a 429 quota error, check Google AI Studio usage and rate limits.

## Security and limitations
- Never commit `.env` or share your API key.
- The app sends a summarized dataset profile (column names, data types, missing counts, descriptive statistics and category counts) to Gemini when analysis is run. Do not upload sensitive data unless authorized.
- This starter app computes statistics locally; AI-generated interpretations still require human validation.
- CSV upload limit is 20 MB and 200,000 rows in this starter version.
