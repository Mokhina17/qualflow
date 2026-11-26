# QualFlow

QualFlow is a Streamlit MVP for schema-guided qualitative coding of interview
transcripts. It proposes response segments and codes with Gemini, then lets a
researcher review and confirm the results before analysis or export.

## Status and limitations

- This is a research-assistance prototype, not an autonomous or validated
  qualitative-analysis method. Treat model output as suggestions and verify it
  against the source transcript and your research protocol.
- Gemini first receives the **full transcript** and schema to select relevant
  response units. The app then sends each selected response and its question
  context in separate coding requests.
- Response-unit selection can omit, combine, or misclassify material. Inspect
  the transcript and segment boundaries; do not assume that unselected text is
  irrelevant.
- The app asks for an interviewee name or label as it appears in the transcript.
  Use a pseudonym consistently in both places. Names and transcript content sent
  to the API are disclosed to Google.
- This public Gemini API setup has not been confirmed as approved by TUM. Do not
  upload identifiable or confidential research data unless your institution's
  ethics, data-protection, and contractual requirements permit it. For
  sensitive data, use an institution-approved environment.
- API availability, latency, quotas, and pricing depend on Google's current
  service terms. A free quota is not a privacy guarantee.

## Features

- Select an active JSON schema from `schemas/`.
- Upload a UTF-8 `.txt` transcript.
- Ask Gemini to select relevant response units from numbered source units. The
  app reconstructs the selected text from the original transcript rather than
  accepting model-generated quotes.
- Code each selected response against the active schema, with the preceding
  question available as context.
- Review and edit codes; confirm segments before including them in analysis.
- View confirmed coding in a table and download a CSV.
- Store interview JSON locally under `data/`.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Internet access for Gemini requests

## Setup (Windows PowerShell)

From the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your key. Keep `.env` private; it is
excluded from Git. The default model is configured by `GEMINI_MODEL` in
`.env.example`; change it only to a model available to your API key.

Start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Stop the server
with Ctrl+C in the terminal.

## Typical workflow

1. Select the schema in the sidebar.
2. Enter a unique interview ID and the interviewee's pseudonym/label as written
   in the transcript.
3. Upload the transcript and run coding.
4. Inspect the selected response segments, contexts, model reasoning, and codes.
   Correct them before confirming.
5. Use **Analysis & Export** to review confirmed segments and download a CSV.

Interview IDs may contain letters, numbers, `_`, and `-`; they must start with
a letter or number and be at most 64 characters. Existing IDs are not
overwritten.

## Schemas

Each `.json` file in `schemas/` must contain a `categories` array. Supported
category types:

- `text`: a string value
- `select`: one value from an `options` array
- `scale`: an integer between `min` and `max`
- `list`: a list of strings

Category `id` values must be unique within a schema. Researcher notes are stored
separately from model-generated codes and are not schema categories.

See [`schemas/example_general.json`](schemas/example_general.json) and
[`schemas/minzberg_ai.json`](schemas/minzberg_ai.json) for examples. Use clear
category names, precise definitions, and meaningful `Unclear` options where
appropriate. The current schema format does not yet support detailed
codebook definitions or examples.

## Local files and privacy

- `.env` contains the API key and must not be committed or shared.
- `data/` contains transcripts and coding results; it is ignored by Git.
- The `.venv/` virtual environment and Python cache files are generated locally
  and ignored by Git.
- The full transcript and selected segments leave the computer when sent to the
  Gemini API. Local JSON storage does not make API processing private.

Review Google's current API terms and your institution's requirements before
using real research data.