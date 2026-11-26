import streamlit as st
import json
import os
import random
import re
import time
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from openai import APIStatusError, OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path("data")
SCHEMAS_DIR = Path("schemas")
DATA_DIR.mkdir(exist_ok=True)
SCHEMAS_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env before starting QualFlow.")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60,
    max_retries=0
)
MODEL = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")

st.set_page_config(
    page_title="QualFlow",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Helpers
# ============================================================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def list_schemas():
    return sorted(SCHEMAS_DIR.glob("*.json"))

def list_interviews():
    return sorted([p.stem for p in DATA_DIR.glob("*.json")])

def normalize_transcript(transcript: str) -> str:
    return transcript.replace("\r\n", "\n").replace("\r", "\n").strip()

def build_source_units(transcript: str, max_chars: int = 600) -> list[dict]:
    units = []
    for line in re.finditer(r"[^\n]+", transcript):
        text = line.group()
        start = 0
        while start < len(text):
            while start < len(text) and text[start].isspace():
                start += 1
            if start >= len(text):
                break

            end = min(start + max_chars, len(text))
            if end < len(text):
                split_at = text.rfind(" ", start, end)
                if split_at > start:
                    end = split_at

            units.append({
                "id": len(units) + 1,
                "start": line.start() + start,
                "end": line.start() + end,
                "text": text[start:end]
            })
            start = end

    return units

def is_valid_category_value(value, category: dict) -> bool:
    if category["type"] == "select":
        return value in category["options"]
    if category["type"] == "scale":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and category["min"] <= value <= category["max"]
        )
    if category["type"] == "list":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    return isinstance(value, str)

def validate_coding_result(result: dict, schema: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")

    if not isinstance(result.get("has_interviewee_response"), bool):
        raise ValueError("The response must indicate whether this segment contains an interviewee response.")

    coding = result.get("coding")
    if not isinstance(coding, dict):
        raise ValueError("The response is missing a coding object.")

    for category in schema["categories"]:
        category_id = category["id"]
        if category_id not in coding:
            raise ValueError(f"The response is missing '{category_id}'.")
        if not is_valid_category_value(coding[category_id], category):
            raise ValueError(f"The response contains an invalid '{category_id}' value.")

    if not isinstance(result.get("reasoning"), str):
        raise ValueError("The response is missing text reasoning.")
    confidence = result.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Confidence must be a number between 0 and 1.")

    return result

def describe_schema(schema: dict) -> str:
    descriptions = []
    for category in schema["categories"]:
        if category["type"] == "select":
            descriptions.append(
                f'- {category["id"]} ({category["label"]}): one of {category["options"]}'
            )
        elif category["type"] == "scale":
            descriptions.append(
                f'- {category["id"]} ({category["label"]}): integer {category["min"]}-{category["max"]}'
            )
        elif category["type"] == "list":
            descriptions.append(
                f'- {category["id"]} ({category["label"]}): list of short strings'
            )
        else:
            descriptions.append(f'- {category["id"]} ({category["label"]}): string')
    return "\n".join(descriptions)

def validate_response_segments(result: dict, transcript: str, source_units: list[dict]) -> list[dict]:
    if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
        raise ValueError("The segmentation response must contain a segments list.")

    def validate_unit_ids(value, segment_index: int, field: str, required: bool) -> list[int]:
        if not isinstance(value, list) or (required and not value):
            raise ValueError(f"Response segment {segment_index} must contain {field}.")
        if any(not isinstance(unit_id, int) or isinstance(unit_id, bool) for unit_id in value):
            raise ValueError(f"{field} for response segment {segment_index} must be integer IDs.")
        if value != sorted(set(value)):
            raise ValueError(f"{field} for response segment {segment_index} must be unique and ordered.")
        if any(unit_id < 1 or unit_id > len(source_units) for unit_id in value):
            raise ValueError(f"{field} for response segment {segment_index} contains an unknown ID.")
        return value

    def assemble_units(unit_ids: list[int]) -> str:
        parts = []
        previous_id = None
        previous_unit = None
        for unit_id in unit_ids:
            unit = source_units[unit_id - 1]
            if previous_id is not None:
                separator = (
                    transcript[previous_unit["end"]:unit["start"]]
                    if unit_id == previous_id + 1
                    else "\n"
                )
                parts.append(separator)
            parts.append(unit["text"])
            previous_id = unit_id
            previous_unit = unit
        return "".join(parts).strip()

    segments = []
    last_response_id = 0
    used_response_ids = set()
    for index, item in enumerate(result["segments"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Response segment {index} must be an object.")
        response_ids = validate_unit_ids(
            item.get("response_unit_ids"),
            index,
            "response_unit_ids",
            required=True
        )
        context_ids = validate_unit_ids(
            item.get("context_unit_ids", []),
            index,
            "context_unit_ids",
            required=False
        )
        if response_ids[0] <= last_response_id:
            raise ValueError("Response segments must be in transcript order and cannot overlap.")
        if context_ids and context_ids[-1] >= response_ids[0]:
            raise ValueError(f"Context for response segment {index} must precede its response.")
        if used_response_ids.intersection(context_ids):
            raise ValueError(f"Context for response segment {index} overlaps another response.")

        response_text = assemble_units(response_ids)
        context = assemble_units(context_ids)
        segments.append({
            "text": response_text,
            "context": context,
            "source_unit_ids": response_ids
        })
        used_response_ids.update(response_ids)
        last_response_id = response_ids[-1]

    return segments

def create_completion(**request):
    retryable_statuses = {408, 429, 500, 502, 503, 504}
    for attempt in range(3):
        try:
            return client.chat.completions.create(**request)
        except APIStatusError as e:
            if e.status_code not in retryable_statuses or attempt == 2:
                raise

            delay = 2 ** attempt + random.uniform(0, 0.5)
            retry_after = e.response.headers.get("retry-after")
            if retry_after:
                try:
                    delay = max(delay, min(float(retry_after), 30))
                except ValueError:
                    pass
            time.sleep(delay)

def extract_relevant_segments(transcript: str, schema: dict, interviewee_label: str) -> list[dict]:
    source_units = build_source_units(transcript)
    if not source_units:
        return []

    system = f"""You are a careful qualitative research assistant.
Find the interviewee's response units that contain evidence relevant to at least one category in the schema.
The interviewee is identified by this name or label as it appears in the transcript: {json.dumps(interviewee_label, ensure_ascii=False)}.
The transcript is split into numbered source units. Select only units clearly attributable to the interviewee and relevant to the schema.
Exclude interviewer speech and unrelated content. Make one segment per coherent response by grouping source units; units may be non-adjacent when interviewer or irrelevant text lies between them.
Do not output, quote, or paraphrase any transcript text. Use only the source unit IDs.
For each segment, return "response_unit_ids" as a unique ordered list of IDs, excluding interviewer and irrelevant units. Include preceding interviewer question IDs as "context_unit_ids" when useful, otherwise use an empty list.
If no response is clearly attributable to the interviewee or relevant to the schema, return an empty segments list.

Schema:
{describe_schema(schema)}

Return only a JSON object with a "segments" array. Each item must have "response_unit_ids" and "context_unit_ids" arrays of integer IDs.
"""
    source_text = [
        {"id": unit["id"], "text": unit["text"]}
        for unit in source_units
    ]
    user = f"Numbered transcript source units (JSON-encoded):\n{json.dumps(source_text, ensure_ascii=False)}"
    response = create_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )
    if not response.choices or not isinstance(response.choices[0].message.content, str):
        raise ValueError("The model returned no response segments.")
    result = json.loads(response.choices[0].message.content)
    return validate_response_segments(result, transcript, source_units)

def call_llm(segment: str, schema: dict, interviewee_label: str, context: str = "") -> dict:
    """Transparent structured coding with explanation."""
    category_ids = ", ".join(c["id"] for c in schema["categories"])
    system = f"""You are a rigorous qualitative research assistant.
Code only the target response using the schema below. Use evidence from that response; do not invent facts.
The interviewee is identified by this name or label as it appears in the transcript: {json.dumps(interviewee_label, ensure_ascii=False)}.
Code only statements clearly attributable to this interviewee. Treat the interviewer context as context only.
Do not infer the interviewee's opinion from an interviewer question. If the target response contains no clear interviewee statement, set
"has_interviewee_response" to false, use "Unclear" where available, leave text/list values empty, use the minimum for a required scale,
set confidence to 0, and explain that no interviewee response is present.
Use an allowed "Unclear" option when the evidence does not support a definite selection.
For key_quote, return an exact quote from the segment or an empty string.
Give a brief evidence-based rationale, not hidden chain-of-thought.

Schema:
{describe_schema(schema)}

The coding object must contain exactly these category ids: {category_ids}.
Return only a JSON object with top-level keys "has_interviewee_response" (boolean), "coding" (object),
"reasoning" (string), and "confidence" (number between 0 and 1).
"""

    user = f"""Interviewer context (context only; do not code it):
{context or "[No interviewer context]"}

Target response to code:
\"\"\"{segment}\"\"\"
"""

    resp = create_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.15,
        max_tokens=512,
        response_format={"type": "json_object"}
    )
    if not resp.choices or not isinstance(resp.choices[0].message.content, str):
        raise ValueError("The model returned no JSON content.")
    result = json.loads(resp.choices[0].message.content)
    return validate_coding_result(result, schema)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("QualFlow")
st.sidebar.caption("Schema-first qualitative analysis")

schema_files = list_schemas()
if not schema_files:
    st.sidebar.error("No schemas found in /schemas")
    st.stop()

schema_names = [f.name for f in schema_files]
selected_schema_file = st.sidebar.selectbox("Active Schema", schema_names)
schema = load_json(SCHEMAS_DIR / selected_schema_file)

st.sidebar.markdown(f"**{schema.get('name', 'Unnamed')}**")
st.sidebar.caption(schema.get("description", ""))

page = st.sidebar.radio(
    "Navigation",
    ["Upload & Code", "Review", "Analysis & Export", "Schema Info"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("MVP • Transparent AI + Human review")

# ============================================================
# PAGE 1: Upload & Code
# ============================================================
if page == "Upload & Code":
    st.header("Upload & Code Interview")

    interview_id = st.text_input(
        "Interview ID",
        value=f"int_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )

    interviewee_label = st.text_input(
        "Interviewee name/label as it appears in the transcript",
        help="Use an anonymized label (for example, Participant A). This label and the transcript are sent to Gemini."
    )

    uploaded = st.file_uploader("Upload transcript (.txt)", type=["txt"])

    if uploaded and st.button("Run Coding", type="primary", use_container_width=True):
        interview_id = interview_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", interview_id):
            st.error("Interview ID must start with a letter or number and contain only letters, numbers, underscores, or hyphens (up to 64 characters).")
            st.stop()
        if (DATA_DIR / f"{interview_id}.json").exists():
            st.error(f"An interview with ID '{interview_id}' already exists. Choose a different ID to avoid overwriting it.")
            st.stop()
        if not interviewee_label.strip():
            st.error("Enter the interviewee's name or label as it appears in the transcript.")
            st.stop()

        transcript = normalize_transcript(uploaded.read().decode("utf-8-sig"))
        if not transcript:
            st.error("The uploaded transcript is empty.")
            st.stop()

        try:
            with st.spinner("Finding schema-relevant interviewee responses..."):
                response_segments = extract_relevant_segments(
                    transcript,
                    schema,
                    interviewee_label.strip()
                )
        except OpenAIError as e:
            st.error(f"Model request failed while finding relevant responses: {e}")
            st.stop()
        except ValueError as e:
            st.error(f"Could not validate response segments: {e}")
            st.stop()

        if not response_segments:
            st.warning("No schema-relevant interviewee responses were identified. Check the interviewee label and schema.")
            st.stop()

        st.info(f"Found {len(response_segments)} relevant response segment(s). Coding...")
        progress = st.progress(0)
        coded_segments = []
        try:
            for i, response_segment in enumerate(response_segments):
                result = call_llm(
                    response_segment["text"],
                    schema,
                    interviewee_label.strip(),
                    response_segment["context"]
                )
                coded_segments.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": response_segment["text"],
                    "context": response_segment["context"],
                    "source_unit_ids": response_segment["source_unit_ids"],
                    "has_interviewee_response": result["has_interviewee_response"],
                    "coding": result["coding"],
                    "reasoning": result["reasoning"],
                    "confidence": result["confidence"],
                    "confirmed": False,
                    "memo": ""
                })
                progress.progress((i + 1) / len(response_segments))
        except OpenAIError as e:
            st.error(f"Model request failed for segment {i + 1}: {e}")
            st.stop()
        except ValueError as e:
            st.error(f"Invalid model response for segment {i + 1}: {e}")
            st.stop()

        interview_data = {
            "id": interview_id,
            "schema": selected_schema_file,
            "created_at": datetime.now().isoformat(),
            "transcript": transcript,
            "segments": coded_segments
        }
        save_json(DATA_DIR / f"{interview_id}.json", interview_data)
        st.success(f"Saved as **{interview_id}**. Go to **Review**.")

# ============================================================
# PAGE 2: Review (Human-in-the-loop)
# ============================================================
elif page == "Review":
    st.header("Review & Refine")

    interviews = list_interviews()
    if not interviews:
        st.info("No interviews yet.")
        st.stop()

    selected = st.selectbox("Interview", interviews)
    data = load_json(DATA_DIR / f"{selected}.json")
    review_schema = load_json(SCHEMAS_DIR / data["schema"])

    st.caption(f"Schema: `{data['schema']}` • {len(data['segments'])} segments")

    for idx, seg in enumerate(data["segments"]):
        status = "✅" if seg["confirmed"] else "🟡"
        confidence = seg.get("confidence")
        confidence_label = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
        with st.expander(f"{status} Segment {idx+1}  •  confidence {confidence_label}", expanded=not seg["confirmed"]):
            
            if seg.get("context"):
                st.markdown("**Previous paragraph (context only)**")
                st.text(seg["context"][:1000] + ("..." if len(seg["context"]) > 1000 else ""))

            st.markdown("**Transcript excerpt**")
            st.text(seg["text"][:1000] + ("..." if len(seg["text"]) > 1000 else ""))

            st.markdown("**Model reasoning**")
            st.info(seg.get("reasoning") or "—")

            has_interviewee_response = st.checkbox(
                "This segment contains an interviewee response",
                value=seg.get("has_interviewee_response", True),
                key=f"response_{seg['id']}"
            )
            if not has_interviewee_response:
                st.warning("The model did not identify a clear interviewee response. Check this before confirming codes.")

            coding = seg.get("coding") if isinstance(seg.get("coding"), dict) else {}
            invalid_categories = [
                category["id"]
                for category in review_schema["categories"]
                if not is_valid_category_value(coding.get(category["id"]), category)
            ]
            if invalid_categories:
                st.warning(
                    "Missing or invalid saved codes for: "
                    + ", ".join(invalid_categories)
                    + ". Review these values before confirming."
                )

            new_coding = {}
            cols = st.columns(2)

            for i, cat in enumerate(review_schema["categories"]):
                cid = cat["id"]
                current = coding.get(cid)

                with cols[i % 2]:
                    if cat["type"] == "select":
                        opts = cat["options"]
                        ix = opts.index(current) if current in opts else None
                        new_coding[cid] = st.selectbox(
                            cat["label"],
                            opts,
                            index=ix,
                            placeholder="Select a value",
                            key=f"{seg['id']}_{cid}"
                        )
                    elif cat["type"] == "scale":
                        val = current if is_valid_category_value(current, cat) else cat["min"]
                        new_coding[cid] = st.slider(cat["label"], cat["min"], cat["max"], value=val, key=f"{seg['id']}_{cid}")
                    elif cat["type"] == "list":
                        if isinstance(current, list):
                            txt = ", ".join(current)
                        else:
                            txt = str(current) if current is not None else ""
                        raw = st.text_input(cat["label"] + " (comma-separated)", value=txt, key=f"{seg['id']}_{cid}")
                        new_coding[cid] = [x.strip() for x in raw.split(",") if x.strip()]
                    else:
                        value = current if isinstance(current, str) else ""
                        new_coding[cid] = st.text_area(cat["label"], value=value, key=f"{seg['id']}_{cid}")

            new_memo = st.text_area("Researcher memo", value=seg.get("memo", ""), key=f"memo_{seg['id']}")

            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("Confirm", key=f"conf_{seg['id']}", type="primary"):
                    if not has_interviewee_response:
                        st.error("Confirm only after verifying that this segment contains an interviewee response.")
                    else:
                        missing_codes = [
                            category["id"]
                            for category in review_schema["categories"]
                            if not is_valid_category_value(new_coding.get(category["id"]), category)
                        ]
                        if missing_codes:
                            st.error("Choose valid values for: " + ", ".join(missing_codes))
                        else:
                            data["segments"][idx]["has_interviewee_response"] = True
                            data["segments"][idx]["coding"] = new_coding
                            data["segments"][idx]["memo"] = new_memo
                            data["segments"][idx]["confirmed"] = True
                            save_json(DATA_DIR / f"{selected}.json", data)
                            st.rerun()
            with c2:
                if st.button("Save without confirming", key=f"save_{seg['id']}"):
                    data["segments"][idx]["has_interviewee_response"] = has_interviewee_response
                    data["segments"][idx]["coding"] = new_coding
                    data["segments"][idx]["memo"] = new_memo
                    save_json(DATA_DIR / f"{selected}.json", data)
                    st.success("Saved")

# ============================================================
# PAGE 3: Analysis & Export
# ============================================================
elif page == "Analysis & Export":
    st.header("Analysis & Export")

    interviews = list_interviews()
    if not interviews:
        st.info("No interviews yet.")
        st.stop()

    chosen = st.multiselect("Select interviews", interviews, default=interviews)

    rows = []
    for iid in chosen:
        data = load_json(DATA_DIR / f"{iid}.json")
        for seg in data["segments"]:
            if not seg.get("confirmed"):
                continue
            row = {
                "interview": iid,
                "segment_id": seg["id"],
                "confidence": seg.get("confidence", None),
                "preview": seg["text"][:100] + "..."
            }
            row.update(seg["coding"])
            row["memo"] = seg.get("memo", "")
            rows.append(row)

    if not rows:
        st.warning("No confirmed segments. Review interviews first.")
        st.stop()

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)

    st.subheader("Distributions")
    cat_cols = [c["id"] for c in schema["categories"] if c["type"] in ("select", "scale")]
    if cat_cols:
        cols = st.columns(min(3, len(cat_cols)))
        for i, col in enumerate(cat_cols):
            if col in df.columns:
                with cols[i % 3]:
                    st.write(f"**{col}**")
                    st.write(df[col].value_counts())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"qualflow_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        type="primary"
    )

# ============================================================
# PAGE 4: Schema Info
# ============================================================
elif page == "Schema Info":
    st.header("Current Schema")
    st.json(schema)
    st.markdown("To create a new schema, add a `.json` file in the `/schemas` folder following the same structure.")