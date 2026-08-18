import io
import json
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Matcher - Groq",
    layout="wide",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MAX_RESUMES = 5

# Preferred models. The application will only offer models
# that are actually visible to the current Groq API key.
PREFERRED_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

client = None

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state["results"] = []

if "reset_id" not in st.session_state:
    st.session_state["reset_id"] = 0

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = ""


def clear_all():
    st.session_state.clear()
    st.rerun()


# ============================================================
# MODEL DISCOVERY
# ============================================================

def get_available_models():
    """
    Return models currently visible to the Groq API key.
    """
    if client is None:
        return []

    try:
        response = client.models.list()

        model_ids = []

        for model in response.data:
            model_id = getattr(model, "id", None)

            if model_id:
                model_ids.append(model_id)

        return sorted(set(model_ids))

    except Exception:
        return []


def choose_default_model(available_models):
    """
    Prefer known good models when available.
    Otherwise use the first visible model.
    """
    for preferred in PREFERRED_MODELS:
        if preferred in available_models:
            return preferred

    if available_models:
        return available_models[0]

    return ""


# ============================================================
# HEADER
# ============================================================

st.title("AI Resume Matcher and ATS-style Scorer")

st.write(
    """
Screen up to 5 resumes against a Job Description using Groq-hosted AI.

Features:
- AI score from 1 to 10
- Bad / Good / Best classification
- Matching and missing skills
- Strong points
- Automatic JD keyword extraction
- ATS-style name/email/phone parsing
- Per-skill weighting
- Candidate ranking
- Dashboard
- Candidate similarity heatmap
- CSV export
- Excel export
- Email-ready report
"""
)

if client:
    st.success("Groq API key detected.")
else:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Set it before running AI analysis."
    )

st.markdown("---")


# ============================================================
# CONTROLS / MODEL
# ============================================================

st.subheader("Controls")

control_col1, control_col2 = st.columns([1, 2])

with control_col1:
    st.button(
        "Clear All Resumes and Inputs",
        on_click=clear_all,
        use_container_width=True,
    )

with control_col2:

    available_models = get_available_models()

    if available_models:

        default_model = choose_default_model(
            available_models
        )

        if (
            st.session_state["selected_model"]
            not in available_models
        ):
            st.session_state["selected_model"] = default_model

        model_name = st.selectbox(
            "Select Groq Model",
            options=available_models,
            index=available_models.index(
                st.session_state["selected_model"]
            ),
            key="model_selector",
            help=(
                "Only models visible to your Groq API key "
                "are shown here."
            ),
        )

        st.session_state["selected_model"] = model_name

        st.success(
            f"Your API key can access "
            f"{len(available_models)} model(s)."
        )

    else:

        st.warning(
            "Could not retrieve the Groq model list."
        )

        model_name = st.text_input(
            "Groq Model",
            value=(
                st.session_state["selected_model"]
                or "openai/gpt-oss-20b"
            ),
            key="manual_model",
        )

        st.session_state["selected_model"] = model_name

if model_name:
    st.info(
        f"AI model selected: `{model_name}`"
    )

st.markdown("---")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

STOPWORDS = set(
    """
a an the and or of to in for with on at from by is are as be this
that these those it its if then than into over under was were has
have had been being will would should could can may might do does did
about above below between through during while where which who whom
what when how very more most other some any each all both few many much
such also not no nor but
""".split()
)


def tokenize(text):
    text = text.lower()

    tokens = re.split(
        r"[^a-zA-Z0-9+#./-]+",
        text
    )

    return [
        token.strip()
        for token in tokens
        if token.strip()
        and token.strip() not in STOPWORDS
        and len(token.strip()) > 2
    ]


def extract_jd_skills(text, top_n=20):
    tokens = tokenize(text)

    frequency = {}

    for token in tokens:
        frequency[token] = frequency.get(token, 0) + 1

    ranked = sorted(
        frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item[0]
        for item in ranked[:top_n]
    ]


def parse_skill_weights(raw_text):
    weights = {}

    for line in raw_text.splitlines():

        line = line.strip()

        if not line:
            continue

        if "," in line:
            skill, weight = line.split(",", 1)

        elif ":" in line:
            skill, weight = line.split(":", 1)

        else:
            skill = line
            weight = "1"

        skill = skill.strip().lower()

        try:
            numeric_weight = float(
                weight.strip()
            )
        except ValueError:
            numeric_weight = 1.0

        if numeric_weight <= 0:
            numeric_weight = 1.0

        if skill:
            weights[skill] = numeric_weight

    return weights


def skill_matches(skill, matching_skills):
    skill = skill.lower().strip()

    for item in matching_skills:

        item = str(item).lower().strip()

        if skill == item:
            return True

        if skill in item:
            return True

        if item in skill:
            return True

    return False


def compute_weighted_score(
    matching_skills,
    skill_weights
):
    if not skill_weights:
        return None

    total_weight = sum(
        skill_weights.values()
    )

    if total_weight <= 0:
        return None

    matched_weight = 0.0

    for skill, weight in skill_weights.items():

        if skill_matches(
            skill,
            matching_skills
        ):
            matched_weight += weight

    ratio = matched_weight / total_weight

    score = round(ratio * 10)

    return max(
        1,
        min(10, score)
    )


def ats_parse_resume_text(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    name = lines[0] if lines else ""

    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    email = (
        email_match.group(0)
        if email_match
        else ""
    )

    phone_match = re.search(
        r"(\+?\d[\d\s().-]{8,}\d)",
        text
    )

    phone = (
        phone_match.group(0).strip()
        if phone_match
        else ""
    )

    return name, email, phone


def safe_list(value):
    if isinstance(value, list):
        return [
            str(item)
            for item in value
        ]

    if value is None:
        return []

    return [str(value)]


def clean_json_response(content):
    if not content:
        raise ValueError(
            "Groq returned an empty response."
        )

    content = content.strip()

    # Remove JSON markdown fences if present.
    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"^```\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    content = content.strip()

    # First attempt: direct JSON.
    try:
        return json.loads(content)

    except json.JSONDecodeError:
        pass

    # Second attempt: extract the first object.
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Groq did not return valid JSON.\n\n"
            + content
        )

    extracted = content[
        start:end + 1
    ]

    try:
        return json.loads(
            extracted
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Groq returned malformed JSON.\n\n"
            + content
        ) from exc


def normalize_result(
    data,
    candidate
):
    result = {}

    result["candidate_name"] = str(
        data.get(
            "candidate_name",
            candidate["name"]
        )
    )

    result["job_title"] = str(
        data.get(
            "job_title",
            ""
        )
    )

    try:
        score = int(
            float(
                data.get(
                    "score_1_10",
                    1
                )
            )
        )
    except Exception:
        score = 1

    score = max(
        1,
        min(10, score)
    )

    result["score_1_10"] = score

    label = str(
        data.get(
            "label",
            ""
        )
    ).strip()

    if label not in [
        "Bad",
        "Good",
        "Best"
    ]:

        if score <= 4:
            label = "Bad"

        elif score <= 7:
            label = "Good"

        else:
            label = "Best"

    result["label"] = label

    result["matching_skills"] = safe_list(
        data.get(
            "matching_skills",
            []
        )
    )

    result["missing_skills"] = safe_list(
        data.get(
            "missing_skills",
            []
        )
    )

    result["strong_points"] = safe_list(
        data.get(
            "strong_points",
            []
        )
    )

    result["summary"] = str(
        data.get(
            "summary",
            ""
        )
    )

    result["email"] = candidate.get(
        "email",
        ""
    )

    result["phone"] = candidate.get(
        "phone",
        ""
    )

    result["source"] = candidate.get(
        "source",
        ""
    )

    return result


def groq_evaluate_candidate(
    job_description,
    candidate_name,
    resume_text,
):
    if not client:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    if not model_name:
        raise RuntimeError(
            "No Groq model is selected."
        )

    system_prompt = """
You are an expert technical recruiter and ATS-style resume screening assistant.

Evaluate the candidate strictly against the Job Description.

Return ONLY valid JSON.

Required schema:

{
  "candidate_name": "string",
  "job_title": "string",
  "score_1_10": 1,
  "label": "Bad",
  "matching_skills": [],
  "missing_skills": [],
  "strong_points": [],
  "summary": "string"
}

Scoring:

1-4 = Bad
5-7 = Good
8-10 = Best

Rules:

- score_1_10 must be an integer from 1 to 10.
- label must be exactly Bad, Good, or Best.
- matching_skills must contain skills clearly supported by the resume.
- missing_skills must contain important JD requirements absent or weak in the resume.
- strong_points must contain relevant achievements, experience, certifications,
  education, domain expertise, leadership, or other professional strengths.
- Never invent experience or skills.
- Base the evaluation on job-relevant professional evidence only.
- Do not use age, gender, race, religion, marital status, health information,
  nationality, or other protected characteristics.
- Do not add extra JSON fields.
- Do not return markdown.
"""

    user_prompt = (
        "JOB DESCRIPTION:\n\n"
        + job_description
        + "\n\n"
        + "CANDIDATE NAME:\n\n"
        + candidate_name
        + "\n\n"
        + "CANDIDATE RESUME:\n\n"
        + resume_text
    )

    try:

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.strip()
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={
                "type": "json_object"
            }
        )

    except Exception as exc:

        error_text = str(exc).lower()

        if (
            "429" in error_text
            or "rate limit" in error_text
            or "rate_limit" in error_text
        ):

            raise RuntimeError(
                "Groq rate limit reached. "
                "Please wait and try again."
            ) from exc

        if (
            "404" in error_text
            or "model_not_found" in error_text
        ):

            raise RuntimeError(
                f"The selected model "
                f"'{model_name}' is not accessible "
                f"to this Groq API key."
            ) from exc

        raise

    content = (
        response
        .choices[0]
        .message
        .content
    )

    return clean_json_response(
        content
    )


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.subheader("1. Job Description")

job_description = st.text_area(
    "Paste the full Job Description:",
    height=220,
    placeholder=(
        "Include required skills, experience, tools, "
        "certifications and domain requirements."
    ),
    key=(
        f"jd_"
        f"{st.session_state['reset_id']}"
    )
)

jd_skills = []

if job_description.strip():

    jd_skills = extract_jd_skills(
        job_description
    )

    with st.expander(
        "Automatically Extracted JD Skills / Keywords"
    ):

        if jd_skills:

            st.write(
                ", ".join(jd_skills)
            )

        else:

            st.write(
                "No meaningful keywords detected."
            )

st.markdown("---")


# ============================================================
# RESUMES
# ============================================================

st.subheader(
    "2. Candidate Resumes"
)

st.write(
    "Maximum 5 resumes per screening run."
)

tab1, tab2 = st.tabs(
    [
        "Paste Resume Text",
        "Upload PDF/TXT"
    ]
)

candidates = []


# ============================================================
# PASTE RESUMES
# ============================================================

with tab1:

    pasted_count = st.number_input(
        "Number of resumes to paste:",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        key=(
            f"paste_count_"
            f"{st.session_state['reset_id']}"
        )
    )

    for i in range(
        int(pasted_count)
    ):

        candidate_name = st.text_input(
            f"Candidate {i + 1} Name",
            key=(
                f"paste_name_"
                f"{st.session_state['reset_id']}_"
                f"{i}"
            )
        )

        resume_text = st.text_area(
            f"Resume {i + 1}",
            height=220,
            key=(
                f"paste_resume_"
                f"{st.session_state['reset_id']}_"
                f"{i}"
            ),
            placeholder="Paste resume text here..."
        )

        if resume_text.strip():

            parsed_name, email, phone = (
                ats_parse_resume_text(
                    resume_text
                )
            )

            final_name = (
                candidate_name.strip()
                if candidate_name.strip()
                else parsed_name
                or f"Candidate_{i + 1}"
            )

            candidates.append(
                {
                    "name": final_name,
                    "resume_text": resume_text.strip(),
                    "email": email,
                    "phone": phone,
                    "source": "Pasted"
                }
            )


# ============================================================
# UPLOAD RESUMES
# ============================================================

with tab2:

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT resumes:",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=(
            f"uploads_"
            f"{st.session_state['reset_id']}"
        ),
        help="Maximum 5 resumes."
    )

    if uploaded_files:

        if len(uploaded_files) > MAX_RESUMES:

            st.warning(
                f"You selected "
                f"{len(uploaded_files)} files. "
                f"Only the first "
                f"{MAX_RESUMES} will be used."
            )

        for i, uploaded_file in enumerate(
            uploaded_files[:MAX_RESUMES]
        ):

            filename = uploaded_file.name

            extracted_text = ""

            if filename.lower().endswith(
                ".txt"
            ):

                try:

                    extracted_text = (
                        uploaded_file
                        .getvalue()
                        .decode(
                            "utf-8",
                            errors="ignore"
                        )
                    )

                except Exception as exc:

                    st.error(
                        f"Error reading "
                        f"{filename}: {exc}"
                    )

            elif filename.lower().endswith(
                ".pdf"
            ):

                if not PDF_AVAILABLE:

                    st.error(
                        "PyPDF2 is not installed."
                    )

                else:

                    try:

                        reader = (
                            PyPDF2.PdfReader(
                                uploaded_file
                            )
                        )

                        pages = []

                        for page in reader.pages:

                            text = page.extract_text()

                            if text:
                                pages.append(text)

                        extracted_text = (
                            "\n".join(pages)
                            .strip()
                        )

                    except Exception as exc:

                        st.error(
                            f"Error reading "
                            f"{filename}: {exc}"
                        )

            if extracted_text.strip():

                edited_text = st.text_area(
                    f"Extracted text - {filename}",
                    value=extracted_text,
                    height=220,
                    key=(
                        f"file_text_"
                        f"{st.session_state['reset_id']}_"
                        f"{i}"
                    )
                )

                parsed_name, email, phone = (
                    ats_parse_resume_text(
                        edited_text
                    )
                )

                candidate_name = (
                    parsed_name
                    or filename.rsplit(
                        ".",
                        1
                    )[0]
                )

                candidates.append(
                    {
                        "name": candidate_name,
                        "resume_text": edited_text.strip(),
                        "email": email,
                        "phone": phone,
                        "source": "Uploaded"
                    }
                )

            else:

                st.warning(
                    f"No text could be extracted from "
                    f"{filename}."
                )


if len(candidates) > MAX_RESUMES:

    candidates = candidates[
        :MAX_RESUMES
    ]

st.info(
    f"Candidates ready: "
    f"{len(candidates)} / {MAX_RESUMES}"
)

st.markdown("---")


# ============================================================
# SKILL WEIGHTING
# ============================================================

st.subheader(
    "3. Skill Weighting"
)

skill_weights_text = st.text_area(
    "Enter weighted skills:",
    height=150,
    placeholder=(
        "python,3\n"
        "sql,2\n"
        "aws,2\n"
        "communication,1"
    ),
    key=(
        f"weights_"
        f"{st.session_state['reset_id']}"
    )
)

skill_weights = parse_skill_weights(
    skill_weights_text
)

if skill_weights:

    st.success(
        f"Loaded {len(skill_weights)} weighted skills."
    )

else:

    st.info(
        "No skill weights defined. "
        "Ranking will use the AI score."
    )

st.markdown("---")


# ============================================================
# ANALYSIS
# ============================================================

st.subheader(
    "4. AI Analysis"
)

if st.button(
    "Analyze All Candidates",
    type="primary",
    use_container_width=True
):

    if not client:

        st.error(
            "GROQ_API_KEY is not configured."
        )

    elif not model_name:

        st.error(
            "No Groq model is selected."
        )

    elif not job_description.strip():

        st.error(
            "Please enter the Job Description."
        )

    elif not candidates:

        st.error(
            "Please add at least one resume."
        )

    else:

        results = []

        progress = st.progress(0)

        status = st.empty()

        total = len(candidates)

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            status.write(
                f"Analyzing "
                f"{candidate['name']} "
                f"({index}/{total}) using "
                f"`{model_name}`"
            )

            try:

                raw_result = (
                    groq_evaluate_candidate(
                        job_description,
                        candidate["name"],
                        candidate["resume_text"]
                    )
                )

                result = normalize_result(
                    raw_result,
                    candidate
                )

                result[
                    "weighted_score_1_10"
                ] = compute_weighted_score(
                    result[
                        "matching_skills"
                    ],
                    skill_weights
                )

                results.append(
                    result
                )

            except Exception as exc:

                st.error(
                    f"Error evaluating "
                    f"{candidate['name']}:\n{exc}"
                )

            progress.progress(
                index / total
            )

            # Small delay reduces burstiness.
            if index < total:
                time.sleep(0.5)

        progress.empty()
        status.empty()

        st.session_state[
            "results"
        ] = results

        if results:

            st.success(
                f"Completed analysis for "
                f"{len(results)} candidate(s)."
            )

        else:

            st.error(
                "No candidate results were returned."
            )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.get(
    "results",
    []
)

if results:

    st.markdown("---")

    st.subheader(
        "5. Final Ranked Results"
    )

    def rank_key(result):

        weighted = result.get(
            "weighted_score_1_10"
        )

        if weighted is not None:
            return weighted

        return result.get(
            "score_1_10",
            0
        )

    ranked_results = sorted(
        results,
        key=rank_key,
        reverse=True
    )

    rows = []

    for rank, result in enumerate(
        ranked_results,
        start=1
    ):

        rows.append(
            {
                "Rank": rank,
                "Candidate": result.get(
                    "candidate_name",
                    ""
                ),
                "Job Title": result.get(
                    "job_title",
                    ""
                ),
                "AI Score": result.get(
                    "score_1_10",
                    ""
                ),
                "Weighted Score": result.get(
                    "weighted_score_1_10",
                    ""
                ),
                "Label": result.get(
                    "label",
                    ""
                ),
                "Email": result.get(
                    "email",
                    ""
                ),
                "Phone": result.get(
                    "phone",
                    ""
                ),
                "Matching Skills": ", ".join(
                    result.get(
                        "matching_skills",
                        []
                    )
                ),
                "Missing Skills": ", ".join(
                    result.get(
                        "missing_skills",
                        []
                    )
                ),
                "Strong Points": ", ".join(
                    result.get(
                        "strong_points",
                        []
                    )
                ),
                "Summary": result.get(
                    "summary",
                    ""
                )
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader(
        "6. Dashboard"
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    numeric_scores = pd.to_numeric(
        df["AI Score"],
        errors="coerce"
    )

    average_score = numeric_scores.mean()
    highest_score = numeric_scores.max()

    with metric1:
        st.metric(
            "Candidates",
            len(df)
        )

    with metric2:
        st.metric(
            "Average Score",
            (
                f"{average_score:.2f}/10"
                if not pd.isna(average_score)
                else "N/A"
            )
        )

    with metric3:
        st.metric(
            "Highest Score",
            (
                f"{int(highest_score)}/10"
                if not pd.isna(highest_score)
                else "N/A"
            )
        )

    with metric4:

        if not df.empty:

            common_label = (
                df["Label"]
                .mode()
                .iloc[0]
            )

        else:

            common_label = "N/A"

        st.metric(
            "Most Common Label",
            common_label
        )

    chart_df = df[
        ["Candidate", "AI Score"]
    ].copy()

    chart_df["AI Score"] = pd.to_numeric(
        chart_df["AI Score"],
        errors="coerce"
    )

    chart_df = chart_df.dropna()

    if not chart_df.empty:

        chart_df = chart_df.set_index(
            "Candidate"
        )

        st.bar_chart(
            chart_df
        )


    # ========================================================
    # HEATMAP
    # ========================================================

    st.subheader(
        "7. Candidate Similarity Heatmap"
    )

    count = len(ranked_results)

    if count >= 2:

        similarity = np.zeros(
            (count, count)
        )

        skill_sets = []

        for result in ranked_results:

            skills = {
                str(skill)
                .lower()
                .strip()
                for skill in result.get(
                    "matching_skills",
                    []
                )
            }

            skill_sets.append(
                skills
            )

        for i in range(count):

            for j in range(count):

                if i == j:

                    similarity[i, j] = 1.0

                else:

                    union = (
                        skill_sets[i]
                        | skill_sets[j]
                    )

                    intersection = (
                        skill_sets[i]
                        & skill_sets[j]
                    )

                    if union:

                        similarity[i, j] = (
                            len(intersection)
                            / len(union)
                        )

        fig, ax = plt.subplots(
            figsize=(
                max(7, count * 1.5),
                max(5, count * 1.2)
            )
        )

        image = ax.imshow(
            similarity,
            vmin=0,
            vmax=1
        )

        ax.set_xticks(
            range(count)
        )

        ax.set_yticks(
            range(count)
        )

        ax.set_xticklabels(
            [
                result["candidate_name"]
                for result in ranked_results
            ],
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            [
                result["candidate_name"]
                for result in ranked_results
            ]
        )

        ax.set_title(
            "Candidate Similarity Based on Matching Skills"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="Similarity"
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    else:

        st.info(
            "At least two candidates are required for the heatmap."
        )


    # ========================================================
    # DETAILS
    # ========================================================

    st.subheader(
        "8. Detailed Candidate Reports"
    )

    for rank, result in enumerate(
        ranked_results,
        start=1
    ):

        with st.expander(
            f"#{rank} - "
            f"{result['candidate_name']}"
        ):

            score = result[
                "score_1_10"
            ]

            weighted = result.get(
                "weighted_score_1_10"
            )

            st.write(
                f"AI Score: {score}/10"
            )

            if weighted is not None:

                st.write(
                    f"Weighted Score: "
                    f"{weighted}/10"
                )

            st.write(
                f"Label: "
                f"{result['label']}"
            )

            st.progress(
                score / 10
            )

            if result.get("email"):
                st.write(
                    f"Email: "
                    f"{result['email']}"
                )

            if result.get("phone"):
                st.write(
                    f"Phone: "
                    f"{result['phone']}"
                )

            st.markdown(
                "### Matching Skills"
            )

            if result[
                "matching_skills"
            ]:

                for skill in result[
                    "matching_skills"
                ]:

                    st.write(
                        f"- {skill}"
                    )

            else:

                st.write(
                    "No matching skills identified."
                )

            st.markdown(
                "### Missing Skills"
            )

            if result[
                "missing_skills"
            ]:

                for skill in result[
                    "missing_skills"
                ]:

                    st.write(
                        f"- {skill}"
                    )

            else:

                st.write(
                    "No significant missing skills identified."
                )

            st.markdown(
                "### Strong Points"
            )

            if result[
                "strong_points"
            ]:

                for point in result[
                    "strong_points"
                ]:

                    st.write(
                        f"- {point}"
                    )

            else:

                st.write(
                    "No additional strong points identified."
                )

            st.markdown(
                "### AI Summary"
            )

            st.write(
                result[
                    "summary"
                ]
            )


    # ========================================================
    # CSV
    # ========================================================

    st.subheader(
        "9. Export Results"
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="resume_screening_results.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # EXCEL
    # ========================================================

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="xlsxwriter"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Resume Scores"
        )

        workbook = writer.book

        worksheet = writer.sheets[
            "Resume Scores"
        ]

        header_format = (
            workbook.add_format(
                {
                    "bold": True,
                    "text_wrap": True,
                    "valign": "top"
                }
            )
        )

        for col_index, col_name in enumerate(
            df.columns
        ):

            worksheet.write(
                0,
                col_index,
                col_name,
                header_format
            )

        worksheet.set_column(
            0,
            len(df.columns) - 1,
            22
        )

        worksheet.freeze_panes(
            1,
            0
        )

    excel_bytes = (
        excel_buffer.getvalue()
    )

    st.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name="resume_screening_results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )


    # ========================================================
    # EMAIL REPORT
    # ========================================================

    st.subheader(
        "10. Email-ready Report"
    )

    email_lines = [
        "Subject: AI Resume Screening Report",
        "",
        "Dear Hiring Manager,",
        "",
        "Please find below the AI-generated resume screening results.",
        ""
    ]

    if jd_skills:

        email_lines.append(
            "Key JD Skills: "
            + ", ".join(
                jd_skills[:10]
            )
        )

        email_lines.append("")

    for rank, result in enumerate(
        ranked_results,
        start=1
    ):

        email_lines.append(
            f"{rank}. "
            f"{result['candidate_name']}"
        )

        email_lines.append(
            f"AI Score: "
            f"{result['score_1_10']}/10"
        )

        if result.get(
            "weighted_score_1_10"
        ) is not None:

            email_lines.append(
                f"Weighted Score: "
                f"{result['weighted_score_1_10']}/10"
            )

        email_lines.append(
            f"Label: "
            f"{result['label']}"
        )

        email_lines.append(
            "Matching Skills: "
            + ", ".join(
                result[
                    "matching_skills"
                ]
            )
        )

        email_lines.append(
            "Missing Skills: "
            + ", ".join(
                result[
                    "missing_skills"
                ]
            )
        )

        email_lines.append(
            "Strong Points: "
            + ", ".join(
                result[
                    "strong_points"
                ]
            )
        )

        email_lines.append(
            "Summary: "
            + result[
                "summary"
            ]
        )

        email_lines.append("")

    email_lines.extend(
        [
            "Regards,",
            "AI Resume Screening App"
        ]
    )

    email_report = "\n".join(
        email_lines
    )

    st.text_area(
        "Copy into Outlook or Gmail:",
        value=email_report,
        height=400
    )


st.markdown("---")

st.caption(
    "Powered by Groq. "
    "The model list is detected automatically from your Groq API key."
)
