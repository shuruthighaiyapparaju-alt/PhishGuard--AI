"""
app.py
------
Streamlit web application for the Phishing Email Detector.

Run it from the project folder:
    streamlit run app.py

Pages:
  - Analyze Email    : paste an email and get a prediction
  - Model Evaluation : shows the results saved by train_model.py
"""

import json

import joblib
import pandas as pd
import streamlit as st

from preprocess import (CLASS_DIST_PATH, CONFUSION_MATRIX_PATH, FEATURE_NAMES,
                        METRICS_PATH, MODEL_PATH, analyze_url, extract_urls,
                        find_suspicious_words, sender_indicators,
                        top_contributing_terms)

st.set_page_config(page_title="Phishing Email Detector",
                   page_icon="🛡️", layout="centered")

# ---- A little CSS for a clean look ----------------------------------------
st.markdown("""
<style>
.result-box {padding: 1rem 1.25rem; border-radius: 10px; font-size: 1.4rem;
             font-weight: 700; margin: 0.5rem 0 1rem 0;}
.result-phishing {background:#fdecea; color:#b3261e; border:1px solid #f5b5b0;}
.result-safe {background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7;}
.small-note {color:#666; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)


# ---- Loading helpers (with error handling) --------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved pipeline. Raises FileNotFoundError if missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model file was not found "
            f"({MODEL_PATH.name}).")
    return joblib.load(MODEL_PATH)["pipeline"]


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


MODEL_HELP = (
    "Train the model first. In the VS Code terminal run:\n\n"
    "`python train_model.py`\n\n"
    "Make sure `data/emails.csv` exists (see README.md), then refresh this "
    "page."
)


# ---- Page 1: Analyze Email --------------------------------------------------
def page_analyze():
    st.title("🛡️ Phishing Email Detector")
    st.caption("Paste an email below. A machine learning model (TF-IDF + "
               "Logistic Regression) will estimate whether it looks like "
               "phishing.")

    try:
        pipeline = load_model()
    except FileNotFoundError as error:
        st.error(f"{error}")
        st.info(MODEL_HELP)
        return
    except Exception as error:  # e.g. corrupted file / version mismatch
        st.error(f"The model file could not be loaded: {error}")
        st.info("Delete `models/phishing_model.joblib` and train again with "
                "`python train_model.py`.")
        return

    sender = st.text_input("Sender address",
                           placeholder="support@example.com")
    subject = st.text_input("Email subject",
                            placeholder="Your account has been suspended")
    body = st.text_area("Email body", height=220,
                        placeholder="Paste the email text here ...")

    if st.button("Analyze Email", type="primary"):
        if not subject.strip() and not body.strip():
            st.error("Please enter an email subject or body before "
                     "analyzing.")
            return
        if len((subject + body).strip()) < 15:
            st.warning("The text is very short, so the result is not "
                       "reliable. Paste the full email for a better "
                       "analysis.")

        email_df = pd.DataFrame({"subject": [subject], "body": [body]})
        try:
            prediction = int(pipeline.predict(email_df)[0])
            phishing_probability = float(
                pipeline.predict_proba(email_df)[0][1])
        except Exception as error:
            st.error(f"Something went wrong while analyzing the email: "
                     f"{error}")
            return

        show_result(pipeline, email_df, sender, subject, body,
                    prediction, phishing_probability)


def show_result(pipeline, email_df, sender, subject, body, prediction,
                phishing_probability):
    st.subheader("Result")
    if prediction == 1:
        st.markdown('<div class="result-box result-phishing">'
                    '⚠️ Phishing</div>', unsafe_allow_html=True)
        score = phishing_probability
        score_label = "Model score for 'Phishing'"
    else:
        st.markdown('<div class="result-box result-safe">'
                    '✅ Safe</div>', unsafe_allow_html=True)
        score = 1 - phishing_probability
        score_label = "Model score for 'Safe'"

    st.metric(score_label, f"{score * 100:.1f}%")
    st.progress(min(max(score, 0.0), 1.0))
    st.markdown(
        '<p class="small-note">This score is the model\'s own estimate. '
        'It is <b>not</b> a guarantee and may be wrong, especially for '
        'emails that look different from its training data. Always verify '
        'suspicious emails through an official channel.</p>',
        unsafe_allow_html=True)

    # ---- URLs ----
    st.subheader("URLs found")
    urls = extract_urls(subject + " " + body)
    if urls:
        table = pd.DataFrame({
            "URL": urls,
            "Warnings": ["; ".join(analyze_url(u)) or "none detected"
                         for u in urls],
        })
        st.dataframe(table, hide_index=True)
    else:
        st.write("No URLs were found in this email.")

    # ---- Indicators ----
    st.subheader("Suspicious indicators")
    indicators = []
    words = find_suspicious_words(subject + " " + body)
    if words:
        indicators.append("Suspicious words/phrases: " + ", ".join(words))
    if len(urls) >= 3:
        indicators.append(f"Contains {len(urls)} URLs")
    for u in urls:
        for note in analyze_url(u):
            indicators.append(f"URL {note}")
    indicators.extend(f"Sender: {n}" for n in sender_indicators(sender))
    exclaims = (subject + body).count("!")
    if exclaims >= 3:
        indicators.append(f"Many exclamation marks ({exclaims})")

    if indicators:
        for item in dict.fromkeys(indicators):  # remove repeats, keep order
            st.write(f"- {item}")
    else:
        st.write("No warning indicators were detected.")
    st.caption("These are warning signs only. They do not prove that an "
               "email is malicious, and legitimate emails can contain "
               "them too.")

    # ---- Explanation ----
    st.subheader("Why did the model say this?")
    try:
        phishing_terms, safe_terms = top_contributing_terms(
            pipeline, email_df, n=5)
    except Exception:
        phishing_terms, safe_terms = [], []

    verdict = "Phishing" if prediction == 1 else "Safe"
    text = f"The model classified this email as **{verdict}**. "
    if phishing_terms:
        text += ("Words that pushed the score towards phishing: "
                 f"**{', '.join(phishing_terms)}**. ")
    if safe_terms:
        text += ("Words that pushed it towards safe: "
                 f"**{', '.join(safe_terms)}**. ")
    if not phishing_terms and not safe_terms:
        text += ("Few words in this email were recognised by the model, "
                 "so the decision is weak.")
    st.write(text)
    st.caption("The model also uses numeric features: " +
               ", ".join(FEATURE_NAMES) + ".")


# ---- Page 2: Model Evaluation -----------------------------------------------
def page_evaluation():
    st.title("📊 Model Evaluation")
    metrics = load_metrics()
    if metrics is None:
        st.error("No evaluation results found (results/metrics.json).")
        st.info(MODEL_HELP)
        return

    st.caption(f"Trained at {metrics['trained_at']} on "
               f"{metrics['train_size']} emails and tested on "
               f"{metrics['test_size']} unseen emails. "
               "'Phishing' is the positive class.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.3f}")
    c4.metric("F1-score", f"{metrics['f1_score']:.3f}")
    st.caption("Do not judge the model by accuracy alone - check precision "
               "and recall too, especially if the classes are imbalanced.")

    st.subheader("Confusion matrix")
    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH))
    else:
        st.warning("confusion_matrix.png not found. Run train_model.py.")

    cm = metrics["confusion_matrix"]
    st.markdown(f"""
| Term | Meaning in simple words | Count |
|---|---|---|
| **True Positive (TP)** | A phishing email correctly caught as phishing | {cm['TP']} |
| **True Negative (TN)** | A safe email correctly marked as safe | {cm['TN']} |
| **False Positive (FP)** | A safe email wrongly flagged as phishing (false alarm) | {cm['FP']} |
| **False Negative (FN)** | A phishing email wrongly marked safe (missed threat - the most dangerous error) | {cm['FN']} |
""")

    st.subheader("Dataset information")
    ds = metrics["dataset_stats"]
    st.write(f"Rows loaded: {ds['rows_loaded']} → after removing empty rows: "
             f"{ds['rows_after_empty_removed']} → after removing "
             f"duplicates: {ds['rows_after_duplicates_removed']}")
    st.write(f"Class counts: Safe = {metrics['class_counts']['Safe']}, "
             f"Phishing = {metrics['class_counts']['Phishing']}")
    if CLASS_DIST_PATH.exists():
        st.image(str(CLASS_DIST_PATH), width=380)


# ---- Navigation --------------------------------------------------------------
page = st.sidebar.radio("Go to", ["Analyze Email", "Model Evaluation"])
st.sidebar.markdown("---")
st.sidebar.caption("College project: Phishing Email Detection using "
                   "Scikit-learn. This tool is an educational demo and "
                   "should not replace real email security software.")

if page == "Analyze Email":
    page_analyze()
else:
    page_evaluation()
