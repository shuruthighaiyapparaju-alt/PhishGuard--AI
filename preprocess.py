"""
preprocess.py
-------------
Shared helper code used by BOTH train_model.py and app.py.

It contains:
  1. File paths used across the project
  2. Dataset loading and cleaning (missing values, duplicates)
  3. Text cleaning (lowercase, remove junk characters)
  4. URL extraction and warning-indicator features
  5. The scikit-learn Pipeline (TF-IDF + URL features + Logistic Regression)
  6. Helper functions that explain a prediction in simple words

IMPORTANT: URL and keyword features are WARNING INDICATORS only.
They are not proof that an email is malicious.
"""

import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ---------------------------------------------------------------------------
# 1. Paths (every file in the project uses these same paths)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emails.csv"
MODEL_PATH = BASE_DIR / "models" / "phishing_model.joblib"
RESULTS_DIR = BASE_DIR / "results"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
CLASS_DIST_PATH = RESULTS_DIR / "class_distribution.png"
METRICS_PATH = RESULTS_DIR / "metrics.json"

LABEL_NAMES = {0: "Safe", 1: "Phishing"}

# ---------------------------------------------------------------------------
# 2. Word lists used for warning indicators
# ---------------------------------------------------------------------------
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "bl.ink",
}

SUSPICIOUS_WORDS = [
    "urgent", "immediately", "verify", "verification", "suspended", "suspend",
    "password", "confirm your", "account", "bank", "login", "log in",
    "click here", "limited time", "winner", "won", "prize", "lottery",
    "security alert", "unusual activity", "update your", "expire", "expires",
    "act now", "free gift", "gift card", "invoice", "refund", "wire transfer",
    "social security", "credit card", "billing",
]

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "mail.com", "proton.me", "protonmail.com", "gmx.com", "yandex.com",
}

BRAND_NAMES = ["paypal", "amazon", "microsoft", "apple", "google", "netflix",
               "facebook", "instagram", "bank", "irs", "dhl", "fedex"]

# ---------------------------------------------------------------------------
# 3. Regular expressions
# ---------------------------------------------------------------------------
URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s<>\"'\)\]]+", re.IGNORECASE)
IP_REGEX = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
HTML_TAG_REGEX = re.compile(r"<[^>]+>")
EMAIL_ADDR_REGEX = re.compile(r"[\w\.\-+]+@[\w\-]+\.[\w\.\-]+")
NON_LETTER_REGEX = re.compile(r"[^a-z\s]")
SPACES_REGEX = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 4. Dataset loading and cleaning
# ---------------------------------------------------------------------------
# Different public datasets use different column names. We rename them to
# the four names used in this project: sender, subject, body, label.
COLUMN_ALIASES = {
    "sender": ["sender", "from", "from_address"],
    "subject": ["subject", "email_subject"],
    "body": ["body", "email text", "email_text", "text", "text_combined",
             "message", "content"],
    "label": ["label", "email type", "email_type", "class", "target"],
}

LABEL_ALIASES = {
    "1": 1, "phishing": 1, "phishing email": 1, "spam": 1, "malicious": 1,
    "0": 0, "safe": 0, "safe email": 0, "legitimate": 0, "ham": 0,
}


def load_dataset(path=DATA_PATH, max_rows=None):
    """Read the CSV file and return a DataFrame with columns
    sender, subject, body, label (label: 0 = Safe, 1 = Phishing).

    Raises FileNotFoundError / ValueError with simple messages.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download a phishing email CSV and save it as data/emails.csv "
            "(see README.md, Step 5)."
        )

    try:
        raw = pd.read_csv(path, engine="python", on_bad_lines="skip",
                          encoding="utf-8", encoding_errors="replace")
    except Exception as exc:  # unreadable or corrupted file
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    if max_rows:
        raw = raw.head(max_rows)

    # Make column names lowercase so matching is easy
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    df = pd.DataFrame()
    for target, options in COLUMN_ALIASES.items():
        found = next((c for c in options if c in raw.columns), None)
        if found is not None:
            df[target] = raw[found]
        elif target in ("sender", "subject"):
            df[target] = ""  # optional columns
        else:
            raise ValueError(
                f"The CSV has no '{target}' column. Columns found: "
                f"{list(raw.columns)}. Expected something like "
                f"{options}."
            )

    df["label"] = _convert_labels(df["label"])
    return df


def _convert_labels(series):
    """Turn label values such as 'Phishing Email' / 'ham' / 1 into 0 or 1."""
    cleaned = series.astype(str).str.strip().str.lower()
    # "1.0" -> "1"
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    converted = cleaned.map(LABEL_ALIASES)
    unknown = sorted(set(cleaned[converted.isna()]) - {"nan", ""})
    if unknown:
        raise ValueError(
            f"Unknown label values in the CSV: {unknown[:10]}. "
            "Use 0/1 or names like 'Phishing Email' / 'Safe Email'."
        )
    return converted


def clean_dataframe(df):
    """Handle missing values and duplicate records.

    Returns (clean_df, stats). Duplicates are removed BEFORE the train/test
    split so the same email can never appear in both sets (data leakage).
    """
    stats = {"rows_loaded": int(len(df))}

    df = df.copy()
    for col in ("sender", "subject", "body"):
        df[col] = df[col].fillna("").astype(str)

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    stats["rows_after_missing_label"] = int(len(df))

    # Remove emails that have no content at all
    has_text = (df["subject"].str.strip() + df["body"].str.strip()) != ""
    df = df[has_text]
    stats["rows_after_empty_removed"] = int(len(df))

    # Remove duplicates (same subject + body)
    df = df.drop_duplicates(subset=["subject", "body"])
    stats["rows_after_duplicates_removed"] = int(len(df))

    df = df.reset_index(drop=True)
    return df, stats


# ---------------------------------------------------------------------------
# 5. Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text):
    """Lowercase the text and remove unnecessary characters.

    URLs are replaced with the word 'urltoken' so the model can still learn
    that a link was present.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = HTML_TAG_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" urltoken ", text)
    text = EMAIL_ADDR_REGEX.sub(" emailtoken ", text)
    text = NON_LETTER_REGEX.sub(" ", text)
    return SPACES_REGEX.sub(" ", text).strip()


def combine_text(df):
    """Join subject and body, then clean. Used inside the Pipeline."""
    combined = df["subject"].fillna("").astype(str) + " " + \
        df["body"].fillna("").astype(str)
    return combined.apply(clean_text)


# ---------------------------------------------------------------------------
# 6. URL extraction and warning indicators
# ---------------------------------------------------------------------------
def extract_urls(text):
    """Return a list of URLs found in the text."""
    if not isinstance(text, str):
        return []
    urls = URL_REGEX.findall(text)
    return [u.rstrip(".,;:!?") for u in urls]


def get_host(url):
    """Return the host name of a URL (e.g. 'www.example.com')."""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def analyze_url(url):
    """Return a list of warning notes for ONE URL."""
    notes = []
    host = get_host(url)
    if IP_REGEX.match(host):
        notes.append("uses an IP address instead of a domain name")
    if host in URL_SHORTENERS:
        notes.append("uses a URL shortener (hides the real destination)")
    if "@" in url:
        notes.append("contains '@' (can disguise the real destination)")
    if url.lower().startswith("http://"):
        notes.append("not encrypted (http instead of https)")
    if host.count(".") >= 4:
        notes.append("has many sub-domains")
    if "xn--" in host:
        notes.append("uses punycode (may imitate another website)")
    return notes


def find_suspicious_words(text):
    """Return the suspicious words/phrases found in the text."""
    lowered = str(text).lower()
    found = []
    for word in SUSPICIOUS_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            found.append(word)
    return found


def sender_indicators(sender):
    """Simple warning notes about the sender address (shown in the app only;
    NOT used by the model, because sender patterns differ between datasets)."""
    notes = []
    if not sender or not sender.strip():
        return notes
    match = EMAIL_ADDR_REGEX.search(sender)
    if not match:
        return ["sender address does not look like a valid email address"]
    address = match.group(0).lower()
    domain = address.split("@")[-1]
    display = sender.lower().replace(address, "")
    if domain in FREE_EMAIL_DOMAINS:
        notes.append(f"sent from a free email provider ({domain})")
    if re.search(r"\d", domain.split(".")[0]):
        notes.append("sender domain contains digits")
    for brand in BRAND_NAMES:
        if (brand in display or brand in address.split("@")[0]) \
                and brand not in domain:
            notes.append(
                f"mentions '{brand}' but the domain is '{domain}'"
            )
            break
    return notes


# The numeric features given to the model (order matters!)
FEATURE_NAMES = [
    "num_urls", "has_ip_url", "has_shortener_url", "has_at_symbol_url",
    "num_http_only_urls", "num_suspicious_words", "num_exclamation_marks",
    "uppercase_ratio", "log_text_length",
]


def _row_features(subject, body):
    text = f"{subject} {body}"
    urls = extract_urls(text)
    hosts = [get_host(u) for u in urls]
    letters = [c for c in text if c.isalpha()]
    upper_ratio = (sum(c.isupper() for c in letters) / len(letters)
                   if letters else 0.0)
    return [
        len(urls),
        int(any(IP_REGEX.match(h) for h in hosts)),
        int(any(h in URL_SHORTENERS for h in hosts)),
        int(any("@" in u for u in urls)),
        sum(u.lower().startswith("http://") for u in urls),
        len(find_suspicious_words(text)),
        text.count("!"),
        upper_ratio,
        np.log1p(len(text)),
    ]


def build_feature_frame(df):
    """Create the numeric URL/keyword features for every email in df."""
    rows = [
        _row_features(str(s) if pd.notna(s) else "",
                      str(b) if pd.notna(b) else "")
        for s, b in zip(df["subject"], df["body"])
    ]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# ---------------------------------------------------------------------------
# 7. The machine learning pipeline
# ---------------------------------------------------------------------------
def build_pipeline():
    """TF-IDF text features + numeric URL features -> Logistic Regression.

    Everything is inside ONE Pipeline, so when we call fit() on the training
    data only, the TF-IDF vocabulary and the scaler learn from training data
    only. This prevents data leakage.
    """
    text_branch = Pipeline([
        ("clean", FunctionTransformer(combine_text)),
        ("tfidf", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2,
            stop_words="english", sublinear_tf=True)),
    ])
    url_branch = Pipeline([
        ("extract", FunctionTransformer(build_feature_frame)),
        ("scale", StandardScaler()),
    ])
    features = FeatureUnion([("text", text_branch), ("url", url_branch)])

    return Pipeline([
        ("features", features),
        # class_weight="balanced" handles class imbalance
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42)),
    ])


# ---------------------------------------------------------------------------
# 8. Explaining a prediction
# ---------------------------------------------------------------------------
def top_contributing_terms(pipeline, email_df, n=5):
    """Return (phishing_terms, safe_terms): the words that pushed the score
    most towards each class for ONE email (a 1-row DataFrame)."""
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    vocab = features.transformer_list[0][1].named_steps["tfidf"] \
        .get_feature_names_out()

    row = features.transform(email_df).tocsr()
    coefs = classifier.coef_[0]
    contributions = row.data * coefs[row.indices]

    words = []
    for idx, value in zip(row.indices, contributions):
        if idx < len(vocab):  # skip the numeric URL features
            words.append((vocab[idx], float(value)))
    words.sort(key=lambda item: item[1], reverse=True)

    phishing_terms = [w for w, v in words if v > 0][:n]
    safe_terms = [w for w, v in sorted(words, key=lambda i: i[1]) if v < 0][:n]
    return phishing_terms, safe_terms
