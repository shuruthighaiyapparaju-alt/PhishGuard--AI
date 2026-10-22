# Phishing Email Detection Model Using Scikit-learn

A beginner-friendly machine learning project that classifies an email as **Phishing** or **Safe** using email text (TF-IDF) and URL/keyword warning features, with a Streamlit web interface.

> **Educational demo.** It is not a replacement for real email security software. No accuracy is claimed here: you generate your own results by training on your own copy of the dataset.

---

## 1. Project architecture

```
 emails.csv ──► load + clean ──► train/test split (80/20, stratified)
 (dataset)      (missing values,        │
                 duplicates)            ▼
                              ┌─────────────────────────────┐
                              │ Pipeline (fit on TRAIN only)│
                              │  A. text → clean → TF-IDF   │
                              │  B. URL/keyword features    │
                              │     → StandardScaler        │
                              │  C. Logistic Regression     │
                              └──────────────┬──────────────┘
                                             │
              evaluate on unseen TEST set ◄──┤
              (accuracy, precision, recall,  │
               F1, confusion matrix)         ▼
                                   models/phishing_model.joblib
                                             │
                                             ▼
                              Streamlit app (app.py): paste email
                              → prediction + score + URLs + indicators
```

## 2. Files in this project

| File / folder | Purpose |
|---|---|
| `preprocess.py` | Shared helpers: paths, dataset loading/cleaning, text cleaning, URL extraction, warning indicators, the ML pipeline, prediction explanation. Both other scripts import it, so names always match. |
| `train_model.py` | Trains and evaluates the model; saves the model, confusion matrix and metrics. |
| `app.py` | Streamlit web app (two pages: *Analyze Email* and *Model Evaluation*). |
| `requirements.txt` | List of Python libraries to install. |
| `.gitignore` | Tells Git to ignore the virtual environment, big data file and generated files. |
| `data/emails.csv` | **You add this** (the dataset, Step 5). |
| `models/phishing_model.joblib` | **Generated** by `train_model.py`. |
| `results/confusion_matrix.png`, `class_distribution.png`, `metrics.json` | **Generated** by `train_model.py`. |
| `docs/PROJECT_REPORT.md` | Report text for college submission. |
| `docs/VIVA_QA.md` | Viva questions and answers. |

## 3. Dataset (recommended)

**Phishing Email Curated Datasets** (Champa, Rabbi & Zibran), on Zenodo: <https://zenodo.org/records/8339691>. It contains several real-world email datasets. Use the file **`CEAS_08.csv`**.

Expected columns: `sender, receiver, date, subject, body, label, urls`
(`label`: `1` = phishing/spam, `0` = legitimate).

Important honesty notes for your report:
- In CEAS_08, label `1` covers unwanted/fraudulent emails collected for the 2008 CEAS spam challenge, so the model learns "phishing/spam-like vs legitimate" patterns from 2008-era emails. State this as a limitation.
- Only `subject`, `body`, `label` are required. `sender` is optional. Other column names (`Email Text`, `Email Type`, `text_combined`) are also auto-detected, so the Kaggle "Phishing_Email.csv" file works too.

Alternatives: the Kaggle dataset "Phishing Email Dataset" by Naser Abdullah Alam (<https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset>; Kaggle needs a free account).

The code removes missing values, empty emails and duplicate emails automatically (before splitting, to avoid leakage).

## 4. Setup on Windows (VS Code)

Open **VS Code → Terminal → New Terminal** (PowerShell) for each command block.

**Step 1: Create and open the project folder.** Skip this if you already copied the project folder.

```powershell
cd $HOME\Documents
mkdir phishing-email-detector
cd phishing-email-detector
code .
```

If you received this project as a folder or zip, extract it and use **File → Open Folder** in VS Code. Create the folders `data`, `models`, `results` if they are missing.

**Step 2: Create a virtual environment**

```powershell
python -m venv venv
```

**Step 3: Activate it** (you should see `(venv)` at the start of the prompt)

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks it ("running scripts is disabled"), run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

(Or use Command Prompt: `venv\Scripts\activate.bat`.)

**Step 4: Install the libraries**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Step 5: Download and place the dataset**

1. Open <https://zenodo.org/records/8339691> in your browser.
2. Download **CEAS_08.csv**.
3. Move and rename it to `data\emails.csv`:

```powershell
Move-Item "$env:USERPROFILE\Downloads\CEAS_08.csv" .\data\emails.csv
```

4. Check the first line (column names):

```powershell
Get-Content .\data\emails.csv -TotalCount 1
```

You should see something like `sender,receiver,date,subject,body,label,urls`.

**Step 6 & 7: Train, evaluate, and generate the model + confusion matrix**

```powershell
python train_model.py
```

This can take a few minutes. If your laptop is slow or runs out of memory, test first with a smaller sample: `python train_model.py --max-rows 20000`.
At the end it prints your real accuracy, precision, recall and F1, and creates `models\phishing_model.joblib`, `results\confusion_matrix.png`, `results\metrics.json`.

**Step 8 & 9: Launch the app and open it**

```powershell
streamlit run app.py
```

Your browser opens automatically at <http://localhost:8501>. If not, copy that address into your browser. Press `Ctrl + C` in the terminal to stop the app.

## 5. Testing with example emails

Open **Analyze Email** and paste these. The model gives a *probabilistic estimate*, not a guarantee, so a result can differ from what is "expected" below. Differences are worth discussing in your viva.

| # | Type | Sender | Subject | Body | Likely behaviour |
|---|---|---|---|---|---|
| 1 | Obvious phishing | `security@paypa1-support.com` | `URGENT: Your account has been suspended` | `Dear customer, we detected unusual activity. Verify your password and credit card details immediately or your account will be closed. Click here: http://paypa1-support.com/verify` | Likely **Phishing**; indicators: urgent, verify, password, account, credit card, http link, sender mentions brand but domain differs |
| 2 | Suspicious URL | `winner@mail-promo.net` | `You have won a free gift card!` | `Congratulations! Claim your prize now: http://bit.ly/3xYz9Ab or http://192.168.10.5/claim` | URL table should flag the **URL shortener** and **IP address** (this part is rule-based and reliable); the model will probably say Phishing |
| 3 | Legitimate-looking | `priya.sharma@college.edu` | `Notes from today's database lecture` | `Hi team, I have attached the notes from today's lecture. Please read chapter 4 before Friday's lab. Thanks, Priya` | Likely **Safe**; no URLs, no suspicious words |
| 4 | Urgent language | `hr@company-example.com` | `Action required immediately` | `Your password expires today. Act now and update your login details or lose access to your account.` | Several warning words are detected; the model may lean Phishing, but this is a case where the model could be uncertain |
| 5 | No URL | `friend@gmail.com` | `Congratulations, you won a lottery prize` | `You are the winner of our lottery. Send your bank account number and social security number to claim your money.` | No URLs found (the app should say so). Prediction depends on the words only |
| 6 | Empty input | (leave blank) | (blank) | (blank) | Error message: "Please enter an email subject or body" |

Also test error handling: temporarily rename `models\phishing_model.joblib` and reload the app. You should see a clear message telling you to run `train_model.py`.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Install Python from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"** during installation. Close and reopen VS Code. Also try `py --version` and use `py -m venv venv`. |
| `ModuleNotFoundError: No module named 'sklearn'` (or pandas, streamlit, ...) | The virtual environment is not active or libraries are missing. Activate it (Step 3) and run `pip install -r requirements.txt`. In VS Code press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one inside `venv`. |
| `Dataset not found at ...` | The file must be exactly `data\emails.csv` inside the project folder. Run `dir data` to check. Make sure you ran commands from the project folder. |
| `The CSV has no 'body' column` | Check the header line (Step 5, item 4). Your file needs email text and a label column. |
| `Unknown label values in the CSV` | The label column must be `0/1` or names like `Phishing Email`/`Safe Email`. |
| App says "model file was not found" | Run `python train_model.py` first. It must finish with "Done!". |
| `MemoryError` or very slow training | Use `python train_model.py --max-rows 20000`. |
| "model could not be loaded" after upgrading libraries | Delete `models\phishing_model.joblib` and retrain. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| `streamlit` not recognized | Activate the virtual environment, or run `python -m streamlit run app.py`. |

## 7. Important limitations

- URL and keyword indicators are **warning signs, not proof**.
- The model only knows patterns from its training dataset; new phishing styles can fool it.
- Model score is not a guarantee. Logistic Regression scores are the model's own estimates and are not perfectly calibrated.
- Both kinds of error exist: false positives (safe flagged) and false negatives (phishing missed).
- The app does not check sender authentication (SPF/DKIM/DMARC), attachments, or whether a website is live.

## 8. References

- Champa, A. I., Rabbi, M. F., & Zibran, M. F. (2024). *Curated datasets and feature analysis for phishing email detection with machine learning.* IEEE ICMI. Dataset: <https://zenodo.org/records/8339691>
- Champa, A. I., Rabbi, M. F., & Zibran, M. F. (2024). *Why phishing emails escape detection: A closer look at the failure points.* ISDFS.
- Al-Subaiey, A., et al. (2024). *Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection.* arXiv:2405.11619 (cited by the Kaggle compilation).
- Scikit-learn documentation: <https://scikit-learn.org/stable/>
- Pandas: <https://pandas.pydata.org/docs/> · NumPy: <https://numpy.org/doc/> · Matplotlib: <https://matplotlib.org/stable/> · Seaborn: <https://seaborn.pydata.org/> · Streamlit: <https://docs.streamlit.io/>
