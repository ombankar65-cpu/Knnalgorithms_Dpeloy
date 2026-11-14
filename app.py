import os
import pickle
import warnings

import pandas as pd
from flask import Flask, render_template_string, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (must sit next to app.py in the repo)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knnmodel.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
# ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
#  'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

# Label encodings (alphabetical, identical to sklearn LabelEncoder).
SEX = {"F": 0, "M": 1}
CHEST_PAIN = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
RESTING_ECG = {"LVH": 0, "Normal": 1, "ST": 2}
EXERCISE_ANGINA = {"N": 0, "Y": 1}
ST_SLOPE = {"Down": 0, "Flat": 1, "Up": 2}

# (min, max) sanity limits for numeric inputs
LIMITS = {
    "Age": (1, 120),
    "RestingBP": (0, 250),
    "Cholesterol": (0, 700),
    "MaxHR": (40, 230),
    "Oldpeak": (-3.0, 7.0),
}

# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HeartCheck · Heart disease risk screening</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#10303a;
    --ink-soft:#4b6a72;
    --bg:#eaf3f1;
    --card:#ffffff;
    --line:#d3e3e0;
    --teal:#0b7a75;
    --teal-dark:#075752;
    --teal-tint:#dff0ed;
    --rose:#c93c53;
    --rose-tint:#fbe6ea;
    --green:#1f8a5b;
    --green-tint:#def3e8;
    --radius:14px;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{
    margin:0;font-family:'Manrope',system-ui,sans-serif;color:var(--ink);
    background:var(--bg);line-height:1.55;
  }
  header.top{
    background:var(--teal-dark);color:#fff;padding:44px 20px 96px;
  }
  .wrap{max-width:1080px;margin:0 auto}
  header.top h1{
    font-family:'Fraunces',serif;font-weight:700;font-size:clamp(2rem,4.5vw,3.1rem);
    margin:0 0 10px;letter-spacing:-.01em;
  }
  header.top p{margin:0;max-width:56ch;color:#c9e6e2;font-size:1.05rem}
  .brand{display:flex;align-items:center;gap:10px;margin-bottom:26px;font-weight:700}
  .brand svg{width:30px;height:30px}

  main{margin-top:-64px;padding:0 20px 60px}
  .grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:22px;align-items:start}
  @media(max-width:880px){.grid{grid-template-columns:1fr}}

  .card{
    background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:28px;box-shadow:0 8px 28px rgba(16,48,58,.07);
  }
  fieldset{border:0;margin:0 0 26px;padding:0}
  legend{
    font-family:'Fraunces',serif;font-weight:600;font-size:1.25rem;
    padding:0 0 12px;margin-bottom:4px;border-bottom:2px solid var(--teal-tint);width:100%;
  }
  .row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px 18px;margin-top:16px}
  @media(max-width:560px){.row{grid-template-columns:1fr}}
  label{display:block;font-weight:600;font-size:.92rem;margin-bottom:6px}
  .hint{display:block;font-weight:400;color:var(--ink-soft);font-size:.8rem;margin-top:2px}
  input,select{
    width:100%;padding:11px 12px;font:inherit;color:var(--ink);background:#f7fbfa;
    border:1.5px solid var(--line);border-radius:10px;transition:border-color .15s, box-shadow .15s;
  }
  input:focus,select:focus{outline:none;border-color:var(--teal);box-shadow:0 0 0 3px rgba(11,122,117,.2);background:#fff}
  button.go{
    width:100%;padding:15px;font:inherit;font-weight:700;font-size:1.05rem;color:#fff;
    background:var(--teal);border:0;border-radius:12px;cursor:pointer;transition:background .15s,transform .05s;
  }
  button.go:hover{background:var(--teal-dark)}
  button.go:active{transform:translateY(1px)}
  button.go:focus-visible{outline:3px solid #86d3cd;outline-offset:2px}

  .error{background:var(--rose-tint);color:#8e2236;border:1px solid #f0b8c2;
    padding:12px 14px;border-radius:10px;margin-bottom:20px;font-weight:600;font-size:.92rem}

  aside .card{position:sticky;top:20px}
  .placeholder h2,.result h2{font-family:'Fraunces',serif;margin:0 0 8px;font-size:1.4rem}
  .placeholder p{color:var(--ink-soft);margin:0 0 10px}
  .placeholder ul{margin:12px 0 0;padding-left:18px;color:var(--ink-soft);font-size:.92rem}

  .result.high{border-left:8px solid var(--rose)}
  .result.low{border-left:8px solid var(--green)}
  .badge{display:inline-block;padding:5px 12px;border-radius:999px;font-weight:700;font-size:.85rem;margin-bottom:12px}
  .high .badge{background:var(--rose-tint);color:var(--rose)}
  .low .badge{background:var(--green-tint);color:var(--green)}
  .bar{height:12px;border-radius:999px;background:#e5eeec;overflow:hidden;margin:14px 0 6px}
  .bar span{display:block;height:100%;border-radius:999px}
  .high .bar span{background:var(--rose)}
  .low .bar span{background:var(--green)}
  .meter-label{display:flex;justify-content:space-between;font-size:.82rem;color:var(--ink-soft)}
  .advice{margin:16px 0 0;font-size:.95rem}
  .note{
    margin-top:22px;padding:14px 16px;background:var(--teal-tint);border-radius:10px;
    font-size:.82rem;color:var(--ink-soft)
  }
  footer{text-align:center;color:var(--ink-soft);font-size:.82rem;padding:0 20px 36px}
  @media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<header class="top">
  <div class="wrap">
    <div class="brand">
      <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/>
        <path d="M3 12h4l2-3 3 6 2-3h7" stroke="#86d3cd"/>
      </svg>
      HeartCheck
    </div>
    <h1>Check your heart disease risk</h1>
    <p>Enter your clinical details and a K-Nearest Neighbors model compares them with 734 past patient records to estimate risk.</p>
  </div>
</header>

<main>
  <div class="wrap grid">
    <section class="card">
      {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
      <form method="post" action="/" novalidate>
        <fieldset>
          <legend>About you</legend>
          <div class="row">
            <div>
              <label for="Age">Age <span class="hint">in years</span></label>
              <input type="number" id="Age" name="Age" min="1" max="120" step="1" required value="{{ v.get('Age','') }}">
            </div>
            <div>
              <label for="Sex">Sex</label>
              <select id="Sex" name="Sex" required>
                <option value="M" {{ 'selected' if v.get('Sex')=='M' }}>Male</option>
                <option value="F" {{ 'selected' if v.get('Sex')=='F' }}>Female</option>
              </select>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Resting measurements</legend>
          <div class="row">
            <div>
              <label for="RestingBP">Resting blood pressure <span class="hint">mm Hg</span></label>
              <input type="number" id="RestingBP" name="RestingBP" min="0" max="250" required value="{{ v.get('RestingBP','') }}">
            </div>
            <div>
              <label for="Cholesterol">Serum cholesterol <span class="hint">mg/dL</span></label>
              <input type="number" id="Cholesterol" name="Cholesterol" min="0" max="700" required value="{{ v.get('Cholesterol','') }}">
            </div>
            <div>
              <label for="FastingBS">Fasting blood sugar</label>
              <select id="FastingBS" name="FastingBS" required>
                <option value="0" {{ 'selected' if v.get('FastingBS')=='0' }}>120 mg/dL or lower</option>
                <option value="1" {{ 'selected' if v.get('FastingBS')=='1' }}>Above 120 mg/dL</option>
              </select>
            </div>
            <div>
              <label for="RestingECG">Resting ECG result</label>
              <select id="RestingECG" name="RestingECG" required>
                <option value="Normal" {{ 'selected' if v.get('RestingECG')=='Normal' }}>Normal</option>
                <option value="ST" {{ 'selected' if v.get('RestingECG')=='ST' }}>ST-T wave abnormality</option>
                <option value="LVH" {{ 'selected' if v.get('RestingECG')=='LVH' }}>Left ventricular hypertrophy</option>
              </select>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Symptoms and exercise test</legend>
          <div class="row">
            <div>
              <label for="ChestPainType">Chest pain type</label>
              <select id="ChestPainType" name="ChestPainType" required>
                <option value="ASY" {{ 'selected' if v.get('ChestPainType')=='ASY' }}>Asymptomatic</option>
                <option value="ATA" {{ 'selected' if v.get('ChestPainType')=='ATA' }}>Atypical angina</option>
                <option value="NAP" {{ 'selected' if v.get('ChestPainType')=='NAP' }}>Non-anginal pain</option>
                <option value="TA" {{ 'selected' if v.get('ChestPainType')=='TA' }}>Typical angina</option>
              </select>
            </div>
            <div>
              <label for="MaxHR">Maximum heart rate <span class="hint">beats per minute</span></label>
              <input type="number" id="MaxHR" name="MaxHR" min="40" max="230" required value="{{ v.get('MaxHR','') }}">
            </div>
            <div>
              <label for="ExerciseAngina">Exercise-induced angina</label>
              <select id="ExerciseAngina" name="ExerciseAngina" required>
                <option value="N" {{ 'selected' if v.get('ExerciseAngina')=='N' }}>No</option>
                <option value="Y" {{ 'selected' if v.get('ExerciseAngina')=='Y' }}>Yes</option>
              </select>
            </div>
            <div>
              <label for="Oldpeak">ST depression (Oldpeak) <span class="hint">exercise vs. rest</span></label>
              <input type="number" id="Oldpeak" name="Oldpeak" min="-3" max="7" step="0.1" required value="{{ v.get('Oldpeak','') }}">
            </div>
            <div>
              <label for="ST_Slope">Slope of peak-exercise ST segment</label>
              <select id="ST_Slope" name="ST_Slope" required>
                <option value="Up" {{ 'selected' if v.get('ST_Slope')=='Up' }}>Upsloping</option>
                <option value="Flat" {{ 'selected' if v.get('ST_Slope')=='Flat' }}>Flat</option>
                <option value="Down" {{ 'selected' if v.get('ST_Slope')=='Down' }}>Downsloping</option>
              </select>
            </div>
          </div>
        </fieldset>

        <button class="go" type="submit">Get my risk estimate</button>
      </form>
    </section>

    <aside>
      <div class="card" id="result">
        {% if result %}
          <div class="result {{ result.level }}">
            <span class="badge">{{ result.label }}</span>
            <h2>{{ result.headline }}</h2>
            <div class="bar" role="img" aria-label="Estimated risk {{ result.percent }} percent">
              <span style="width: {{ result.percent }}%"></span>
            </div>
            <div class="meter-label"><span>Lower</span><span>Estimated risk: {{ result.percent }}%</span><span>Higher</span></div>
            <p class="advice">{{ result.advice }}</p>
          </div>
        {% else %}
          <div class="placeholder">
            <h2>Your result will appear here</h2>
            <p>Fill in all eleven fields and select <strong>Get my risk estimate</strong>.</p>
            <ul>
              <li>Use values from a recent medical report</li>
              <li>Nothing you enter is stored</li>
            </ul>
          </div>
        {% endif %}
        <div class="note">
          This is a student machine-learning project, not a medical device. It cannot diagnose
          any condition. Please talk to a doctor about your heart health.
        </div>
      </div>
    </aside>
  </div>
</main>

<footer>HeartCheck · built with Flask and scikit-learn</footer>
{% if result or error %}
<script>document.getElementById('result').scrollIntoView({block:'nearest'});</script>
{% endif %}
</body>
</html>
"""


def parse_form(form):
    """Validate the posted form and return a one-row DataFrame in model order."""
    row = {}
    try:
        for col in ("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"):
            raw = (form.get(col) or "").strip()
            if raw == "":
                raise ValueError(f"Please fill in {col}.")
            val = float(raw)
            lo, hi = LIMITS[col]
            if not lo <= val <= hi:
                raise ValueError(f"{col} should be between {lo} and {hi}.")
            row[col] = val

        row["Sex"] = SEX[form.get("Sex", "")]
        row["ChestPainType"] = CHEST_PAIN[form.get("ChestPainType", "")]
        row["RestingECG"] = RESTING_ECG[form.get("RestingECG", "")]
        row["ExerciseAngina"] = EXERCISE_ANGINA[form.get("ExerciseAngina", "")]
        row["ST_Slope"] = ST_SLOPE[form.get("ST_Slope", "")]
        row["FastingBS"] = int(form.get("FastingBS", ""))
        if row["FastingBS"] not in (0, 1):
            raise KeyError
    except KeyError:
        raise ValueError("One of the dropdown choices is not valid. Please reselect it.")
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("could not convert") or msg.startswith("invalid literal"):
            msg = "Please enter numbers only in the numeric fields."
        raise ValueError(msg)

    return pd.DataFrame([row], columns=FEATURES)


def build_result(X):
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    risk = float(proba[classes.index(1)]) if 1 in classes else float(pred)
    percent = round(risk * 100)

    if pred == 1:
        return {
            "level": "high",
            "label": "Higher risk",
            "headline": "Your details match patients with heart disease",
            "percent": percent,
            "advice": "Please book a check-up with a doctor or cardiologist soon and share these values with them.",
        }
    return {
        "level": "low",
        "label": "Lower risk",
        "headline": "Your details match patients without heart disease",
        "percent": percent,
        "advice": "Keep up regular exercise, a balanced diet and routine check-ups.",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    values, result, error = {}, None, None
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            X = parse_form(request.form)
            result = build_result(X)
        except ValueError as exc:
            error = str(exc)
    return render_template_string(PAGE, v=values, result=result, error=error)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
