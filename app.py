import streamlit as st
import pickle
import numpy as np
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutismSense — Screening Tool",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("rf_smote_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    color: #e8eaf0;
  }

  /* ── Hide default streamlit elements ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── Hero section ── */
  .hero-wrap {
    background: linear-gradient(135deg, rgba(92,124,250,0.15) 0%, rgba(168,85,247,0.1) 100%);
    border: 1px solid rgba(92,124,250,0.25);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(168,85,247,0.08);
    filter: blur(40px);
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a855f7, #5c7cfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
  }
  .hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    max-width: 580px;
    line-height: 1.6;
    margin: 0;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(92,124,250,0.15);
    border: 1px solid rgba(92,124,250,0.35);
    color: #93c5fd;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
  }

  /* ── Section card ── */
  .section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
  }
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title span {
    background: rgba(92,124,250,0.15);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #93c5fd;
  }

  /* ── Streamlit widget overrides ── */
  .stSlider > div > div > div { background: rgba(92,124,250,0.3) !important; }
  .stSlider > div > div > div > div { background: #5c7cfa !important; }

  div[data-testid="stSelectbox"] > div,
  div[data-testid="stNumberInput"] > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
  }

  .stRadio > div { gap: 0.5rem; }
  .stRadio label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 6px 14px;
    color: #cbd5e1;
    cursor: pointer;
    transition: all 0.2s;
  }

  /* ── Predict button ── */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #5c7cfa, #a855f7) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.8rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(92,124,250,0.35) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(92,124,250,0.5) !important;
  }

  /* ── Result cards ── */
  .result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: slideUp 0.5s ease-out;
  }
  .result-negative {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(22,163,74,0.06));
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: slideUp 0.5s ease-out;
  }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
  .result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0.3rem 0;
  }
  .result-pos-title { color: #f87171; }
  .result-neg-title { color: #4ade80; }
  .result-desc { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem; }

  /* ── Confidence bar ── */
  .conf-wrap { margin-top: 1.5rem; }
  .conf-label { font-size: 0.8rem; color: #64748b; margin-bottom: 6px; font-weight: 500; }
  .conf-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
  }
  .conf-bar-fill-pos {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #f87171, #ef4444);
    transition: width 1s ease;
  }
  .conf-bar-fill-neg {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #4ade80, #22c55e);
    transition: width 1s ease;
  }
  .conf-pct {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    margin-top: 0.6rem;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 0.8rem;
    color: #fbbf24;
    line-height: 1.6;
    margin-top: 1rem;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(15,22,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
  }
  [data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

  /* ── Score button grid ── */
  .score-btn-group { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
  .score-q { font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 4px; }

  /* ── Info pills ── */
  .pill {
    display: inline-block;
    background: rgba(92,124,250,0.1);
    border: 1px solid rgba(92,124,250,0.2);
    color: #93c5fd;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-size:2.5rem'>🧩</div>
      <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                  background:linear-gradient(90deg,#a855f7,#5c7cfa);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        AutismSense
      </div>
      <div style='font-size:0.75rem; color:#475569; margin-top:4px;'>AI Screening Tool</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    <div style='font-size:0.85rem; color:#64748b; line-height:1.7;'>
    This tool uses a <strong style='color:#93c5fd'>Random Forest</strong> model 
    trained with <strong style='color:#93c5fd'>SMOTE</strong> oversampling to 
    screen for Autism Spectrum Disorder (ASD) based on behavioral Q&A and 
    demographic inputs.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Model Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Features", "29")
        st.metric("Classes", "2")
    with col2:
        st.metric("Algorithm", "RF")
        st.metric("Sampling", "SMOTE")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#334155; line-height:1.6;'>
    ⚠️ This is a <strong>screening aid only</strong>. 
    Not a clinical diagnosis. Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🧠 Autism Spectrum Disorder · AI Screening</div>
  <div class="hero-title">AutismSense</div>
  <p class="hero-sub">
    Answer a short set of behavioral questions and provide some basic information. 
    Our AI model will analyze the responses and give you an early screening result 
    — quick, private, and easy to understand.
  </p>
</div>
""", unsafe_allow_html=True)


# ── AQ-10 Questions ───────────────────────────────────────────────────────────
AQ_QUESTIONS = [
    "I often notice small sounds when others do not",
    "I usually concentrate more on the whole picture, rather than the small details",
    "I find it easy to do more than one thing at once",
    "If there is an interruption, I can switch back to what I was doing very quickly",
    "I find it easy to 'read between the lines' when someone is talking to me",
    "I know how to tell if someone listening to me is getting bored",
    "When I'm reading a story I find it difficult to work out the characters' intentions",
    "I like to collect information about categories of things",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face",
    "I find it difficult to work out people's intentions",
]

# ── Section 1: AQ-10 Behavioral Questions ─────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">🧪 AQ-10 Behavioral Assessment <span>10 Questions</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='color:#64748b; font-size:0.85rem; margin-bottom:1.5rem;'>"
    "For each statement below, select <strong style='color:#93c5fd'>1</strong> if it applies to you, "
    "or <strong style='color:#94a3b8'>0</strong> if it does not.</p>",
    unsafe_allow_html=True
)

scores = []
cols_q = st.columns(2)
for i, question in enumerate(AQ_QUESTIONS):
    with cols_q[i % 2]:
        st.markdown(f"<p class='score-q'><strong style='color:#5c7cfa'>Q{i+1}.</strong> {question}</p>", unsafe_allow_html=True)
        val = st.radio(
            label=f"q{i+1}",
            options=[0, 1],
            horizontal=True,
            label_visibility="collapsed",
            key=f"A{i+1}_Score"
        )
        scores.append(val)
        st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)


st.markdown("<div style='margin: 1.5rem 0 0.5rem'></div>", unsafe_allow_html=True)

# ── Section 2: Personal Information ───────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">👤 Personal Information <span>Demographics</span></div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    ethnicity = st.selectbox("Ethnicity", [
        "White-European", "Black", "Hispanic", "Latino",
        "Middle Eastern", "Others", "Pasifika", "South Asian", "Turkish"
    ])
    jaundice = st.selectbox("Born with Jaundice?", ["No", "Yes"])

with col3:
    autism_family = st.selectbox("Family member with Autism?", ["No", "Yes"])
    used_app = st.selectbox("Used screening app before?", ["No", "Yes"])

result_score = st.number_input(
    "AQ-10 Result Score (sum of your answers above)",
    min_value=0, max_value=10,
    value=sum(scores),
    help="This is auto-calculated from your answers above, but you can override it."
)

st.markdown("<div style='margin: 1.5rem 0 0.5rem'></div>", unsafe_allow_html=True)

# ── Section 3: Relation ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">🔗 Who is filling this form? <span>Relation</span></div>
</div>
""", unsafe_allow_html=True)

relation = st.radio(
    "Relation to the person being screened",
    ["Self", "Parent", "Relative", "Others"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='margin: 2rem 0 1rem'></div>", unsafe_allow_html=True)


# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Run Autism Screening", use_container_width=True)

if predict_clicked:
    with st.spinner("Analyzing responses..."):
        time.sleep(1.2)

    # ── Build feature vector ──────────────────────────────────────────────────
    # Ethnicity one-hot
    eth_cols = ['Black', 'Hispanic', 'Latino', 'Middle Eastern ',
                'Others', 'Pasifika', 'South Asian', 'Turkish', 'White-European']
    eth_map = {e.strip(): e for e in eth_cols}
    eth_vec = [1 if ethnicity.strip() in e else 0 for e in eth_cols]

    # Gender
    gender_m = 1 if gender == "Male" else 0

    # Binary flags
    jaundice_yes    = 1 if jaundice == "Yes" else 0
    autism_yes      = 1 if autism_family == "Yes" else 0
    used_app_yes    = 1 if used_app == "Yes" else 0

    # Relation one-hot: Others, Parent, Relative, Self
    rel_others   = 1 if relation == "Others"   else 0
    rel_parent   = 1 if relation == "Parent"   else 0
    rel_relative = 1 if relation == "Relative" else 0
    rel_self     = 1 if relation == "Self"     else 0

    features = (
        scores +                          # A1–A10
        [age, result_score] +             # age, result
        eth_vec +                         # ethnicity (9 cols)
        [gender_m, jaundice_yes,          # gender, jaundice
         autism_yes, used_app_yes,        # austim, used_app_before
         rel_others, rel_parent,          # relation
         rel_relative, rel_self]
    )

    X = np.array(features).reshape(1, -1)

    prediction   = model.predict(X)[0]
    proba        = model.predict_proba(X)[0]
    confidence   = round(proba[prediction] * 100, 1)

    st.markdown("<div style='margin: 1rem 0'></div>", unsafe_allow_html=True)

    # ── Result display ────────────────────────────────────────────────────────
    if prediction == 1:
        bar_color = "conf-bar-fill-pos"
        st.markdown(f"""
        <div class="result-positive">
          <div class="result-emoji">⚠️</div>
          <div class="result-title result-pos-title">Autism Traits Detected</div>
          <div class="result-desc">
            The model has identified behavioral patterns that may be 
            associated with Autism Spectrum Disorder. This is not a diagnosis — 
            please consult a certified psychologist or neurologist for a full evaluation.
          </div>
          <div class="conf-wrap">
            <div class="conf-label">Model Confidence</div>
            <div class="conf-bar-bg">
              <div class="conf-bar-fill-pos" style="width:{confidence}%"></div>
            </div>
            <div class="conf-pct" style="color:#f87171">{confidence}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
          <div class="result-emoji">✅</div>
          <div class="result-title result-neg-title">No Strong Indicators Found</div>
          <div class="result-desc">
            Based on the responses provided, the model did not detect strong 
            indicators of Autism Spectrum Disorder. If you still have concerns, 
            a professional assessment is always recommended.
          </div>
          <div class="conf-wrap">
            <div class="conf-label">Model Confidence</div>
            <div class="conf-bar-bg">
              <div class="conf-bar-fill-neg" style="width:{confidence}%"></div>
            </div>
            <div class="conf-pct" style="color:#4ade80">{confidence}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown("<div style='margin:1.5rem 0 0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; font-size:0.8rem; font-weight:600; letter-spacing:1px; text-transform:uppercase;'>Probability Breakdown</p>", unsafe_allow_html=True)
    pb_col1, pb_col2 = st.columns(2)
    with pb_col1:
        st.metric("🟢 No ASD", f"{round(proba[0]*100, 1)}%")
    with pb_col2:
        st.metric("🔴 ASD Detected", f"{round(proba[1]*100, 1)}%")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Important:</strong> This tool is for <strong>educational and screening purposes only</strong>. 
      It is not a substitute for a clinical diagnosis by a licensed medical professional. 
      If you have concerns about yourself or a loved one, please seek professional evaluation.
    </div>
    """, unsafe_allow_html=True)