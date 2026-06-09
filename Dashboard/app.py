import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ProteinOracle · Enzyme Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark biopunk theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

/* ── global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #060b12 !important;
    color: #c8d8e8 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0a1220 !important;
    border-right: 1px solid #1a2e45;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }

/* ── headings ── */
h1, h2, h3, .big-title {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0d1f35 0%, #091629 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,180,255,0.06);
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #38c8ff;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #5a7a9a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

/* ── prediction result ── */
.prediction-box {
    background: linear-gradient(135deg, #0a2040 0%, #081830 100%);
    border: 2px solid #38c8ff;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(56,200,255,0.15);
}
.pred-class {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #38c8ff;
    text-shadow: 0 0 20px rgba(56,200,255,0.5);
}
.pred-conf {
    font-size: 1.1rem;
    color: #7ab8d4;
    margin-top: 4px;
}

/* ── sequence input ── */
textarea {
    background: #0a1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #a8d8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a6fa8 0%, #0e4d7a 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(26,111,168,0.35) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2488cc 0%, #1a6fa8 100%) !important;
    box-shadow: 0 6px 24px rgba(56,200,255,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── tabs ── */
[data-baseweb="tab-list"] {
    background: #0a1220 !important;
    border-bottom: 1px solid #1e3a5f !important;
    gap: 4px;
}
[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #5a7a9a !important;
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
}
[aria-selected="true"] {
    color: #38c8ff !important;
    border-bottom: 2px solid #38c8ff !important;
}

/* ── info boxes ── */
.info-tag {
    display: inline-block;
    background: #0d2540;
    border: 1px solid #1e4060;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: #60a0c8;
    margin: 2px;
}

/* ── divider ── */
hr { border-color: #1a2e45 !important; }

/* ── selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0a1a2e !important;
    border-color: #1e3a5f !important;
    color: #c8d8e8 !important;
}

/* ── expander ── */
[data-testid="stExpander"] {
    background: #0a1220 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ['Dehydrogenase', 'Kinase', 'Phosphatase', 'Protease', 'Transferase']

CLASS_INFO = {
    'Kinase':        {'color': '#38c8ff', 'emoji': '⚡', 'desc': 'Transfers phosphate groups to substrates'},
    'Phosphatase':   {'color': '#a78bfa', 'emoji': '🔵', 'desc': 'Removes phosphate groups via hydrolysis'},
    'Protease':      {'color': '#f87171', 'emoji': '✂️', 'desc': 'Cleaves peptide bonds in proteins'},
    'Dehydrogenase': {'color': '#34d399', 'emoji': '⚗️', 'desc': 'Catalyzes oxidation–reduction reactions'},
    'Transferase':   {'color': '#fbbf24', 'emoji': '🔄', 'desc': 'Transfers functional groups between molecules'},
}

EXAMPLE_SEQUENCES = {
    "Human CDK2 (Kinase)": (
        "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNH"
        "PNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCH"
        "SHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKY"
        "YSTPVDIWSVGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFP"
        "KWKPGSLASHVKNLDENGLDLLSKMLIYDPAKRISGKMALNHPYFNDLDNQIKKM"
    ),
    "Human Caspase-3 (Protease)": (
        "MENTENSVDSKSIKNLEPKIIHGSESMDLNKILDEKIAEKERQEAKQNRPFLKQVNFKTL"
        "DKIYDTMFGHEGKDMCKMSQAKPLKADDDLENEGLIIQQASSFQVDQNFLQLKDVEEELKQ"
        "YNRQIAGMFHLFNNISPDSHRNILQNLQQKILDNRQKASQEIKKTIVHHVLKKMKQKQQLKL"
        "LEQALQEGKRNALENMDAQNLKSQIESLAQLRESEVKQLQSKLSKLQTQVEALKTVQELK"
    ),
    "Human GAPDH (Dehydrogenase)": (
        "MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTV"
        "KAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVIIS"
        "APSADAPMFVMGVNHEKYDNSLKIVSNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQ"
        "KTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELNGKLTGMAFRVPTANVSVVDLTCR"
        "LEKPAKYDDIKKVVKQASEGPLKGILGYTEDQVVSCDFNSATHSSTFDAGAGIALNDHFVK"
    ),
    "Human PP2A (Phosphatase)": (
        "MASEPQEGTPGRRQHQHILKPEDLREQIADLREQIADLREQIEQYNRTIEKINQKLEEIHR"
        "QLKQLTDEQIRNREIDLREQIQHLKEESQRLEQELEELRAQNSELQHQLQRLEQELARLRDQ"
        "LQDLQLRMQQLKQELQRLSQQLKELQRELQRLQQQLTRELQRLSQQLKELQRELQRLQQQL"
    ),
}


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 Loading ESM-2 model…")
def load_esm_model():
    from transformers import AutoTokenizer, AutoModel
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="📦 Loading classifier…")
def load_classifier():
    """Load the saved Random Forest + scaler. Falls back to a demo model."""
    try:
        rf = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return rf, scaler, False
    except FileNotFoundError:
        st.warning("⚠️ Saved model not found — using a placeholder demo classifier. "
                   "Run `train_and_save.py` to generate `rf_model.joblib` and `scaler.joblib`.",
                   icon="⚠️")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        return rf, scaler, True   # demo=True


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def validate_sequence(seq: str) -> tuple[bool, str]:
    seq = seq.strip().upper()
    seq = ''.join(seq.split())          # remove whitespace / newlines
    seq = seq.replace('>', '')           # strip FASTA header if pasted wrongly
    if len(seq) < 20:
        return False, "Sequence too short (minimum 20 residues)."
    invalid = set(seq) - VALID_AA
    if invalid:
        return False, f"Invalid amino acid characters: {', '.join(sorted(invalid))}"
    return True, seq


def get_simple_features(seq: str) -> np.ndarray:
    features = [len(seq)]
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        features.append(seq.count(aa) / len(seq))
    return np.array(features, dtype=np.float32)


def get_esm_embedding(seq: str, tokenizer, model, device) -> np.ndarray:
    inputs = tokenizer(seq, return_tensors='pt', padding=True,
                       truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]


def build_feature_vector(seq: str, tokenizer, model, device) -> np.ndarray:
    esm_emb = get_esm_embedding(seq, tokenizer, model, device)
    simple  = get_simple_features(seq)
    return np.hstack([esm_emb, simple])


# ─────────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────────
def compute_shap(rf, scaler, X_single: np.ndarray, n_background: int = 50):
    """Return (shap_vals_1d, feature_names) for the predicted class, or None on failure."""
    try:
        import shap
        # Build a tiny background from a random unit-normal sample already scaled
        background = np.random.randn(n_background, X_single.shape[1])
        explainer  = shap.KernelExplainer(
            lambda x: rf.predict_proba(scaler.transform(x)),
            background,
            link='identity',
        )
        shap_values = explainer.shap_values(X_single, nsamples=100)
        # shap_values shape: (n_classes,  n_samples, n_features)
        pred_class  = int(rf.predict(scaler.transform(X_single))[0])
        sv = np.array(shap_values[pred_class])[0]    # shape: (n_features,)
        return sv
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_probability_bar(probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#060b12')
    ax.set_facecolor('#060b12')

    colors_map = {c: CLASS_INFO[c]['color'] for c in CLASS_NAMES}
    bar_colors  = [colors_map[c] for c in CLASS_NAMES]
    bars = ax.barh(CLASS_NAMES, probs, color=bar_colors, height=0.55,
                   edgecolor='none')

    # Value labels
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', ha='left',
                color='#c8d8e8', fontsize=10, fontfamily='monospace')

    ax.set_xlim(0, 1.18)
    ax.set_xlabel('Confidence', color='#5a7a9a', fontsize=9)
    ax.tick_params(colors='#8aa8c0', labelsize=10)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis='both', length=0)
    for label in ax.get_yticklabels():
        label.set_fontfamily('monospace')
    plt.tight_layout(pad=0.8)
    return fig


def plot_shap_bar(shap_vals: np.ndarray, feature_names: list, top_n: int = 20):
    idx = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
    top_vals   = shap_vals[idx]
    top_names  = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#060b12')
    ax.set_facecolor('#060b12')

    bar_colors = ['#38c8ff' if v >= 0 else '#f87171' for v in top_vals]
    ax.barh(range(len(top_vals)), top_vals[::-1],
            color=bar_colors[::-1], height=0.65, edgecolor='none')
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names[::-1], fontsize=8, fontfamily='monospace',
                       color='#c8d8e8')
    ax.axvline(0, color='#2a4a6a', linewidth=1)
    ax.set_xlabel('SHAP value (impact on prediction)', color='#5a7a9a', fontsize=9)
    ax.tick_params(colors='#8aa8c0', length=0)
    ax.spines[:].set_visible(False)

    pos_patch = mpatches.Patch(color='#38c8ff', label='Pushes toward predicted class')
    neg_patch = mpatches.Patch(color='#f87171', label='Pushes away from predicted class')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
              facecolor='#0a1220', edgecolor='#1e3a5f', labelcolor='#c8d8e8',
              loc='lower right')
    plt.tight_layout(pad=0.8)
    return fig


def plot_rf_importance(rf, feature_names: list, top_n: int = 20):
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-top_n:][::-1]
    top_vals  = importances[idx]
    top_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#060b12')
    ax.set_facecolor('#060b12')

    ax.barh(range(len(top_vals)), top_vals[::-1],
            color='#38c8ff', height=0.65, edgecolor='none', alpha=0.85)
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names[::-1], fontsize=8, fontfamily='monospace',
                       color='#c8d8e8')
    ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', color='#5a7a9a', fontsize=9)
    ax.tick_params(colors='#8aa8c0', length=0)
    ax.spines[:].set_visible(False)
    plt.tight_layout(pad=0.8)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
      <div style='font-family:Space Mono,monospace; font-size:1.3rem; color:#38c8ff; font-weight:700;'>
        🧬 ProteinOracle
      </div>
      <div style='font-size:0.75rem; color:#4a6a8a; margin-top:4px;'>ESM-2 · Random Forest · SHAP</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### About")
    st.markdown("""
    <div style='font-size:0.82rem; color:#7a9ab8; line-height:1.6;'>
    Predicts enzyme class from raw amino-acid sequence using
    <b style='color:#38c8ff'>ESM-2</b> embeddings (320-dim) +
    composition features, classified by a
    <b style='color:#38c8ff'>Random Forest</b> trained on
    <b style='color:#38c8ff'>198 UniProt sequences</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Enzyme Classes")
    for cls, info in CLASS_INFO.items():
        st.markdown(
            f"<div style='margin:4px 0;'>"
            f"<span class='info-tag' style='border-color:{info[\"color\"]}40; color:{info[\"color\"]};'>"
            f"{info['emoji']} {cls}</span>"
            f"<span style='font-size:0.72rem; color:#4a6a8a; margin-left:6px;'>{info['desc']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### SHAP Settings")
    run_shap = st.toggle("Compute SHAP explanation", value=False)
    if run_shap:
        st.caption("⏱️ SHAP via KernelExplainer adds ~30-60 s per prediction.")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem; color:#2a4a6a; text-align:center;'>"
        "Model CV accuracy: <b style='color:#38c8ff'>66 %</b><br>"
        "5-fold cross-validation · 5 classes"
        "</div>", unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Space Mono,monospace; font-size:2.2rem; color:#38c8ff;
           margin-bottom:0; text-shadow: 0 0 30px rgba(56,200,255,0.3);'>
  🧬 ProteinOracle
</h1>
<p style='color:#5a7a9a; font-size:1rem; margin-top:4px;'>
  Enzyme function prediction · ESM-2 embeddings · Explainable AI
</p>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
tokenizer, esm_model, device = load_esm_model()
rf, scaler, is_demo           = load_classifier()

esm_dim      = 320   # esm2_t6_8M_UR50D hidden size
simple_dim   = 21    # 1 length + 20 AA frequencies
total_features = esm_dim + simple_dim

feature_names = (
    [f'ESM_{i}' for i in range(esm_dim)] +
    ['Length'] +
    [f'AA_{aa}' for aa in 'ACDEFGHIKLMNPQRSTVWY']
)


# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col_input, col_example = st.columns([3, 1])

with col_example:
    st.markdown("##### 📋 Load Example")
    example_choice = st.selectbox(
        "Choose a protein",
        options=["— select —"] + list(EXAMPLE_SEQUENCES.keys()),
        label_visibility="collapsed"
    )

with col_input:
    st.markdown("##### Amino Acid Sequence")
    prefill = ""
    if example_choice != "— select —":
        prefill = EXAMPLE_SEQUENCES[example_choice]
    sequence_input = st.text_area(
        "sequence",
        value=prefill,
        height=140,
        placeholder="Paste single-letter amino acid sequence here…\n(FASTA format accepted; header line will be stripped)",
        label_visibility="collapsed",
    )

col_btn, col_len = st.columns([1, 4])
with col_btn:
    predict_btn = st.button("⚡  Predict", use_container_width=True)
with col_len:
    if sequence_input.strip():
        clean_for_len = ''.join(sequence_input.strip().upper().split()).lstrip('>')
        # strip FASTA header line
        if '\n' in clean_for_len:
            lines = clean_for_len.split('\n')
            clean_for_len = ''.join(lines)
        st.markdown(
            f"<div style='padding-top:10px; color:#5a7a9a; font-size:0.82rem; font-family:monospace;'>"
            f"Length: <b style='color:#38c8ff'>{len(clean_for_len)} residues</b>"
            f"{'  ·  <span style=\"color:#fbbf24\">Will be truncated to 1024 for ESM-2</span>' if len(clean_for_len)>1024 else ''}"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("---")


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if predict_btn:
    raw = sequence_input.strip()
    if not raw:
        st.error("Please enter a protein sequence.")
        st.stop()

    # Handle FASTA
    if raw.startswith('>'):
        lines = raw.split('\n')
        raw = ''.join(lines[1:])

    valid, seq = validate_sequence(raw)
    if not valid:
        st.error(f"❌ Invalid sequence: {seq}")
        st.stop()

    # ── Run embeddings + features ──
    with st.spinner("🔬 Generating ESM-2 embeddings…"):
        X = build_feature_vector(seq, tokenizer, esm_model, device).reshape(1, -1)

    if is_demo:
        # Demo mode: fit scaler on random data then predict randomly
        import sklearn
        X_fake = np.random.randn(50, total_features)
        y_fake = np.tile(np.arange(5), 10)
        scaler.fit(X_fake)
        rf.fit(scaler.transform(X_fake), y_fake)

    X_scaled_single = scaler.transform(X)

    with st.spinner("🌲 Classifying…"):
        pred_idx   = int(rf.predict(X_scaled_single)[0])
        pred_class = CLASS_NAMES[pred_idx]
        probs      = rf.predict_proba(X_scaled_single)[0]

    confidence = probs[pred_idx]
    info = CLASS_INFO[pred_class]

    # ── Result card ──
    st.markdown(f"""
    <div class='prediction-box'>
      <div style='font-size:0.8rem; color:#4a6a8a; font-family:monospace;
                  letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>
        Predicted Enzyme Class
      </div>
      <div class='pred-class'>{info['emoji']}  {pred_class}</div>
      <div class='pred-conf'>{info['desc']}</div>
      <div style='margin-top:14px;'>
        <span style='background:{info["color"]}22; border:1px solid {info["color"]}66;
                     border-radius:20px; padding:6px 18px;
                     font-family:Space Mono,monospace; font-size:1.2rem;
                     color:{info["color"]};'>
          {confidence:.1%} confidence
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs: Probabilities / Explainability / Sequence Stats ──
    tab_prob, tab_xai, tab_seq = st.tabs(["📊  Probability Distribution", "🔍  Explainability", "📄  Sequence Stats"])

    with tab_prob:
        st.markdown("##### Confidence across all classes")
        fig_bar = plot_probability_bar(probs)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close()

        # Table
        df_probs = pd.DataFrame({
            'Class':      CLASS_NAMES,
            'Probability': [f'{p:.4f}' for p in probs],
            'Confidence':  [f'{p:.1%}'  for p in probs],
        })
        st.dataframe(df_probs, use_container_width=True, hide_index=True)

    with tab_xai:
        if run_shap and not is_demo:
            with st.spinner("🧮 Computing SHAP values (KernelExplainer, ~30-60 s)…"):
                shap_vals = compute_shap(rf, scaler, X)

            if shap_vals is not None:
                st.markdown("##### SHAP Feature Importance (for predicted class)")
                st.caption("Blue bars push the prediction **toward** this class; red bars push away.")
                fig_shap = plot_shap_bar(shap_vals, feature_names)
                st.pyplot(fig_shap, use_container_width=True)
                plt.close()

                # Top amino acid features
                aa_features = {f'AA_{aa}': shap_vals[feature_names.index(f'AA_{aa}')]
                               for aa in 'ACDEFGHIKLMNPQRSTVWY'}
                top_aa = sorted(aa_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                st.markdown("**Top amino acid composition drivers:**")
                cols_aa = st.columns(5)
                for col, (fname, val) in zip(cols_aa, top_aa):
                    col.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value' style='font-size:1.4rem; "
                        f"color:{\"#38c8ff\" if val>=0 else \"#f87171\"};'>"
                        f"{'↑' if val>=0 else '↓'} {fname.replace('AA_','')}</div>"
                        f"<div class='metric-label'>SHAP {val:+.4f}</div>"
                        f"</div>", unsafe_allow_html=True
                    )
            else:
                st.warning("SHAP computation failed — showing Random Forest feature importances instead.")
                fig_rf = plot_rf_importance(rf, feature_names)
                st.pyplot(fig_rf, use_container_width=True)
                plt.close()

        else:
            st.markdown("##### Random Forest Feature Importance (global)")
            if is_demo:
                st.info("Demo mode — feature importances are random. Train & save a real model for meaningful results.")
            else:
                st.caption("Enable **Compute SHAP explanation** in the sidebar for per-prediction explanations.")
            fig_rf = plot_rf_importance(rf, feature_names)
            st.pyplot(fig_rf, use_container_width=True)
            plt.close()

    with tab_seq:
        aa_counts = {aa: seq.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        aa_freq   = {aa: seq.count(aa) / len(seq) for aa in 'ACDEFGHIKLMNPQRSTVWY'}

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-value'>{len(seq)}</div><div class='metric-label'>Residues</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value'>{len(set(seq))}</div><div class='metric-label'>Unique AA</div></div>", unsafe_allow_html=True)

        # Hydrophobic fraction
        hydrophobic = sum(seq.count(aa) for aa in 'AVILMFYW')
        c3.markdown(f"<div class='metric-card'><div class='metric-value'>{hydrophobic/len(seq):.1%}</div><div class='metric-label'>Hydrophobic</div></div>", unsafe_allow_html=True)

        # Charged fraction
        charged = sum(seq.count(aa) for aa in 'RKDE')
        c4.markdown(f"<div class='metric-card'><div class='metric-value'>{charged/len(seq):.1%}</div><div class='metric-label'>Charged</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # AA frequency bar
        fig_aa, ax_aa = plt.subplots(figsize=(10, 2.8))
        fig_aa.patch.set_facecolor('#060b12')
        ax_aa.set_facecolor('#060b12')
        aa_list = list('ACDEFGHIKLMNPQRSTVWY')
        freqs   = [aa_freq[aa] for aa in aa_list]
        colors_aa = ['#38c8ff' if aa in 'RKDE' else
                     '#34d399' if aa in 'AVILMFYW' else '#7ab8d4' for aa in aa_list]
        ax_aa.bar(aa_list, freqs, color=colors_aa, edgecolor='none')
        ax_aa.set_ylabel('Frequency', color='#5a7a9a', fontsize=9)
        ax_aa.tick_params(colors='#8aa8c0', labelsize=9, length=0)
        ax_aa.spines[:].set_visible(False)
        ax_aa.set_facecolor('#060b12')

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#38c8ff', label='Charged'),
                           Patch(facecolor='#34d399', label='Hydrophobic'),
                           Patch(facecolor='#7ab8d4', label='Other')]
        ax_aa.legend(handles=legend_elements, fontsize=8,
                     facecolor='#0a1220', edgecolor='#1e3a5f', labelcolor='#c8d8e8')
        plt.tight_layout(pad=0.6)
        st.pyplot(fig_aa, use_container_width=True)
        plt.close()

        with st.expander("Full amino acid counts"):
            df_aa = pd.DataFrame({
                'AA':    aa_list,
                'Count': [aa_counts[aa] for aa in aa_list],
                'Freq':  [f'{aa_freq[aa]:.4f}' for aa in aa_list],
            })
            st.dataframe(df_aa, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER  (no active prediction)
# ─────────────────────────────────────────────
if not predict_btn:
    st.markdown("""
    <div style='text-align:center; padding: 60px 0 20px; color:#2a4a6a;
                font-family:Space Mono,monospace; font-size:0.8rem;'>
      Enter a sequence above and click <b style='color:#38c8ff'>⚡ Predict</b><br>
      or load an example from the dropdown
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding: 24px 0 8px; color:#1a3050;
            font-size:0.7rem; border-top: 1px solid #0e2040; margin-top:40px;'>
  ProteinOracle · ESM-2 (facebook/esm2_t6_8M_UR50D) · UniProt sequences · Random Forest
</div>
""", unsafe_allow_html=True)
