import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import altair as alt
import difflib
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="NeuroNext • AI Predictive Text Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# HIGH-END MODERN STYLING (CSS)
# --------------------------------------------------
st.markdown("""
<style>
/* Font & Base Adjustments */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header Banner */
.hero-container {
    padding: 26px 32px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    margin-bottom: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.4);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    border-radius: 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 45%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
    letter-spacing: -0.6px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin: 0 0 16px 0;
    line-height: 1.5;
}

.metric-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 5px 12px;
    border-radius: 10px;
    font-size: 12px;
    color: #e2e8f0;
}

.metric-pill-val {
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    margin-bottom: 18px;
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.3);
}

/* Primary Prediction Showcase Card */
.showcase-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    position: relative;
    box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.2);
    margin-bottom: 20px;
}

.showcase-rank {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(99, 102, 241, 0.4);
    margin-bottom: 8px;
}

.showcase-word {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #ffffff;
    margin: 8px 0;
    text-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}

.showcase-metrics {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 16px;
    margin-top: 10px;
}

.conf-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
}

.certainty-badge {
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #38bdf8;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
}

/* Sentence Result Box */
.sentence-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 17px;
    line-height: 1.6;
    color: #f1f5f9;
    margin-bottom: 20px;
}

.word-highlight {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: #ffffff;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 700;
    margin-left: 2px;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
}

.generated-token {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    margin: 0 2px;
    display: inline-block;
}

/* Diagnostic Token Pills */
.token-pill-ok {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #34d399;
    padding: 3px 9px;
    border-radius: 8px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
}

.token-pill-oov {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.25);
    color: #f87171;
    padding: 3px 9px;
    border-radius: 8px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
}

/* Stat Tile in Grid */
.stat-tile {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}

.stat-tile-val {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
}

.stat-tile-lbl {
    font-size: 11px;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* Neural Network Layer Card */
.layer-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

/* Clean UI tweaks */
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.02);
}

.stButton button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CACHED MODEL AND ASSET LOADING
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model = tf.keras.models.load_model("lstm_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

try:
    with st.spinner("🧠 Initializing Neural Core & Model Weights..."):
        model, tokenizer, max_len = load_assets()
except Exception as e:
    st.error("❌ Failed to load neural weights or tokenizer files.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# CORE INFERENCE & HELPER FUNCTIONS
# --------------------------------------------------
def predict_next_words(text: str, top_n: int = 5):
    """Predict top N words with associated probabilities."""
    if not text.strip():
        return []

    tokens = tokenizer.texts_to_sequences([text])[0]
    if len(tokens) == 0:
        return []

    # Sequence truncation & padding
    tokens = tokens[-(max_len - 1):]
    padded = pad_sequences([tokens], maxlen=max_len, padding="pre")

    # Predict distribution
    preds = model.predict(padded, verbose=0)[0]

    # Sort descending
    top_indices = np.argsort(preds)[::-1]

    results = []
    index_word = tokenizer.index_word

    for idx in top_indices:
        if idx in index_word:
            word = index_word[idx]
            prob = float(preds[idx])
            results.append((word, prob))
            if len(results) >= top_n:
                break

    return results


def generate_sequence(seed_text: str, n_words: int = 5, temperature: float = 0.7, top_k: int = 5):
    """Continuously generate next N words using temperature/sampling."""
    current_text = seed_text.strip()
    words_added = []
    index_word = tokenizer.index_word

    for _ in range(n_words):
        tokens = tokenizer.texts_to_sequences([current_text])[0]
        if not tokens:
            break

        tokens = tokens[-(max_len - 1):]
        padded = pad_sequences([tokens], maxlen=max_len, padding="pre")

        preds = model.predict(padded, verbose=0)[0]
        top_indices = np.argsort(preds)[-top_k:][::-1]
        valid_indices = [idx for idx in top_indices if idx in index_word]

        if not valid_indices:
            break

        probs = np.array([preds[idx] for idx in valid_indices], dtype=np.float64)

        if temperature <= 0.05:
            # Pure greedy
            chosen_idx = valid_indices[0]
        else:
            # Temperature sampling
            probs = np.log(probs + 1e-12) / temperature
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / np.sum(exp_probs)
            chosen_idx = np.random.choice(valid_indices, p=probs)

        next_word = index_word[chosen_idx]
        current_text += " " + next_word
        words_added.append(next_word)

    return current_text, words_added


def analyze_input(text: str):
    """Inspect which words are recognized by the tokenizer and which are OOV."""
    words = re.findall(r"\b\w+\b", text.lower())
    recognized = []
    unrecognized = []
    for w in words:
        if w in tokenizer.word_index:
            recognized.append((w, tokenizer.word_index[w]))
        else:
            unrecognized.append(w)
    return recognized, unrecognized


def calculate_metrics(results):
    """Calculate confidence, Shannon entropy, and certainty index."""
    if not results:
        return 0.0, 0.0, "N/A"

    probs = np.array([p for _, p in results], dtype=np.float64)
    probs_norm = probs / np.sum(probs)

    # Shannon Entropy over top-k
    entropy = -np.sum(probs_norm * np.log2(probs_norm + 1e-12))
    max_entropy = np.log2(len(results)) if len(results) > 1 else 1.0
    certainty_pct = max(0.0, min(100.0, (1.0 - (entropy / max_entropy)) * 100))

    if certainty_pct > 70:
        certainty_label = "High Certainty"
    elif certainty_pct > 40:
        certainty_label = "Moderate Certainty"
    else:
        certainty_label = "Exploratory / Diffuse"

    return results[0][1], certainty_pct, certainty_label


# --------------------------------------------------
# SESSION STATE MANAGEMENT
# --------------------------------------------------
if "input_text" not in st.session_state:
    st.session_state["input_text"] = "the future of"

def set_text(preset_text: str):
    st.session_state["input_text"] = preset_text

def append_token(token: str):
    current = st.session_state.get("input_text", "").strip()
    if current:
        st.session_state["input_text"] = f"{current} {token}".strip()
    else:
        st.session_state["input_text"] = token

def clear_input_text():
    st.session_state["input_text"] = ""


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="font-size: 26px;">⚡</div>
            <div>
                <div style="font-weight: 800; font-size: 18px; color: #f8fafc;">NeuroNext Studio</div>
                <div style="font-size: 11px; color: #34d399; font-weight: 600;">● Neural Core Active</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🎛️ Inference Controls")
    top_k_slider = st.slider(
        "Top-K Suggestions",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of candidate next words to predict and rank."
    )

    st.markdown("---")

    st.subheader("✍️ Sequence Autocomplete")
    gen_length_slider = st.slider(
        "Auto-generate Length",
        min_value=1,
        max_value=25,
        value=5,
        help="Number of consecutive words to generate in multi-word mode."
    )

    sampling_mode = st.radio(
        "Decoding Strategy",
        options=["Deterministic (Greedy)", "Creative (Temperature)"],
        index=0
    )

    temperature_val = 0.0
    if sampling_mode == "Creative (Temperature)":
        temperature_val = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.05,
            help="Higher temperature yields more diverse and creative sequences."
        )

    st.markdown("---")

    st.subheader("💡 Quick Starter Prompts")
    presets = [
        "the future of",
        "artificial intelligence",
        "i want to",
        "deep learning is",
        "in the middle of",
        "how are you"
    ]

    for p in presets:
        st.button(f"👉 {p}", key=f"preset_{p}", on_click=set_text, args=(p,), use_container_width=True)

    st.markdown("---")

    with st.expander("📊 Model Blueprint", expanded=False):
        st.markdown(f"""
        - **Type:** Recurrent LSTM
        - **Embedding Dim:** 50
        - **LSTM Units:** 128
        - **Vocab Size:** {len(tokenizer.word_index):,}
        - **Sequence Window:** {max_len} tokens
        - **Total Weights:** 1,881,648 (~7.18 MB)
        """)

    if st.button("🧹 Reset All State", use_container_width=True):
        st.session_state["input_text"] = ""
        st.rerun()


# --------------------------------------------------
# MAIN HERO HEADER
# --------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">⚡ Neural Language Engine</div>
    <h1 class="hero-title">LSTM Next Word Predictor</h1>
    <div class="hero-subtitle">
        Intelligent real-time next-token forecasting and sequential story continuation powered by deep recurrent neural networks.
    </div>
    <div class="metric-pill-row">
        <div class="metric-pill">🧠 Parameters: <span class="metric-pill-val">1.88M</span></div>
        <div class="metric-pill">📖 Vocabulary: <span class="metric-pill-val">{len(tokenizer.word_index):,} words</span></div>
        <div class="metric-pill">📏 Context Window: <span class="metric-pill-val">{max_len} tokens</span></div>
        <div class="metric-pill">⚡ Architecture: <span class="metric-pill-val">Embedding(50) → LSTM(128) → Dense</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TABS INTERFACE
# --------------------------------------------------
tab_pred, tab_gen, tab_vocab, tab_arch = st.tabs([
    "🔮 Interactive Predictor",
    "✍️ Continuous Autocomplete",
    "🔍 Tokenizer & Vocab Explorer",
    "🧠 Model Architecture & Telemetry"
])


# ==================================================
# TAB 1: INTERACTIVE PREDICTOR
# ==================================================
with tab_pred:
    col_input, col_results = st.columns([1.15, 1], gap="large")

    with col_input:
        st.markdown("### ✍️ Input Sentence")
        st.markdown(
            "<span style='color: #94a3b8; font-size: 13px;'>Type your text below or pick a suggestion chip to build a sentence word-by-word:</span>",
            unsafe_allow_html=True
        )

        user_text = st.text_area(
            "Sentence input",
            key="input_text",
            placeholder="Type your sentence here... (e.g. 'the future of')",
            height=130,
            label_visibility="collapsed"
        )

        # Quick stats for input
        raw_words = user_text.strip().split() if user_text.strip() else []
        char_count = len(user_text)

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #64748b; margin-top: -8px; margin-bottom: 12px;">
                <span>Words: <b>{len(raw_words)}</b> | Characters: <b>{char_count}</b></span>
                <span>Context length: <b>{min(len(raw_words), max_len)} / {max_len}</b></span>
            </div>
        """, unsafe_allow_html=True)

        # Action Buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.2, 0.8])
        with btn_col1:
            do_predict = st.button("🔮 Predict Next Word", use_container_width=True, type="primary")
        with btn_col2:
            do_autocomplete = st.button("✨ Auto-Complete +5", use_container_width=True)
        with btn_col3:
            st.button("🗑️ Clear", on_click=clear_input_text, use_container_width=True)

        # Fast prediction if text exists
        predictions = []
        if user_text.strip():
            predictions = predict_next_words(user_text, top_n=top_k_slider)

        # Interactive Suggestion Chips (QuickType style)
        if predictions:
            st.markdown("#### ⚡ Tap to Append Next Word:")
            st.markdown(
                "<span style='color: #94a3b8; font-size: 12px;'>Click any suggestion chip to instantly append it to your sentence:</span>",
                unsafe_allow_html=True
            )

            chip_cols = st.columns(min(len(predictions), 5))
            for idx, (word, prob) in enumerate(predictions[:5]):
                with chip_cols[idx]:
                    st.button(
                        f"+ {word}\n({prob * 100:.1f}%)",
                        key=f"chip_pred_{idx}_{word}",
                        on_click=append_token,
                        args=(word,),
                        use_container_width=True
                    )

        # Multi-word Autocomplete Action
        if do_autocomplete and user_text.strip():
            with st.spinner("Generating sequence continuation..."):
                completed_text, added_words = generate_sequence(
                    user_text,
                    n_words=5,
                    temperature=temperature_val,
                    top_k=top_k_slider
                )
                st.session_state["input_text"] = completed_text
                st.rerun()

        # Input Diagnostic Breakdown
        if user_text.strip():
            rec_words, oov_words = analyze_input(user_text)
            with st.expander("🔎 Tokenizer Inspection for Input", expanded=False):
                if rec_words:
                    st.markdown("**Recognized Tokens:**")
                    pills_html = " ".join([
                        f'<span class="token-pill-ok">✓ {w} <span style="opacity:0.7">#{idx}</span></span>'
                        for w, idx in rec_words
                    ])
                    st.markdown(pills_html, unsafe_allow_html=True)

                if oov_words:
                    st.markdown("**Out of Vocabulary (Ignored by LSTM):**")
                    oov_html = " ".join([
                        f'<span class="token-pill-oov">⚠️ {w}</span>'
                        for w in oov_words
                    ])
                    st.markdown(oov_html, unsafe_allow_html=True)
                    st.caption("Words not present in the training vocabulary are bypassed during sequence tokenization.")

    # Column 2: Results & Visualizations
    with col_results:
        st.markdown("### 📊 Prediction Intelligence")

        if not user_text.strip():
            st.info("💡 Enter a sentence on the left or select a quick starter prompt to see AI predictions.")
        elif not predictions:
            st.warning("⚠️ No valid tokens recognized from your sentence in the model's vocabulary. Try entering common words or choose a starter prompt.")
        else:
            top_word, top_prob = predictions[0]
            top_prob_val, certainty_pct, certainty_label = calculate_metrics(predictions)

            # Showcase Hero Card for #1 Prediction
            st.markdown(f"""
            <div class="showcase-card">
                <div class="showcase-rank">👑 Rank #1 Candidate</div>
                <div class="showcase-word">{top_word}</div>
                <div class="showcase-metrics">
                    <span class="conf-badge">🎯 Confidence: {top_prob * 100:.2f}%</span>
                    <span class="certainty-badge">📈 {certainty_label} ({certainty_pct:.0f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Complete Sentence Preview
            st.markdown("##### 📝 Resulting Sentence Preview:")
            st.markdown(f"""
            <div class="sentence-box">
                {user_text.strip()} <span class="word-highlight">{top_word}</span>
            </div>
            """, unsafe_allow_html=True)

            # Altair Probability Distribution Chart
            st.markdown(f"##### 📈 Top-{len(predictions)} Probability Distribution:")
            chart_data = pd.DataFrame({
                "Word": [w for w, _ in predictions],
                "Probability": [p * 100 for _, p in predictions],
                "Rank": [f"#{i+1}" for i in range(len(predictions))]
            })

            bar_chart = alt.Chart(chart_data).mark_bar(
                cornerRadiusTopRight=6,
                cornerRadiusBottomRight=6,
                height=22
            ).encode(
                x=alt.X("Probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, max(chart_data['Probability'].max() * 1.15, 10)])),
                y=alt.Y("Word:N", sort=None, title=None),
                color=alt.Color(
                    "Probability:Q",
                    scale=alt.Scale(scheme="purples"),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("Rank:N", title="Rank"),
                    alt.Tooltip("Word:N", title="Word"),
                    alt.Tooltip("Probability:Q", title="Probability (%)", format=".2f")
                ]
            ).properties(
                height=min(260, 45 * len(predictions))
            ).configure_axis(
                labelFontSize=13,
                titleFontSize=12
            )

            st.altair_chart(bar_chart, use_container_width=True)

            # Detailed Candidate Breakdown Table
            with st.expander("📋 View Exact Numerical Table", expanded=False):
                table_df = pd.DataFrame({
                    "Rank": [f"#{i+1}" for i in range(len(predictions))],
                    "Candidate Word": [w for w, _ in predictions],
                    "Probability (%)": [f"{p*100:.3f}%" for _, p in predictions],
                    "Raw Score": [f"{p:.6f}" for _, p in predictions]
                })
                st.dataframe(table_df, use_container_width=True, hide_index=True)


# ==================================================
# TAB 2: CONTINUOUS AUTOCOMPLETE & STORY EXPANDER
# ==================================================
with tab_gen:
    st.markdown("### ✍️ Deep Sequence Generator")
    st.markdown(
        "<span style='color: #94a3b8; font-size: 14px;'>Generate continuous sequences of text from any seed phrase. Choose greedy mode for deterministic logic or temperature sampling for creative storytelling.</span>",
        unsafe_allow_html=True
    )

    gen_col1, gen_col2 = st.columns([1, 1], gap="large")

    with gen_col1:
        st.markdown("#### 1. Configure Seed & Generation")
        gen_seed = st.text_input(
            "Seed Phrase",
            value=st.session_state.get("input_text", "the future of"),
            placeholder="Type seed phrase here..."
        )

        n_words_select = st.slider(
            "Number of Words to Generate",
            min_value=1,
            max_value=30,
            value=gen_length_slider,
            key="deep_gen_words"
        )

        temp_select = st.slider(
            "Creativity / Temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.4 if sampling_mode == "Deterministic (Greedy)" else temperature_val,
            step=0.05,
            key="deep_gen_temp",
            help="0.0 = Deterministic/Greedy (most likely word each step). Higher values inject randomness."
        )

        gen_button = st.button("🚀 Generate Story Sequence", type="primary", use_container_width=True)

    with gen_col2:
        st.markdown("#### 2. Generated Output")

        if gen_button or ("last_gen_result" in st.session_state and st.session_state.get("last_gen_seed") == gen_seed):
            with st.spinner("🧠 Generating sequence token-by-token..."):
                full_gen, added_words = generate_sequence(
                    gen_seed,
                    n_words=n_words_select,
                    temperature=temp_select,
                    top_k=top_k_slider
                )
                st.session_state["last_gen_result"] = full_gen
                st.session_state["last_gen_added"] = added_words
                st.session_state["last_gen_seed"] = gen_seed

            tokens_html = " ".join([f'<span class="generated-token">{w}</span>' for w in added_words])

            st.markdown(f"""
            <div class="sentence-box">
                <span style="color: #94a3b8;">{gen_seed.strip()}</span> {tokens_html}
            </div>
            """, unsafe_allow_html=True)

            st.success(f"✨ Successfully generated **{len(added_words)} tokens**.")

            # Action: Use as main input
            def transfer_to_main():
                st.session_state["input_text"] = st.session_state["last_gen_result"]

            st.button("📥 Load Result into Interactive Predictor", on_click=transfer_to_main, use_container_width=True)
        else:
            st.info("Click **Generate Story Sequence** to watch the LSTM generate successive words.")


# ==================================================
# TAB 3: TOKENIZER & VOCABULARY EXPLORER
# ==================================================
with tab_vocab:
    st.markdown("### 🔍 Tokenizer & Vocabulary Intelligence")
    st.markdown(
        "<span style='color: #94a3b8; font-size: 14px;'>Search the model's vocabulary of 8,978 words, inspect token IDs, and diagnose unknown tokens.</span>",
        unsafe_allow_html=True
    )

    v_col1, v_col2 = st.columns([1, 1], gap="large")

    with v_col1:
        st.markdown("#### 🔎 Word Lookup")
        query_word = st.text_input("Search a word in vocabulary:", value="learning").strip().lower()

        if query_word:
            if query_word in tokenizer.word_index:
                word_id = tokenizer.word_index[query_word]
                st.success(f"✅ **'{query_word}'** is present in the vocabulary!")

                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f"""
                    <div class="stat-tile">
                        <div class="stat-tile-val">#{word_id}</div>
                        <div class="stat-tile-lbl">Token ID</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"""
                    <div class="stat-tile">
                        <div class="stat-tile-val">{len(query_word)}</div>
                        <div class="stat-tile-lbl">Character Length</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col3:
                    freq_rank = "Top 5%" if word_id < 450 else "Standard"
                    st.markdown(f"""
                    <div class="stat-tile">
                        <div class="stat-tile-val" style="color: #38bdf8;">{freq_rank}</div>
                        <div class="stat-tile-lbl">Frequency Tier</div>
                    </div>
                    """, unsafe_allow_html=True)

                def use_looked_up_word(w=query_word):
                    append_token(w)

                st.button(f"➕ Append '{query_word}' to current prompt", on_click=use_looked_up_word)

            else:
                st.error(f"❌ **'{query_word}'** is OUT OF VOCABULARY (OOV).")
                # Fuzzy suggestions
                all_vocab = list(tokenizer.word_index.keys())
                close_matches = difflib.get_close_matches(query_word, all_vocab, n=5, cutoff=0.6)
                if close_matches:
                    st.markdown("**Did you mean one of these known words?**")
                    for m in close_matches:
                        st.button(f"👉 {m} (ID: #{tokenizer.word_index[m]})", key=f"sug_{m}", on_click=set_text, args=(m,))
                else:
                    st.caption("No closely matching words found in vocabulary.")

    with v_col2:
        st.markdown("#### 🔢 Sentence Tokenization Pipeline")
        st.markdown(
            "<span style='color: #94a3b8; font-size: 13px;'>Test how any custom text gets converted into input tensors:</span>",
            unsafe_allow_html=True
        )

        test_text = st.text_input("Enter sentence to encode:", value="artificial intelligence and deep learning")
        if test_text.strip():
            raw_seq = tokenizer.texts_to_sequences([test_text])[0]
            st.markdown(f"**Generated Token IDs:** `{raw_seq}`")

            padded_sample = pad_sequences([raw_seq[-(max_len-1):]], maxlen=max_len, padding="pre")
            st.markdown(f"**Padded Tensor Shape:** `{padded_sample.shape}` *(Batch: 1, Max Length: {max_len})*")

            # Non-zero tokens preview
            non_zeros = padded_sample[0][padded_sample[0] != 0]
            st.markdown(f"**Active Input Indices:** `{list(non_zeros)}`")


# ==================================================
# TAB 4: MODEL ARCHITECTURE & TELEMETRY
# ==================================================
with tab_arch:
    st.markdown("### 🧠 Model Architecture & Telemetry")
    st.markdown(
        "<span style='color: #94a3b8; font-size: 14px;'>Comprehensive specifications and tensor transformation flow for the trained LSTM neural network.</span>",
        unsafe_allow_html=True
    )

    t_col1, t_col2 = st.columns([1.2, 1], gap="large")

    with t_col1:
        st.markdown("#### 📐 Neural Layer Breakdown")

        layers_info = [
            {
                "name": "embedding_1",
                "type": "Embedding Layer",
                "input_dim": "10,000",
                "output_dim": "50",
                "output_shape": f"(None, {max_len}, 50)",
                "params": "500,000",
                "desc": "Maps discrete token IDs into dense 50-dimensional continuous semantic vector representations."
            },
            {
                "name": "lstm",
                "type": "Long Short-Term Memory (LSTM)",
                "input_dim": "50",
                "output_dim": "128",
                "output_shape": "(None, 128)",
                "params": "91,648",
                "desc": "Recurrent memory layer that captures temporal dependencies, sequential context, and long-range syntactic patterns."
            },
            {
                "name": "dense_1",
                "type": "Dense Layer with Softmax",
                "input_dim": "128",
                "output_dim": "10,000",
                "output_shape": "(None, 10000)",
                "params": "1,290,000",
                "desc": "Projects hidden state representations across the vocabulary logits and computes normalized probability distribution."
            }
        ]

        for layer in layers_info:
            st.markdown(f"""
            <div class="layer-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 16px; color: #f8fafc;">{layer['type']}</span>
                    <span style="background: rgba(99, 102, 241, 0.2); color: #818cf8; font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 6px;">{layer['params']} params</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #38bdf8; margin: 4px 0;">
                    Output Shape: {layer['output_shape']}
                </div>
                <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">
                    {layer['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("#### ⚡ Hardware & Computational Telemetry")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="stat-tile">
                <div class="stat-tile-val">1.88M</div>
                <div class="stat-tile-lbl">Total Parameters</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="stat-tile">
                <div class="stat-tile-val">7.18 MB</div>
                <div class="stat-tile-lbl">Model Weight Size</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-tile-val">{max_len}</div>
                <div class="stat-tile-lbl">Context Window</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-tile-val">{len(tokenizer.word_index):,}</div>
                <div class="stat-tile-lbl">Vocabulary Size</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔄 Data Pipeline Flow")
        st.code(f"""
Input Sentence (Raw Text)
  │
  ▼ Regex & Lowercase Tokenizer
Token Sequence [x1, x2, ..., xk]
  │
  ▼ Pre-Padding to Length ({max_len})
Padded Tensor (1, {max_len})
  │
  ▼ Embedding Lookup (Dim: 50)
Vector Matrix (1, {max_len}, 50)
  │
  ▼ Recurrent Memory (LSTM: 128)
Context Vector (1, 128)
  │
  ▼ Fully Connected (Dense: 10,000) + Softmax
Probability Distribution P(w | Context)
  │
  ▼ ArgSort / Top-K Sampling
Ranked Predictions & Confidences
        """, language="text")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #64748b; padding: 10px 0;">
    <span>⚡ <b>NeuroNext Studio</b> • Advanced LSTM Language Modeling Interface</span>
    <span>TensorFlow 2.x • Keras • Python • Streamlit</span>
</div>
""", unsafe_allow_html=True)