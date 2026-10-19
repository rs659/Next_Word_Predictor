import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 18px;
    margin-bottom: 30px;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.predicted-word {
    font-size: 36px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND FILES
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "lstm_model.h5",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


# --------------------------------------------------
# LOAD EVERYTHING
# --------------------------------------------------

try:

    model, tokenizer, max_len = load_model()

except Exception as e:

    st.error("❌ Could not load the model or tokenizer.")

    st.code(str(e))

    st.stop()


# --------------------------------------------------
# PREDICT NEXT WORD
# --------------------------------------------------

def predict_next_words(text, top_n=5):

    # Convert input sentence to token numbers
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    # Keep only the last max_len - 1 tokens
    token_list = token_list[-(max_len - 1):]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len,
        padding="pre"
    )

    # Prediction
    predictions = model.predict(
        token_list,
        verbose=0
    )[0]

    # Get top N word IDs
    top_indices = np.argsort(predictions)[-top_n:][::-1]

    results = []

    # Reverse tokenizer vocabulary
    index_word = tokenizer.index_word

    for index in top_indices:

        word = index_word.get(index)

        if word is not None:

            probability = float(predictions[index])

            results.append(
                (word, probability)
            )

    return results


# --------------------------------------------------
# UI
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 LSTM Next Word Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict the next word using a trained LSTM language model</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("ℹ️ Model Information"):

    st.write("**Model:** LSTM")

    st.write("**Maximum sequence length:**", max_len)

    st.write("**Vocabulary size:**", len(tokenizer.word_index))

    st.write("**Architecture:**")

    st.code("""
Input
  ↓
Embedding
  ↓
LSTM (128 units)
  ↓
Dense
  ↓
Softmax
  ↓
Next Word
    """)


# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.subheader("✍️ Enter your sentence")

text = st.text_area(
    "Start typing...",
    placeholder="Example: machine learning is",
    height=120
)


# --------------------------------------------------
# BUTTON
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    predict_button = st.button(
        "🔮 Predict Next Word",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


# --------------------------------------------------
# CLEAR
# --------------------------------------------------

if clear_button:

    st.rerun()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    if not text.strip():

        st.warning("⚠️ Please enter a sentence first.")

    else:

        with st.spinner("🧠 Predicting..."):

            results = predict_next_words(
                text,
                top_n=5
            )

        if not results:

            st.error(
                "Could not generate a prediction. "
                "The entered words may not exist in the tokenizer vocabulary."
            )

        else:

            # --------------------------------------
            # BEST PREDICTION
            # --------------------------------------

            best_word = results[0][0]
            best_probability = results[0][1]

            st.markdown(
                f"""
                <div class="prediction-box">

                <div>Predicted Next Word</div>

                <div class="predicted-word">
                {best_word}
                </div>

                <div>
                Confidence: {best_probability * 100:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # --------------------------------------
            # COMPLETE SENTENCE
            # --------------------------------------

            st.subheader("📝 Generated Text")

            st.success(
                text.strip() + " " + best_word
            )


            # --------------------------------------
            # TOP 5
            # --------------------------------------

            st.subheader("📊 Top 5 Predictions")

            for i, (word, probability) in enumerate(results):

                percentage = probability * 100

                col1, col2 = st.columns([3, 1])

                with col1:

                    st.write(
                        f"**{i + 1}. {word}**"
                    )

                with col2:

                    st.write(
                        f"{percentage:.2f}%"
                    )

                st.progress(
                    min(probability, 1.0)
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Built with Python • TensorFlow • Keras • Streamlit • LSTM"
)