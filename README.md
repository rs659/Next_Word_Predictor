# ⚡ NeuroNext • LSTM Next Word Predictor & Story Continuation Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000.svg?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/Task-Language%20Modeling-8A2BE2.svg)](#)

An intelligent, real-time natural language processing (NLP) application powered by a deep **Recurrent Neural Network (LSTM)**. The system performs next-token probability forecasting, continuous story expansion, and deep vocabulary analysis with a sleek, cybernetic UI/UX.

---

## 🌟 Key Features

### 1. 🔮 Interactive Next Word Predictor
* **Top-K Candidate Ranking**: Dynamically ranks the most probable next words with exact softmax percentages.
* **⚡ QuickType Tap-to-Append**: Clickable suggestion chips allow you to build sentences word-by-word with single clicks, just like modern mobile keyboards (iOS QuickType / Gboard).
* **🎯 Shannon Certainty Index**: Analyzes the entropy of prediction distributions:
  $$\text{Entropy } H = -\sum_{i=1}^{K} p_i \log_2(p_i)$$
  Translates mathematical entropy into human-readable model certainty (*High Certainty*, *Moderate*, or *Exploratory / Diffuse*).
* **📈 Interactive Visualizations**: Horizontal probability distribution charts built with **Altair** alongside exact numerical telemetry tables.

### 2. ✍️ Continuous Autocomplete (Story Expander)
* **Multi-Token Text Generation**: Extends any seed prompt by $1$ to $30$ consecutive words.
* **Controllable Decoding Strategies**:
  * **Deterministic (Greedy)**: Always selects the maximum likelihood token at each step.
  * **Creative (Temperature Sampling)**: Allows tuning randomness ($0.1$ to $1.5$) to prevent repetitive loops and encourage creative generation.
* **Token Highlighting**: Visually differentiates the original seed phrase from newly synthesized tokens.

### 3. 🔍 Tokenizer & Vocabulary Intelligence
* **Vocabulary Lookup**: Instant verification across the model's **8,978-word vocabulary**, displaying token ID, character count, and frequency tier.
* **Fuzzy Spelling Matcher**: Automatically suggests valid vocabulary alternatives if an entered word is Out-of-Vocabulary (OOV) using string distance heuristics.
* **Live Tensor Pipeline Inspector**: Shows raw token sequences, tensor shapes, and pre-padded input arrays.

### 4. 🧠 Neural Architecture & Telemetry
* Interactive layer-by-layer breakdown:
  * **Embedding Layer**: 50-dimensional latent semantic space.
  * **LSTM Memory Core**: 128 recurrent units with internal gating mechanisms (input, forget, output gates).
  * **Dense Projection**: 10,000-class softmax distribution.

---

## 📐 Neural Network Architecture

```text
Input Sentence (Raw Text)
  │
  ▼ Regex Tokenizer & Lowercasing
Token Sequence [x1, x2, ..., xk]
  │
  ▼ Sequence Truncation & Pre-Padding (maxlen = 745)
Padded Tensor shape: (1, 745)
  │
  ▼ Embedding Layer (10,000 → 50 dimensions)
Latent Vector Matrix: (1, 745, 50)  ─── [500,000 Params]
  │
  ▼ LSTM Recurrent Core (128 units)
Temporal Context Vector: (1, 128)   ─── [91,648 Params]
  │
  ▼ Dense Softmax Output (10,000 classes)
Probability Distribution: P(w | Context) ─── [1,290,000 Params]
  │
  ▼ Top-K ArgSort & Sampling
Ranked Predictions & Visual Telemetry
```

### 📊 Model Specifications

| Attribute | Specification |
| :--- | :--- |
| **Model Type** | Sequential Recurrent Neural Network (LSTM) |
| **Total Parameters** | **1,881,648** (~7.18 MB) |
| **Trainable Weights** | 1,881,648 (100%) |
| **Vocabulary Size** | 8,978 words (Output classes: 10,000) |
| **Context Window (`max_len`)** | 745 tokens |
| **Embedding Dimensions** | 50 |
| **LSTM Hidden Units** | 128 |

---

## 📁 Repository Structure

```text
Next_Word_Pred/
│
├── app.py              # Modern Streamlit UI and inference engine
├── app_backup.py       # Preserved backup of original baseline implementation
├── lstm_model.h5       # Pre-trained Keras LSTM model weights
├── tokenizer.pkl       # Serialized Keras Tokenizer (8,978 words)
├── max_len.pkl         # Maximum sequence length integer (745)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python **3.9**, **3.10**, or **3.11** installed on your system.

### 2. Set Up Virtual Environment

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```

Once launched, open your browser and navigate to:
```text
http://localhost:8501
```

---

## 🛠️ Technology Stack

* **Language**: [Python 3](https://www.python.org/)
* **Deep Learning Framework**: [TensorFlow](https://tensorflow.org/) & [Keras](https://keras.io/)
* **Web & Interactive UI**: [Streamlit](https://streamlit.io/)
* **Visualization & Analytics**: [Altair](https://altair-viz.github.io/) & [Pandas](https://pandas.pydata.org/)
* **Numerical Processing**: [NumPy](https://numpy.org/)
* **Model Serialization**: `pickle` & `h5py`

---

## 💡 How It Works Under the Hood

1. **Text Normalization**: The input sentence is cleaned and tokenized into integer IDs mapped from `tokenizer.pkl`.
2. **Context Window Slicing**: Only the most recent $N \le (\text{max\_len} - 1)$ tokens are preserved to fit the trained sequence length.
3. **Pre-Padding**: `pad_sequences` inserts leading zeros up to a fixed length of $745$, preserving recency bias at the end of the sequence.
4. **Recurrent State Updating**: The LSTM processes the sequence step-by-step, updating hidden and cell states to summarize contextual semantics.
5. **Softmax Output**: The Dense layer produces an unnormalized logit vector transformed by Softmax into a true probability distribution over the lexicon:
   $$P(w_i \mid w_1, \dots, w_{t-1}) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
6. **Decoding & UI Feedback**: Top-$K$ probabilities are extracted and rendered with interactive suggestion pills and reactive charts.

---

## 📄 License
This project is open-source and available for educational and research purposes.
