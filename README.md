# 📄🔍 SkimLit — NLP Sentence Classification for Medical Abstracts

> Classifying sentences in PubMed abstracts into rhetorical roles using a **Tribrid Embedding Model** (Token + Character + Positional), replicating the architecture from the [Neural Networks for Joint Sentence Classification in Medical Paper Abstracts](https://arxiv.org/abs/1612.05251) paper.

---

## 📌 Overview

Reading medical research papers is time-consuming. **SkimLit** solves this by automatically labelling each sentence in a PubMed abstract with its rhetorical role — making it possible to skim straight to the Results or Methods without reading everything.

| Metric | Value |
|---|---|
| **Dataset** | PubMed 20k RCT |
| **Classes** | 5 (BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS) |
| **Best Model** | Tribrid Embedding (Token + Character + Positional) |
| **Framework** | TensorFlow / Keras |
| **Training Environment** | Google Colab (GPU) |

---

## 🗂️ Project Structure

```
skimlit-nlp-classifier/
│
├── SkimLit_NLP_Project.ipynb   # Main notebook (full pipeline)
└── README.md
```

---

## 🔄 Pipeline

```
PubMed 20k RCT Dataset
        │
        ▼
Data Parsing & Preprocessing
(sentence extraction, line numbers, total lines)
        │
        ▼
Feature Engineering
(token vectorization, character splitting, positional one-hot encoding)
        │
        ▼
Tribrid Embedding Model
┌─────────────────────────────────────┐
│  Token Branch   → Pretrained USE    │
│  Character Branch → Bi-LSTM         │
│  Positional Branch → Dense          │
└──────────────┬──────────────────────┘
               │ Concatenate
               ▼
         Dense + Dropout
               │
               ▼
        Softmax (5 classes)
        │
        ▼
Evaluation + Error Analysis + Live Abstract Prediction
```

---

## 🧠 Model Architecture

Five models were built in progression:

| Model | Input Type | Highlights |
|---|---|---|
| Model 1 | Token embeddings | Custom Conv1D |
| Model 2 | Pretrained USE | TF Hub transfer learning |
| Model 3 | Character embeddings | Custom Conv1D on chars |
| Model 4 | Token + Character | Hybrid, Bi-LSTM on chars |
| **Model 5** ✅ | **Token + Character + Positional** | **Best — Tribrid embedding** |

**Model 5 Architecture:**
```
token_inputs (USE) ──► Dense(128)  ──┐
char_inputs (Bi-LSTM) ──────────────►  Concatenate ──► Dense(256) ──► Dropout(0.5)
line_number_inputs ──► Dense(32) ──┐                                        │
total_lines_inputs ──► Dense(32) ──►  Concatenate ───────────────────────────►
                                                                             │
                                                                    Dense(5, softmax)
```

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Loss | CategoricalCrossentropy (label_smoothing=0.2) |
| Optimizer | Adam (default lr) |
| Epochs | 3 |
| Batch size | 32 |
| Steps per epoch | 10% of training batches |

> Label smoothing is used to prevent the model from becoming overconfident on training examples.

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Open `SkimLit_NLP_Project.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** → Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom

### Option 2 — Local

```bash
# Clone the repo
git clone https://github.com/RehanRaza/skimlit-nlp-classifier.git
cd skimlit-nlp-classifier

# Install dependencies
pip install tensorflow tensorflow-hub scikit-learn pandas matplotlib spacy
python -m spacy download en_core_web_sm

# Launch notebook
jupyter notebook SkimLit_NLP_Project.ipynb
```

---

## 📦 Dependencies

```
tensorflow >= 2.4.0
tensorflow-hub
scikit-learn
pandas
numpy
matplotlib
spacy
```

---

## 🔍 Sample Output

Given a raw PubMed abstract, the model labels each sentence:

```
[BACKGROUND]  Hepatitis C virus (HCV) infection is a major cause of chronic liver disease...
[OBJECTIVE]   We assessed the efficacy of combination therapy with interferon...
[METHODS]     Patients were randomly assigned to receive treatment for 24 or 48 weeks...
[RESULTS]     Sustained virological response was achieved in 43% of patients...
[CONCLUSIONS] Combination therapy is effective for treatment of chronic HCV infection...
```

---

## 📚 References

- [PubMed 20k RCT Dataset](https://github.com/Franck-Dernoncourt/pubmed-rct) — Dernoncourt & Lee, 2017
- [Neural Networks for Joint Sentence Classification](https://arxiv.org/abs/1612.05251) — Dernoncourt & Lee, 2016
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) — Google, TF Hub

---

## 👤 Author

**Rehan Raza**  
[GitHub](https://github.com/RehanRaza) · [LinkedIn](https://linkedin.com/in/RehanRaza)
