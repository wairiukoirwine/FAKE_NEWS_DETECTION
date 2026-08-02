# Fake News Detection Using NLP and Machine Learning

Classical NLP + ML pipeline that classifies news articles as fake or real, built as a
step-by-step Jupyter notebook (every code cell preceded by a markdown explanation).

## Project structure

```
fake_news_project/
├── data/
│   ├── Fake.csv          # Kaggle fake-news articles (label = 0)
│   └── True.csv          # Kaggle real-news articles (label = 1)
├── notebook/
│   └── fake_news_detection.ipynb
├── outputs/               # generated automatically when you run the notebook — see below
├── requirements.txt
└── README.md
```

## The `outputs/` folder

Running the notebook writes each key result to its own file in `outputs/`, in addition to
displaying it inline. Useful if you want to reference, share, or explain a single result without
opening the whole notebook:

| File | Produced by | What it is |
|---|---|---|
| `class_distribution.png` | Cell 5 | Bar chart of fake vs. real article counts |
| `cleaned_text_example.csv` | Cell 10 | 5-row sample: raw text vs. preprocessed text, side by side |
| `results_table.csv` | Cell 17 | Accuracy/precision/recall/F1 for all three models |
| `model_comparison_metrics.png` | Cell 18 | Bar chart comparing precision/recall/F1 across models |
| `confusion_matrices.png` | Cell 19 | Confusion matrix heatmaps for all three models, side by side |

The folder is created automatically the first time you run Cell 3 — you don't need to make it
yourself, and it's safe to delete and re-run if you want a clean set of outputs.

## Dataset schema (confirmed from your files)

Both `Fake.csv` and `True.csv` have columns: `title`, `text`, `subject`, `date`.
The notebook loads both, tags them with a `label` column (0 = fake, 1 = real), and combines them.

## What's included vs. excluded

- ✅ Sections 1–6 and 8 from the spec: setup, data loading/EDA, preprocessing, TF-IDF +
  embedding-style features, three baseline models (Logistic Regression, Linear SVM,
  Multinomial Naive Bayes), evaluation, comparison charts, and conclusion.
- ❌ Section 7 (BERT/transformer fine-tuning) is **excluded** per your instructions. A markdown
  cell in the notebook explains what that section would involve if you add it later — it needs a
  GPU.

The notebook has been test-executed end-to-end (on a small data sample, to confirm every cell
runs without errors) before packaging — the full run on all ~44k rows will take longer, mainly due
to Cell 10 (lemmatizing every article, a few minutes on CPU) and Cell 15 (linear SVM training).

## How to run in VS Code

1. **Unzip the project** and open the `fake_news_project` folder in VS Code
   (`File > Open Folder...`).

2. **Create a virtual environment** (recommended, keeps this project's packages isolated):
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the Jupyter extension in VS Code** if you don't have it already
   (search "Jupyter" by Microsoft in the Extensions panel).

5. **Open the notebook:** `notebook/fake_news_detection.ipynb`.
   VS Code will prompt you to select a kernel — choose the `venv` Python interpreter you just
   created (top-right kernel picker, or `Ctrl+Shift+P` → "Notebook: Select Notebook Kernel").

6. **Run all cells:** `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) → "Run All", or use the ▶▶ "Run
   All" button at the top of the notebook. NLTK resource downloads (Cell 2) require an internet
   connection the first time; after that they're cached locally.

7. **Expect Cell 10 (full-dataset preprocessing) to take a few minutes** — it lemmatizes ~44,000
   articles. Everything else runs quickly.

Data paths in the notebook are relative (`../data/Fake.csv`, `../data/True.csv`), so keep the
folder structure intact — don't move the notebook out of `notebook/` without updating `DATA_DIR`
in Cell 3.

## Alternative: classic Jupyter, no VS Code

```bash
source venv/bin/activate
jupyter notebook notebook/fake_news_detection.ipynb
```

## Extending the project later

- **Add BERT (Section 7):** tokenize with a `BertTokenizer`, fine-tune
  `BertForSequenceClassification` (or load a pretrained fake-news model from Hugging Face Hub),
  and reuse the `evaluate_model()` function from Cell 17 for a consistent comparison. Requires a
  GPU — CPU fine-tuning on ~35k documents is impractically slow.
- **Real word embeddings:** Cell 12 uses a `HashingVectorizer` + `TruncatedSVD` stand-in for
  Word2Vec/GloVe (no internet access to download pretrained vector files was available while
  building this). Swap in `gensim.models.KeyedVectors` if you have a local embedding file.
- **Cross-source validation:** test the trained models on a different, out-of-distribution
  fake/real news dataset to check whether performance holds up outside this dataset's specific
  sources and time period (see the limitations note in the notebook's conclusion).
