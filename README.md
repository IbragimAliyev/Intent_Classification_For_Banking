# Intent Classification for Banking

NLP course project (COE025): classifying banking customer messages into 14 intents.
We built a synthetic English dataset with an LLM, trained and compared five methods on it,
and tested all of them on real customer calls from the MINDS-14 dataset.

## Results

Macro-F1 with 95% bootstrap confidence intervals (1,000 resamples, seed 42).
Full tables, paired comparisons and the chart are in `notebooks/09_benchmark.ipynb` and `results/benchmark.png`.

| Model | Type | Architecture | Specific model | Synthetic test | MINDS-14 (real) |
|---|---|---|---|---|---|
| `jev` | new | not public (commercial API) | `jev-1.13.0` (via `jev-latest`) | **0.989** [0.979, 0.998] | **0.951** [0.941, 0.961] |
| `xlmr_base` | new | XLM-RoBERTa, fine-tuned | `xlm-roberta-base` | 0.982 [0.968, 0.992] | 0.924 [0.913, 0.936] |
| `sbert_svm` | new | MPNet encoder (frozen) + linear SVM | `all-mpnet-base-v2` | 0.987 [0.975, 0.996] | 0.917 [0.904, 0.928] |
| `bert_base` | baseline | BERT, fine-tuned | `bert-base-uncased` | 0.980 [0.965, 0.991] | 0.916 [0.903, 0.927] |
| `tfidf_lr` | baseline | TF-IDF (word 1-2 grams) + logistic regression | trained from scratch | 0.939 [0.914, 0.959] | 0.864 [0.847, 0.879] |
| `tfidf_nb` | baseline | TF-IDF (word 1-2 grams) + multinomial Naive Bayes | trained from scratch | 0.930 [0.902, 0.951] | 0.819 [0.801, 0.836] |

**Baseline** = a method covered in the lectures in its vanilla form; **new** = a method beyond them.
Sentence-level embeddings (sentence-transformers) and the multilingual XLM-RoBERTa were not part of
the lectures, so `sbert_svm` and `xlmr_base` count as new methods, next to the Jev API.

**Main findings**

- **The synthetic test set is nearly saturated.** All models except TF-IDF score 0.98-0.99 and are
  statistically tied, so the real MINDS-14 data is needed to separate them.
- **On MINDS-14, Jev is clearly the best model**, even though it is never trained on our data
  (zero-shot, with one-sentence intent descriptions). Paired bootstrap intervals against every
  other model exclude 0.
- **All models lose accuracy on real data.** The trained models drop 5.8 to 11.1 points of macro-F1
  (Naive Bayes the most), Jev 3.8.
- **Naive Bayes is the weakest model.** It ties with logistic regression on the synthetic test, but on
  MINDS-14 logistic regression on the same features is clearly better, so the classifier matters on real data.
- **Fine-tuning did not beat frozen sentence embeddings:** XLM-R, BERT and embeddings + SVM are
  statistically tied on MINDS-14.

Why we use macro-F1 as the main metric: [`report/metric_justification.md`](report/metric_justification.md).

## Data

| Dataset | Rows | Use |
|---|---|---|
| Synthetic (`data/raw/synthetic_raw.csv`) | 3,010 (3,007 after near-duplicate removal) | train / validation / test, 70 / 15 / 15, stratified, seed 42 |
| &nbsp;&nbsp;train / val / test (`data/processed/`) | 2,104 / 451 / 452 | training, model selection, final test |
| MINDS-14 English (`data/processed/minds14_en.csv`) | 1,809 (en-US 563, en-GB 592, en-AU 654) | real-data test only, never used for training or tuning |

The synthetic messages mix typed chat and phone-call style, with typos and US / UK / Australian English.
MINDS-14 ([`PolyAI/minds14`](https://huggingface.co/datasets/PolyAI/minds14)) contains transcribed
customer calls with the same 14 intents.

### How the synthetic dataset was created

The synthetic messages were generated with Claude (Anthropic) in several rounds:

- **Labels:** the 14 intents of MINDS-14, so the data can be tested on real customer calls without
  relabeling. Each intent has a written definition, plus rules for easily confused pairs (e.g. a
  card failing abroad is `abroad`, not `card_issues`).
- **Size:** 215 messages per intent (3,010 in total), so the classes are balanced and the dataset
  stays well above the required 1,000 rows after cleaning.
- **Diversity:** about 60% typed chat and 40% spoken phone-call style; different tones, lengths and
  personas; typos and lowercase text; US, UK and Australian English; about 20% indirect phrasing
  without the obvious keyword; the intent name never appears in the text.
- **Pilot first:** 10 messages per intent were generated and reviewed before the full run.
- **Checks:** class balance, duplicates, label correctness and formatting. Cleaning and splitting
  are done by `src/preprocess.py` (notebook `01`).

**Intents:** `abroad`, `address`, `app_error`, `atm_limit`, `balance`, `business_loan`, `card_issues`,
`cash_deposit`, `direct_debit`, `freeze`, `high_value_payment`, `joint_account`, `latest_transactions`, `pay_bill`.

## Repository structure

```
data/raw/          original synthetic dataset (never edited)
data/processed/    train / val / test split and MINDS-14 English
src/               shared code
  preprocess.py    normalization, near-duplicate removal, stratified split
  evaluate.py      shared scoring: accuracy, macro-F1, report, confusion matrix, metrics.csv
  finetune.py      shared fine-tuning code for BERT and XLM-R
  bootstrap.py     bootstrap confidence intervals and paired comparisons
notebooks/         one notebook per step (see below)
results/           metrics.csv, confusion matrices, benchmark chart
  predictions/     one file per model and dataset: text, intent, pred
  jev/             Jev probabilities, final intent descriptions and raw API answers
report/            written parts of the report
models/            trained models (not in git, recreated by the notebooks)
```

## Notebooks

| # | Notebook | What it does | Author |
|---|---|---|---|
| 01 | `01_preprocessing` | normalization, near-duplicate removal, 70/15/15 split | Mahmut |
| 02 | `02_tfidf_baseline` | TF-IDF + logistic regression, C tuned on validation | Ibrahim |
| 02b | `02b_naive_bayes` | TF-IDF + multinomial Naive Bayes, alpha tuned on validation | Ibrahim |
| 03 | `03_sentence_embeddings_svm` | `all-mpnet-base-v2` embeddings + linear SVM | Shahem |
| 04 | `04_embeddings_analysis` | embeddings vs TF-IDF: error analysis, typo test, t-SNE plot | Tuğra |
| 05 | `05_bert_finetune` | fine-tuning `bert-base-uncased` | Shahem |
| 06 | `06_xlmr_finetune` | fine-tuning `xlm-roberta-base` | Tuğra |
| 07 | `07_minds14` | prepares MINDS-14 English, scores the classic models on it | Mahmut |
| 08 | `08_jev` | Jev API: option design chosen on validation, final tests, cost and confidence analysis | Ibrahim |
| 09 | `09_benchmark` | final benchmark: all models, confidence intervals, paired comparisons, chart | Mahmut |

## How to run

**1. Environment** (Python 3.12):

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

**2. Order.** Run the notebooks from inside `notebooks/` (all paths start with `../`), in this order:

- `01` creates the split. Its output is already in `data/processed/`, so this step is optional.
- `02`, `02b` and `03` train the classic models and save them to `models/`. Notebooks `04` and `07` need these files.
- `05` and `06` need a GPU. We ran them on Google Colab (T4): clone the repository there and run from `notebooks/`.
- `07` prepares MINDS-14 (its output is already in `data/processed/`) and needs the models from `02` and `03`.
- `08` needs a TypeSafe API key (see below).
- `09` only reads `results/predictions/`, so it can be run on its own.

**3. Jev API key.** Create a file `.env` in the project root with `TYPESAFE_API_KEY=your-key`.
`.env` is in `.gitignore`; never commit the key. All raw Jev answers are saved in `results/jev/raw/`,
so the results can be checked without calling the API again.

## Conventions and things to know

- Random seed **42** everywhere. Hyperparameters and Jev's option design were chosen on the
  **validation** set only; each test set was used once per model.
- Every model is scored with `src/evaluate.py`. **Each call appends a row to `results/metrics.csv`**,
  so rerunning an evaluation cell creates a duplicate row that has to be deleted.
- **Do not rerun notebook 08** unless you mean to: it calls the paid Jev API again and adds rows to `metrics.csv`.
- `jev-latest` is an alias. All our requests were answered by `jev-1.13.0`; a later version may give different results.
- Fine-tuning results come from a single run (seed 42), so small differences between transformer models
  should not be over-interpreted.

## License

Code: MIT. Synthetic dataset: CC BY 4.0. MINDS-14 belongs to PolyAI and is used under its own
CC BY 4.0 license. Details in [`LICENSE`](LICENSE).

## Team

Ibrahim Aliyev, Mahmut Örnek, Shahem Hasunoglu, Tuğra Geler
