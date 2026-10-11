# Contribution: Shahem Hasunoglu

- **Student ID:** 2409015433
- **GitHub:** [ShahemHassoun](https://github.com/ShahemHassoun)
- **Role:** sentence-embedding model, shared fine-tuning code, BERT fine-tuning

## Summary

I built the sentence-embedding + linear SVM model, wrote the shared fine-tuning code used for both
transformer models, and fine-tuned BERT.

## 1. Sentence embeddings + linear SVM (`notebooks/03_sentence_embeddings_svm.ipynb`)

- Each message is encoded once by the frozen sentence transformer `all-mpnet-base-v2`
  (MPNet architecture, 768-dimensional vectors); only the linear SVM on top is trained.
- Tuned C on the validation set over `[0.001, ..., 100]`; the best value C = 1 lies inside the
  range (validation macro-F1 0.9867).
- Synthetic test: accuracy 0.9867, macro-F1 0.9866, i.e. 6 errors instead of the baseline's 27.
  The biggest gain was on `abroad` (F1 0.77 -> 1.00).

## 2. Shared fine-tuning code (`src/finetune.py`)

- One reusable module for fine-tuning any Hugging Face model, so BERT and XLM-RoBERTa are trained in
  exactly the same way: tokenizer and `AutoModelForSequenceClassification`, labels in sorted order,
  maximum length 64, seed 42, evaluation after every epoch and the best epoch kept by validation macro-F1.

## 3. BERT fine-tuning (`notebooks/05_bert_finetune.ipynb`)

- Model: `bert-base-uncased` (BERT architecture), trained on Google Colab with a GPU.
- Learning rate tuned on validation over `2e-5`, `3e-5`, `5e-5`, up to 5 epochs each:

  | Learning rate | Best epoch | Validation macro-F1 |
  |---|---|---|
  | 2e-5 | 5 | 0.9736 |
  | **3e-5** | **4** | **0.9867** |
  | 5e-5 | 4 | 0.9845 |

- Results: synthetic test macro-F1 0.980, MINDS-14 macro-F1 0.916. Fine-tuned BERT is statistically
  tied with the frozen sentence-embedding model on both test sets.

## Commits

| Commit | Description |
|---|---|
| `1822fcf` | Add sentence embeddings + linear SVM notebook (03) |
| `b17d705` | Add shared finetune.py and BERT fine-tuning notebook; fix notebook 03 conclusion |
| `86360b9` | BERT fine-tuning: results on synthetic_test and minds14, predictions and confusion matrices |
