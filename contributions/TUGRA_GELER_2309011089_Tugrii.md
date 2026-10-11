# Contribution: Tuğra Geler

- **Student ID:** 2309011089
- **GitHub:** [Tugrii](https://github.com/Tugrii)
- **Role:** embeddings vs baseline analysis, XLM-RoBERTa fine-tuning

## Summary

I analysed where and why the sentence-embedding model beats the TF-IDF baseline, and fine-tuned
XLM-RoBERTa with the shared fine-tuning code.

## 1. Embeddings vs baseline analysis (`notebooks/04_embeddings_analysis.ipynb`)

- **Per-intent comparison** of F1 for both models: the largest gain is on `abroad`
  (0.77 -> 1.00), and no intent got worse.
- **Error analysis** in three groups: wrong in both models (1 message), wrong only in TF-IDF (26),
  wrong only in the embedding model (5), with examples. Embeddings win on context and synonyms
  (e.g. "Lyon" meaning abroad, "plastic" meaning card), but not on typos.
- **Paraphrase and typo test** on 11 hand-written messages, predicted by both models.
- **t-SNE plot** of the test embeddings (`results/embeddings_2d.png`): most intents form separate
  clusters, and the neighbouring ones (`freeze` / `card_issues` / `abroad`) are the pairs that still cause errors.

## 2. XLM-RoBERTa fine-tuning (`notebooks/06_xlmr_finetune.ipynb`)

- Model: `xlm-roberta-base` (XLM-RoBERTa architecture), trained on Google Colab with a GPU,
  using `src/finetune.py` with the same procedure as BERT.
- Learning rate tuned on validation over `2e-5`, `3e-5`, `5e-5` (validation macro-F1 0.9756,
  0.9845, **0.9867**), batch size 16, maximum length 64, seed 42.
- Results: synthetic test macro-F1 0.982, MINDS-14 macro-F1 0.924, the best of the trained models on
  real data, but statistically tied with BERT and the sentence-embedding model.
- Comparison with BERT: accuracy, transfer from synthetic to real data, and training cost.

## Commits

| Commit | Description |
|---|---|
| `1a4ee57` | Add embedding vs baseline analysis, error analysis and t-SNE plot |
| `7c34880` | Add XLM-RoBERTa fine-tuning notebook, predictions, and metrics |
