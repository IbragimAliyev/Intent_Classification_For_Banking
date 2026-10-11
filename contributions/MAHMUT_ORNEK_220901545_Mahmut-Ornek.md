<!-- DRAFT written from the git history. Mahmut: please check, correct and add details, then delete this line. -->

# Contribution: Mahmut Örnek

- **Student ID:** 220901545
- **GitHub:** [Mahmut-Ornek](https://github.com/Mahmut-Ornek)
- **Role:** data preprocessing, MINDS-14 preparation, confidence intervals, final benchmark

## Summary

I built the preprocessing pipeline that cleans the synthetic dataset and creates the
train / validation / test split, prepared the real MINDS-14 test set, wrote the bootstrap code
for confidence intervals, and built the final benchmark that compares all five models.

## 1. Preprocessing pipeline (`src/preprocess.py`, `notebooks/01_preprocessing.ipynb`)

- **Normalization:** Unicode NFKC, curly quotes and long dashes replaced with ASCII, line breaks
  and repeated whitespace collapsed. Casing and punctuation are kept on purpose.
- **Near-duplicate detection:** every message is embedded with
  `paraphrase-multilingual-MiniLM-L12-v2`, and all pairs with cosine similarity >= 0.85 are reported.
- **Removal:** pairs with the same intent and similarity >= 0.95 are treated as duplicates, and the
  later message is dropped (3,010 -> 3,007 rows).
- **Cross-intent check:** pairs with different intents and similarity >= 0.90 are saved to
  `results/cross_intent_similar_pairs.csv` for manual review (none were found).
- **Split:** stratified 70 / 15 / 15 split with seed 42 (2,104 / 451 / 452 rows), saved to
  `data/processed/`, plus a row count after every stage in `results/data_funnel.csv`.

## 2. MINDS-14 test set (`notebooks/07_minds14.ipynb`)

- Loaded the English configurations `en-US`, `en-GB` and `en-AU` of `PolyAI/minds14` (1,809
  transcribed customer calls), converted the intent numbers to names and verified that the
  14 names match ours exactly.
- Saved `data/processed/minds14_en.csv` (`text, intent, locale`) and reported counts per intent and
  locale (115-138 messages per intent).
- Scored the TF-IDF and sentence-embedding models on it: macro-F1 0.864 and 0.917.

## 3. Bootstrap confidence intervals (`src/bootstrap.py`)

- `bootstrap_ci`: 95% confidence interval for macro-F1 from 1,000 resamples of the test set (seed 42).
- `paired_bootstrap_diff`: 95% interval for the difference between two models, using the same
  resampled messages for both. If it excludes 0, the difference is unlikely to be chance.

## 4. Final benchmark (`notebooks/09_benchmark.ipynb`)

- Finds every prediction file in `results/predictions/` automatically and validates its format.
- One table per dataset with type, architecture, specific model, accuracy, macro-F1 and 95% CI.
- Paired comparisons between the important model pairs, and the chart `results/benchmark.png`.

## Commits

| Commit | Description |
|---|---|
| `e29504e` | Implement preprocessing pipeline, near-duplicate filtering, and stratified splits |
| `006f54c` | Prepare minds14_en dataset and add notebook 07 |
| `27c9cb7` | Add minds14 evaluation, bootstrap confidence intervals, and benchmark notebook |
| `3176b55` | Update benchmark notebook and results/benchmark.png for all 5 models |
