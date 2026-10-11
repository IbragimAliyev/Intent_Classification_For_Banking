# Contribution: Mahmut Örnek

- **Student ID:** 220901545
- **GitHub:** [Mahmut-Ornek](https://github.com/Mahmut-Ornek)
- **Role:** Data preprocessing, MINDS-14 dataset preparation, bootstrap confidence intervals, final benchmark pipeline

---

## Summary

I designed and implemented the data preprocessing and near-duplicate cleaning pipeline, created the stratified train/validation/test splits, prepared and validated the real-world MINDS-14 spoken evaluation dataset, developed the statistical bootstrap module for confidence intervals and paired difference testing, and built the end-to-end multi-model benchmark suite comparing all five models across both synthetic and real-world banking test sets.

---

## 1. Preprocessing Pipeline (`src/preprocess.py`, `notebooks/01_preprocessing.ipynb`)

- **Text Normalization:** Applied Unicode NFKC normalization, standardized curly quotes and em/en dashes to ASCII, collapsed extra whitespace/line breaks while deliberately preserving casing, typos, punctuation, and currency symbols (`£`, `€`, `é`).
- **Near-Duplicate Detection & Vectorization:** Embedded all messages using `paraphrase-multilingual-MiniLM-L12-v2`. Optimized pair extraction using vectorized upper-triangle threshold masking (`threshold=0.85`) to avoid combinatoric $O(N^2)$ pandas overhead.
- **Duplicate Removal:** Filtered pairs with identical intent and cosine similarity $\ge 0.95$ (dropped later occurrences, reducing the synthetic dataset from 3,010 to 3,007 rows).
- **Cross-Intent Sanity Check:** Monitored cross-intent pairs with similarity $\ge 0.90$ and saved them to `results/cross_intent_similar_pairs.csv` for manual inspection (0 critical label conflicts found).
- **Stratified Split & Data Funnel:** Created a stratified 70 / 15 / 15 split (`seed=42`) yielding 2,104 train, 451 val, and 452 test rows (`data/processed/`). Tracked dataset sizing across all pipeline stages in `results/data_funnel.csv`.

---

## 2. MINDS-14 Evaluation Dataset (`notebooks/07_minds14.ipynb`)

- **Dataset Acquisition & Filtering:** Loaded the English configurations (`en-US`, `en-GB`, `en-AU`) from `PolyAI/minds14` (1,809 customer voice transcriptions), dropped audio arrays, converted class indices to text names, and verified an exact 14-intent match with our synthetic label set.
- **Data Export & Imbalance Analysis:** Saved `data/processed/minds14_en.csv` (`text, intent, locale`). Documented natural class imbalance (115–138 samples per intent), providing empirical justification for prioritizing Macro-F1 over standard accuracy.
- **Initial Baseline Scoring:** Evaluated the saved `tfidf_lr` (Macro-F1: 0.8640) and `sbert_svm` (Macro-F1: 0.9170) models on MINDS-14. Saved prediction files to `results/predictions/` and exported confusion matrices to `results/confusion_*.png`.

---

## 3. Bootstrap Confidence Intervals (`src/bootstrap.py`)

- **`bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42)`:** Computes 95% bootstrap confidence intervals for Macro-F1 via 1,000 resamples with replacement, deriving the 2.5th and 97.5th percentiles.
- **`paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot=1000, seed=42)`:** Evaluates $\Delta = \text{Macro-F1}_A - \text{Macro-F1}_B$ using **identical resampled indices** per iteration for both models. Provides formal statistical significance testing ($p < 0.05$ if the interval excludes 0).

---

## 4. Final Benchmark Suite (`notebooks/09_benchmark.ipynb`)

- **Dynamic Pipeline Architecture:** Automatically discovers and parses all 12 prediction artifacts (`results/predictions/`) across all six project models (`tfidf_lr`, `tfidf_nb`, `sbert_svm`, `bert_base`, `xlmr_base`, `jev`) and both test datasets (`synthetic_test`, `minds14`).
- **Comprehensive Benchmark Tables:** Produced sorted benchmark tables detailing model type (baseline vs. new), architecture, specific checkpoint, accuracy, Macro-F1, and 95% CIs.
- **Visualization:** Generated a publication-ready grouped bar chart (`results/benchmark.png`, 300 DPI) displaying Macro-F1 across datasets side-by-side with asymmetric 95% bootstrap CI error bars.
- **Key Empirical Findings & Critical Discussion:**
  - **Synthetic Test Saturation:** Top models peaked around ~0.989 Macro-F1 with narrow CIs on synthetic data, proving test saturation and demonstrating that synthetic evaluation alone overestimates real-world generalization.
  - **Real-World Domain Shift:** All local models experienced a noticeable drop on MINDS-14 (ranging from 5.8 to 11.1 percentage points in Macro-F1, with `xlmr_base` dropping the least at 5.77 pp and `tfidf_nb` dropping the most at 11.1 pp), whereas commercial zero-shot API `jev` showed superior resilience (only a 3.8 pp drop, leading MINDS-14 at 0.9510).
  - **XLM-R vs. BERT Comparison:** `xlmr_base` scored slightly higher than `bert_base` on MINDS-14 (0.9240 vs. 0.9160), but the 95% paired bootstrap difference interval includes zero ([-0.019, +0.001]), confirming that the difference is not statistically significant.
  - **Statistical Ties:** Proved that the Macro-F1 difference between `sbert_svm` (0.9170) and fine-tuned `bert_base` (0.9160) on MINDS-14 also includes zero in its 95% paired bootstrap interval, confirming the two models are statistically tied ($p \ge 0.05$).

---

## Commits

| Commit | Description |
|---|---|
| `e29504e` | Implement preprocessing pipeline, near-duplicate filtering, and stratified splits |
| `006f54c` | Prepare minds14_en dataset and add notebook 07 |
| `27c9cb7` | Add minds14 evaluation, bootstrap confidence intervals, and benchmark notebook |
| `3176b55` | Update benchmark notebook and results/benchmark.png for all 5 models |
