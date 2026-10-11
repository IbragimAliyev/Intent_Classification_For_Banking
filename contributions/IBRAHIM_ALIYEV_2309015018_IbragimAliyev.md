# Contribution: Ibrahim Aliyev

- **Student ID:** 2309015018
- **GitHub:** [IbragimAliyev](https://github.com/IbragimAliyev)
- **Role:** synthetic dataset, evaluation, TF-IDF baseline, Jev API ("new" method), repository setup and documentation

## Summary

I created the synthetic dataset, set up the repository, wrote the shared evaluation function that
every model in the project uses, built the TF-IDF + logistic regression baseline, and evaluated the
Jev API as our "new" method on both test sets. I also wrote the metric justification, the README and
the dependency list, and corrected the conclusion of the embeddings analysis notebook.

## 1. Synthetic dataset and repository setup

- **Created the synthetic dataset** `data/raw/synthetic_raw.csv` with Claude (Anthropic), in several rounds:
  - **Labels:** the 14 intents of MINDS-14, so the data could be tested on real customer calls
    without relabeling. I wrote a definition for each intent, plus rules for the easily confused
    pairs (e.g. a stolen card is `freeze` only if the main request is to block it; a card failing
    abroad is `abroad`, not `card_issues`).
  - **Size:** 215 messages per intent, 3,010 in total: perfectly balanced, and well above the
    required minimum of 1,000 rows even after cleaning.
  - **Diversity rules in the prompt:** about 60% typed chat and 40% spoken phone-call style
    (fillers, restarts, no punctuation), because MINDS-14 is transcribed speech; neutral, polite,
    frustrated and urgent tones; lengths from 2 words to 3 sentences; different personas (students,
    elderly people, business owners, travellers, parents); typos and all-lowercase messages; US, UK
    and Australian English with matching spelling and currency; about 20% indirect phrasing without
    the obvious keyword; varied openings and realistic invented details; never the intent name in the text.
  - **Pilot first:** 10 messages per intent, reviewed, and one problem fixed (too many spoken
    messages started with "yeah so") before generating the full dataset.
  - **Checks afterwards:** class balance, duplicates, label correctness and formatting, plus a
    quick baseline to confirm the data was neither broken nor trivially easy.
- Created the repository structure (`data/`, `src/`, `notebooks/`, `results/`, `report/`) and
  `.gitignore` (virtual environments, `models/`, `.env`, model weight files).
- Created `requirements.txt` and later added the missing dependencies for fine-tuning
  (`torch`, `transformers`, `accelerate`, `sentencepiece`) and for the Jev API (`requests`, `python-dotenv`).

## 2. Shared evaluation function (`src/evaluate.py`)

`evaluate(y_true, y_pred, model_name, dataset_name, out_dir="results")` scores every model in
the project in the same way, so results are directly comparable:

- computes accuracy and macro-F1, and prints a per-intent classification report
- saves a confusion matrix as `results/confusion_<model>_<dataset>.png`, with labels sorted alphabetically
- appends one row to `results/metrics.csv` (created with a header on first use)
- returns the scores as a dictionary

Design decisions: the label list is the sorted union of true and predicted labels, so nothing is
hard-coded and the same function works for MINDS-14; `zero_division=0` handles intents a model
never predicts. I tested it on dummy labels before the team used it.

## 3. TF-IDF + logistic regression baseline (`notebooks/02_tfidf_baseline.ipynb`)

- Pipeline: `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)` + `LogisticRegression`.
- Tuned the regularization parameter C on the validation set over `[0.01, ..., 10000]`. The first
  grid ended at the best value (C = 100), so I extended it to show that validation macro-F1 peaks at
  C = 100 (0.9617) and drops for larger C (overfitting).
- Test result: accuracy 0.9403, macro-F1 0.9386. Weakest intent: `abroad` (F1 0.77), confused with
  `card_issues`, `freeze` and `atm_limit` because they share vocabulary.

## 4. Jev API as a "new" method (`notebooks/08_jev.ipynb`)

Jev (TypeSafe AI, `jev-latest`, version `jev-1.13.0` for all 3,634 requests) is a zero-shot
classification API: it is never trained on our data and only receives the 14 intents as options.

- **Client code:** one function per request with retries and exponential backoff for rate limits
  and server errors, and a batch function that saves every answer to a `.jsonl` file immediately,
  so an interrupted run resumes without paying for the same messages twice. The API key is read
  from `.env` and never committed.
- **Option design chosen on validation only:**

  | Design | Validation macro-F1 |
  |---|---|
  | A: intent names only | 0.9564 |
  | B: names + one-sentence descriptions | 0.9890 |
  | C: B with two descriptions corrected | **0.9934** (chosen) |

  The validation errors of design B pointed to two incomplete descriptions. I checked them against
  the training data: 52 of the 150 `address` messages are about changing a phone number or email,
  not a postal address, and `abroad` also covers foreign-currency fees. Design C corrects these two
  descriptions. I stopped there to avoid overfitting the validation set.
- **Final results (design C, each test set used once):** synthetic test macro-F1 0.989, MINDS-14
  macro-F1 0.951, the best of all five models on real data. Paired bootstrap intervals show it is
  clearly better than every other model on MINDS-14, and tied with the best models on the synthetic test.
- **Analysis:** synthetic-to-real drop (3.8 points vs 5.8-7.5 for the trained models), the most
  common MINDS-14 errors and their causes (e.g. speech recognition turning "business loan" into
  "business phone"), confidence analysis (answers with confidence >= 0.9 are 98.4% correct on
  MINDS-14, the rest 60%), latency (0.46 s per message) and cost (about $0.10 for all requests).
- **Saved outputs:** predictions in the shared format (`results/predictions/jev_*.csv`), all
  probabilities (`results/jev/jev_probabilities_*.csv`), the final descriptions
  (`results/jev/final_criteria.json`) and every raw API answer (`results/jev/raw/`), so the results
  can be verified without calling the API again.

## 5. Documentation and review

- `report/metric_justification.md`: why macro-F1 is our main metric, why we also report accuracy,
  why micro-F1 and AUC are not used, and why we add bootstrap confidence intervals.
- `README.md`: project overview, results table, data, structure, notebooks, how to run, conventions.
- Reviewed `notebooks/04_embeddings_analysis.ipynb` and rewrote its conclusion and error analysis so
  they match the notebook's own outputs (the original text claimed the embedding model handles typos
  better, which the results did not support).

## Commits

| Commit | Description |
|---|---|
| `e61a8a3` | Initial project structure and synthetic dataset |
| `0b1ed1a` | Added requirements.txt |
| `347ed96` | Add shared evaluate() function |
| `fcb487b` | Add TF-IDF + logistic regression baseline |
| `c5ba24b` | Fix notebook 04 conclusion and error analysis to match results |
| `591fa8c` | Add Jev API evaluation: design selection on validation, synthetic test and MINDS-14 results, analysis |
| `140363a` | Add evaluation metric justification |
