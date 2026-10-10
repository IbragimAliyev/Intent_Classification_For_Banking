# Choice of evaluation metric

Our main metric is **macro-F1**. We also report **accuracy**, and we add **95% bootstrap
confidence intervals** to both benchmarks. This section explains why.

## 1. Our task and data

- **Task:** single-label classification. Every message belongs to exactly one of 14 intents.
- **Synthetic data:** balanced by design. The test set has 32-33 messages per intent (452 in total).
- **MINDS-14 (real calls):** close to balanced, with 115 to 138 messages per intent (1,809 in total).
- **Every intent matters equally.** For a bank, a customer whose card is blocked abroad is as
  important as one asking for their balance, even if one type of request is rarer.

## 2. Why macro-F1 is the main metric

Macro-F1 computes the F1 score for each intent separately and then takes the plain average.
Each of the 14 intents counts the same, regardless of how many messages it has.

- If a model handles one intent badly, macro-F1 drops noticeably. Accuracy can hide this,
  because the many correct answers on the other intents dominate it.
- F1 combines **precision** (when the model predicts an intent, is it right?) and **recall**
  (does it find the messages of that intent?). Both kinds of error matter here: a missed
  `freeze` request and a wrongly frozen card are both bad outcomes.

## 3. Why we still report accuracy

On balanced data, accuracy and macro-F1 are almost the same. For example, the TF-IDF
baseline scores 0.9403 accuracy and 0.9386 macro-F1 on the synthetic test set. Accuracy is easy
to explain ("94% of messages were classified correctly"), so we report it next to macro-F1.
Macro-F1 stays the main metric because it remains reliable if the data becomes less balanced,
which is more likely with real data such as MINDS-14.

## 4. Why not micro-F1

For single-label classification, micro-F1 is mathematically **identical to accuracy**: every
wrong prediction is at the same time one false positive (for the predicted intent) and one
false negative (for the true intent), so micro-precision, micro-recall and accuracy are all equal.
Reporting micro-F1 would add nothing.

## 5. Why not AUC

AUC measures how well a model *ranks* messages by its probability scores, so it needs a
probability for every intent from every model. Not all our models provide comparable
probabilities: the linear SVM (`sbert_svm`) only gives decision scores, not probabilities, and
Jev's probabilities come from a different kind of model. We therefore rank all models by
macro-F1, and analyse Jev's confidence separately (see `notebooks/08_jev.ipynb`).

## 6. Uncertainty: bootstrap confidence intervals

The test sets are small: with 32 messages per intent in the synthetic test set, one message
changes an intent's F1 by about 3 points. A difference between two models may therefore be
chance. For each model we compute a **95% bootstrap confidence interval** for macro-F1
(1,000 resamples of the test set, seed 42), and for pairs of models a **paired bootstrap** on the
same resampled messages. If the interval of the difference does not include 0, we consider the
difference real (`src/bootstrap.py`, `notebooks/09_benchmark.ipynb`).

## Summary

| Metric | Used? | Reason |
|---|---|---|
| **Macro-F1** | **main metric** | every intent counts equally; drops when one intent fails |
| Accuracy | reported | easy to explain; almost equal to macro-F1 on balanced data |
| Micro-F1 | no | identical to accuracy for single-label classification |
| AUC | no | not all models provide comparable probabilities |
| Bootstrap 95% CI | yes | test sets are small; shows whether differences are real |
