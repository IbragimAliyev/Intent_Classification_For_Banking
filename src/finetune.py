"""Shared fine-tuning code for the transformer baselines.

Used by notebooks/05_bert_finetune.ipynb (bert-base-uncased) and
notebooks/06_xlmr_finetune.ipynb (xlm-roberta-base), so both models are
trained in exactly the same way and the comparison between them is fair.

Nothing in here is model-specific: the model name is an argument.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

MAX_LENGTH = 64
SEED = 42


class IntentDataset(torch.utils.data.Dataset):
    """Minimal torch dataset holding tokenized messages and integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def build_label_maps(intents):
    """Label order comes from the SORTED intent names.

    Both models must use the same order, otherwise the saved predictions and
    confusion matrices are not comparable.
    """
    labels = sorted(set(intents))
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}
    return labels, label2id, id2label


def compute_metrics(eval_pred):
    """Macro-F1 is the metric we select on, matching the rest of the project."""
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(label_ids, preds),
        "macro_f1": f1_score(label_ids, preds, average="macro"),
    }


def _make_trainer(model, args, train_ds, val_ds, tokenizer):
    """Trainer takes the tokenizer under different names across versions."""
    try:
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
    except TypeError:
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )


def _make_args(output_dir, learning_rate, num_epochs, batch_size, seed):
    """TrainingArguments renamed evaluation_strategy to eval_strategy."""
    common = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def finetune(
    model_name,
    train_df,
    val_df,
    learning_rate,
    num_epochs,
    batch_size=16,
    max_length=MAX_LENGTH,
    seed=SEED,
    output_dir=None,
):
    """Fine-tune a Hugging Face model and return (trainer, label list).

    The model is trained on train_df only. val_df is evaluated after every
    epoch, and the epoch with the best validation macro-F1 is the one loaded
    back into the trainer at the end. The test set is never touched here.
    """
    set_seed(seed)
    labels, label2id, id2label = build_label_maps(train_df["intent"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    def encode(df):
        enc = tokenizer(
            df["text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        y = [label2id[name] for name in df["intent"]]
        return IntentDataset(enc, y)

    if output_dir is None:
        tag = model_name.replace("/", "_")
        output_dir = f"./runs/{tag}_lr{learning_rate}"

    args = _make_args(output_dir, learning_rate, num_epochs, batch_size, seed)
    trainer = _make_trainer(model, args, encode(train_df), encode(val_df), tokenizer)
    trainer.train()
    return trainer, labels


def predict_intents(trainer, texts, labels, max_length=MAX_LENGTH):
    """Predict intent names for a list of messages using a fine-tuned trainer."""
    tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
    enc = tokenizer(
        [str(t) for t in texts],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    dummy = IntentDataset(enc, [0] * len(enc["input_ids"]))
    logits = trainer.predict(dummy).predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return [labels[i] for i in np.argmax(logits, axis=-1)]


def best_epoch_score(trainer):
    """Return (best epoch number, best validation macro-F1) from the log history."""
    scores = [
        (entry["epoch"], entry["eval_macro_f1"])
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]
    if not scores:
        return None, None
    epoch, score = max(scores, key=lambda pair: pair[1])
    return int(round(epoch)), score
