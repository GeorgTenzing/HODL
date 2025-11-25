from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import torch.nn as nn
import transformers
import random

# ======================================================
# CONFIGURATION — flip these switches
# ======================================================
BASE_MODEL = "roberta-large"   # "bert-base-uncased", "roberta-base", "roberta-large" 1ep= 9.5, "microsoft/deberta-v3-base"
ARCHITECTURE = "bilstm"        # "base" or "bilstm"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


# ======================================================
# PREPROCESSING
# ======================================================
def augment_text(t):
    words = t.split()
    if len(words) > 10 and random.random() < 0.15:
        del words[random.randint(0, len(words) - 1)]
    return " ".join(words)


def preprocess_function(examples):
    texts = []
    for t in examples["text"]:
        t = t.replace("<br />", " ").strip()
        if random.random() < 0.3:
            t = augment_text(t)
        texts.append(t)

    # Smart crop: keep start and end of long reviews
    def smart_crop(x):
        tokens = tokenizer.tokenize(x)
        if len(tokens) > 512:
            half = 256
            tokens = tokens[:half] + tokens[-half:]
            x = tokenizer.convert_tokens_to_string(tokens)
        return x

    texts = [smart_crop(t) for t in texts]
    return tokenizer(texts, truncation=True, padding="longest", max_length=512)


# ======================================================
# MODEL DEFINITIONS
# ======================================================

# --- Transformer + BiLSTM hybrid ---
class TransformerBiLSTMClassifier(nn.Module):
    def __init__(self, model_name, lstm_hidden=128, lstm_layers=1, dropout=0.3): 
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, 2),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L, H]

        lstm_out, _ = self.lstm(sequence_output)     # [B, L, 2*lstm_hidden]
        pooled = torch.mean(lstm_out, dim=1)         # mean pooling

        logits = self.classifier(pooled)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


# ======================================================
# INIT MODEL
# ======================================================
def init_model() -> nn.Module:
    if ARCHITECTURE == "bilstm":
        model = TransformerBiLSTMClassifier(BASE_MODEL)
    else:
        raise ValueError(f"Unknown ARCHITECTURE={ARCHITECTURE}")

    # optional: enable gradient checkpointing for large models
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    return model


# ======================================================
# TRAINING FUNCTION
# ======================================================
def train_model(model: nn.Module, dev_dataset: torch.utils.data.Dataset) -> nn.Module:
    # # Dynamic batch sizing for large models
    # bs = 8 if "large" in BASE_MODEL else 16
    # accum = 8 if "large" in BASE_MODEL else 4

    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     eval_strategy="no",
    #     save_strategy="epoch",
    #     save_total_limit=1,
    #     learning_rate=3e-5,
    #     per_device_train_batch_size=bs,
    #     gradient_accumulation_steps=accum,
    #     num_train_epochs=2,
    #     weight_decay=0.01,
    #     warmup_ratio=0.05,
    #     lr_scheduler_type="cosine",
    #     fp16=True,
    #     logging_strategy="epoch",
    #     disable_tqdm=True,
    #     report_to="none",
    #     max_grad_norm=1.0,
    # )

    # Detect encoder module (bert / roberta / deberta)
    # if hasattr(model, "encoder"):
    #     encoder = model.encoder
    # else:
    
    encoder = model




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dev_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.args.learning_rate = 1e-5
    trainer.train(resume_from_checkpoint=True)
    return model
