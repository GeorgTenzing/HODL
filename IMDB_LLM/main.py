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
from datasets import load_dataset

# ======================================================
# CONFIGURATION
# ======================================================
BASE_MODEL = "roberta-large"          
EPOCHS = 2                            
MAX_LENGTH = 512   
LR = 2e-5
LR_ENCODER = 2e-5                     
LR_HEAD = 1e-3        
WARMUP_RATIO = 0.05                 
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 8 
ACCUMULATION_STEPS = 8


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
dataset = load_dataset("imdb")

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_function(examples):
    # Text augmentation: randomly delete words
    def augment_text(t):
        words = t.split()
        if len(words) > 10 and random.random() < 0.15:
            del words[random.randint(0, len(words) - 1)]
        return " ".join(words)
    
    # Smart crop: keep start and end of long reviews
    def smart_crop(t):
        tokens = tokenizer.tokenize(t)
        if len(tokens) > MAX_LENGTH:
            half = MAX_LENGTH // 2
            tokens = tokens[:half] + tokens[-half:]
            t = tokenizer.convert_tokens_to_string(tokens)
        return t

    texts = []
    for t in examples["text"]:
        t = t.replace("<br />", " ").strip()
        if random.random() < 0.3:
            t = augment_text(t)
        texts.append(t)
        
    texts = [smart_crop(t) for t in texts]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])


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
        sequence_output = outputs.last_hidden_state 

        lstm_out, _ = self.lstm(sequence_output)     
        pooled = torch.mean(lstm_out, dim=1)         

        logits = self.classifier(pooled)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}




# ======================================================
# MODEL INITIALIZATION 
# ======================================================
def init_model() -> nn.Module:
    model = TransformerBiLSTMClassifier(BASE_MODEL)

    # optional: enable gradient checkpointing for large models
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    return model

def optimizer_model(model):
    encoder = model.roberta

    # Separate encoder vs. head params cleanly
    encoder_params = list(encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    head_params = [p for n, p in model.named_parameters() if id(p) not in encoder_param_ids]
    
    optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": LR_ENCODER},
            {"params": head_params, "lr": LR_HEAD},],
    )

    return optimizer
    
# ======================================================
# TRAINING FUNCTION
# ======================================================
def train_model(model: nn.Module, dev_dataset: torch.utils.data.Dataset) -> nn.Module:

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR_ENCODER,
        #warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,

        lr_scheduler_type="cosine",
        fp16=True,

        output_dir="./results",
        save_strategy="epoch",
        save_total_limit=1,
        
        eval_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(20000)),  # subset for ≤15 min
        eval_dataset=tokenized_datasets["test"].select(range(4000)),
        tokenizer=tokenizer,
        optimizers=(optimizer_model(model), None),
    )

    trainer.train()
    return model
