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
BATCH_SIZE = 8 #8
ACCUMULATION_STEPS = 8


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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
    return tokenizer(texts, truncation=True, padding="longest", max_length=MAX_LENGTH)

def preprocess_function(examples):
    
    texts = list(examples["text"])
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",   # safer and more consistent than "longest"
        max_length=MAX_LENGTH
    )
# ======================================================
# MODEL INITIALIZATION — SIMPLIFIED ROBERTA CLASSIFIER
# ======================================================

def init_model() -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2
    )

    model.gradient_checkpointing_enable()
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
        train_dataset=dev_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer_model(model), None),
    )

    trainer.train()
    return model
