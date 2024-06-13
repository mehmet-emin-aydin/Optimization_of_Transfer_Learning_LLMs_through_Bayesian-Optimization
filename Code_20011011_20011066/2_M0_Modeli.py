
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


# Veriyi yükle ve işle
def load_and_process_data(file_path):
    with open(file_path, "r") as file:
        lines = [line for line in file.readlines()]
    
    data = {"text": lines}
    dataset = Dataset.from_dict(data)
    dataset = DatasetDict({"train": dataset})
    return dataset

def remove_none(example):
    return example['text'] is not None and example['text'] != ''

file_path = "trwiki-67-train-cleaned.txt"
dataset = load_and_process_data(file_path)

dataset["train"] = dataset["train"]
dataset = dataset.filter(remove_none)

total_length = len(dataset["train"])
train_length = int(0.8 * total_length)
dataset["validation"] = dataset["train"].select(range(train_length, total_length))
dataset["train"] = dataset["train"].select(range(train_length))

CONTEXT_LENGTH = 512
# 
# Tokenizer yükleme
tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-gpt2')
tokenizer.pad_token = tokenizer.eos_token
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True
    )
    return outputs

tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# Tokenize edilmiş veriyi kaydet
tokenized_datasets.save_to_disk('tokenized_datasets')
print("Tokenize edilmiş veri kaydedildi.")



###Daha sonra ise tokenize edilmiş veriyi okuyup modele girdi olarak verebiliriz. 
###Böylelikle eğitim yaptığımız oturumda tokenize etme kısmı vakit almaz ve böylelikle zamandan tasarruf.

import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
from huggingface_hub import login

# Hugging Face Hub'a giriş yap
login()
# GPU'yu temizle
torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tokenize edilmiş veriyi yükle
tokenized_datasets = load_from_disk('tokenized_datasets')
print("Tokenize edilmiş veri yüklendi.")

# Tokenizer yükleme
tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-gpt2')
tokenizer.pad_token = tokenizer.eos_token
# Model konfigürasyonu ve oluşturma
config = GPT2Config.from_pretrained('ytu-ce-cosmos/turkish-gpt2')
config.bos_token_id = 0
config.eos_token_id = 0
config.pad_token_id = 0
config.vocab_size = tokenizer.vocab_size
config.n_embd = 384
config.n_head = 6

model = GPT2LMHeadModel(config)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Eğitim argümanları
training_args = TrainingArguments(
    output_dir="eminAydin/turkish-gpt2-mini-M0-cleaned-wiki720-10ep",
    hub_model_id="eminAydin/turkish-gpt2-mini-M0-cleaned-wiki720-10ep",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    optim="adamw_torch",
    auto_find_batch_size=True,
    num_train_epochs=10,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    learning_rate=5e-5,
    fp16=True,
    push_to_hub=True,
    logging_steps=10,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    save_total_limit=2,
    save_steps=50000,
    load_best_model_at_end=True,
    report_to="tensorboard",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_safetensors=True,
)

# Trainer oluşturma
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.003)]
)

# Eğitim
trainer.train()

# Modeli hub'a yükleme
trainer.push_to_hub("eminAydin/turkish-gpt2-mini-M0-cleaned-wiki720-10ep")