# Description: Bu dosya, politika veri seti üzerinde eğitilmiş olan M2 modelini oluşturur ve eğitir.
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer,AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments,Trainer, EarlyStoppingCallback
from huggingface_hub import login

# Hugging Face Hub'a giriş yap
login()
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri setini oku ve "__label__health" etiketini kaldır
def load_and_process_data(file_path):
    # Satırları oku ve "__label__health" etiketini kaldır
    with open(file_path, "r") as file:
        lines = [line.strip("__label__politics") for line in file.readlines()]
    
    # Veri setini hazırla
    data = {"text": lines}  # İlk 50000 satırı al
    dataset = Dataset.from_dict(data)  # Dataset nesnesini oluştur

    # Dataset nesnesini DatasetDict içine yerleştirme
    dataset = DatasetDict({"train": dataset})  # DatasetDict oluştur
    return dataset

# Veri setini yükle ve işle
file_path = "politics_737166_lines.txt"  # Veri setinin dosya yolu
dataset = load_and_process_data(file_path)
dataset['train'] = dataset['train'].select(range(720000))
# Verisetinin uzunlugunu bulma
total_length = len(dataset["train"])
print(total_length)
# İlk %80'lik kesim için indeks aralığı
train_length = int(0.8 * total_length)
dataset["validation"] = dataset["train"].select(range(train_length, total_length))
dataset["train"] = dataset["train"].select(range(train_length))


tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
tokenizer.pad_token = tokenizer.eos_token
CONTEXT_LENGTH = 512
model = AutoModelForCausalLM.from_pretrained("eminAydin/turkish-gpt2-mini-M0-cleaned-wiki720-10ep")

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True
    )
    return outputs

tokenized_datasets = dataset.map(
    tokenize, batched=True, remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    hub_model_id="eminAydin/turkish-gpt2-mini-M2-cleaned-politics720k-10ep",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    optim="adamw_torch",
    auto_find_batch_size=True,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
)
trainer.train()
trainer.push_to_hub("eminAydin/turkish-gpt2-mini-M2-cleaned-politics720k-10ep")