#Açıklama : Bu betik M1, M2 ve M0 veri kümeleriyle sırasıyla A, B ve C modellerini eğitir.
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer,AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
from huggingface_hub import login

# Hugging Face Hub'a giriş yap
login()
# Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Birçok dosyadan veri yüklemek ve işlemek için fonksiyon
def load_and_process_data(file_paths,filter_texts):
    lines = []
    tmplines = []
    for i,file_path in enumerate(file_paths):
        with open(file_path, "r") as file:
            tmplines = [line.strip(filter_texts[i]) for line in file.readlines()]
            size = len(tmplines)
            lines += tmplines[(size*99)//100:size]
    
    data = {"text": lines}
    dataset = Dataset.from_dict(data)
    dataset = DatasetDict({"train": dataset})
    return dataset


# Modeli eğitmek için fonksiyon
char_variable = 'A'
def train_model(pretrained_model_name, dataset_paths,filter_texts):
    global char_variable
    model_name = f"Turkish-GPT2-mini-Model{char_variable}-v2"
    char_variable = chr(ord(char_variable) + 1)
    
    # Veriyi yükle ve işle
    dataset = load_and_process_data(dataset_paths,filter_texts)
    dataset['train'] = dataset['train']

    total_length = len(dataset["train"])
    train_length = int(0.8 * total_length)
    dataset["validation"] = dataset["train"].select(range(train_length, total_length))
    dataset["train"] = dataset["train"].select(range(train_length))
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    CONTEXT_LENGTH = 1024
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
 
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
        output_dir=f"{model_name}",
        hub_model_id=model_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        optim="adamw_torch",
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
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
    trainer.push_to_hub(model_name)
    return model_name

# Model A=> M1+D2Sub
pretrained_model_name = "eminAydin/turkish-gpt2-mini-M1-cleaned-sports720k-10ep"
dataset_path = ["politics_737166_lines.txt"]
trained_model_name = train_model(pretrained_model_name, dataset_path,"__label__politics")
print(f"Trained model: {trained_model_name}")
# Model B=> M2+D1Sub
pretrained_model_name = "eminAydin/turkish-gpt2-mini-M2-cleaned-politics720k-10ep"
dataset_path = ["sports_843648_lines.txt"]
trained_model_name = train_model(pretrained_model_name, dataset_path,"__label__sports")
print(f"Trained model: {trained_model_name}")


# Model C=> M0+D1Sub+D2Sub
pretrained_model_name = "eminAydin/turkish-gpt2-mini-M0-cleaned-wiki720-10ep"
dataset_paths = ["sports_843648_lines.txt", "politics_737166_lines.txt"]
trained_model_name = train_model(pretrained_model_name, dataset_path,["__label__sports","__label__politics"])
print(f"Trained model: {trained_model_name}")
