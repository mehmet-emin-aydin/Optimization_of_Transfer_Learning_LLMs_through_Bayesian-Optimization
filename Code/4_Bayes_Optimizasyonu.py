# Açıklama: Bayes Optimizasyonu ile model birleştirme işlemi yapılır ve başarı skorları değerlendirilir.
import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

# Model ve tokenizer'ı yükle
tokenizer = GPT2Tokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2-large")
tokenizer.pad_token = tokenizer.eos_token

def next_word_stepwise_accuracy(text, model, tokenizer, device='cpu'):

    # Metni tokenizer ile tokenlara ayır
    tokens = tokenizer.tokenize(text)
    
    top1_accuracies = []
    top5_accuracies = []
    
    for i in range(1, len(tokens)):
        input_tokens = tokens[:i]
        true_next_token = tokens[i]
        
        # Tokenları birleştirip giriş metnini oluştur
        input_text = tokenizer.convert_tokens_to_string(input_tokens)
        
        # Metni token haline getir
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(device)
        model = model.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Son token için tahmin yap
        next_token_logits = logits[0, -1, :]
        predictions = torch.softmax(next_token_logits, dim=-1)
        top_5_tokens = torch.topk(predictions, 5).indices
        
        # Gerçek tokenın ID'sini al
        true_next_token_id = tokenizer.convert_tokens_to_ids(true_next_token)
        
        top1_correct = (true_next_token_id == top_5_tokens[0].item())
        top5_correct = (true_next_token_id in top_5_tokens)
        
        top1_accuracy = 1.0 if top1_correct else 0.0
        top5_accuracy = 1.0 if top5_correct else 0.0
        
        top1_accuracies.append(top1_accuracy)
        top5_accuracies.append(top5_accuracy)

    return top1_accuracies, top5_accuracies

def evaluate_merged_models(model_extensions, tokenizer, datasets, U_values):

    fig, axs = plt.subplots(len(datasets), len(model_extensions) * len(U_values), figsize=(20, 12))
    
    for i, dataset in enumerate(datasets):
        for j, model_extension in enumerate(model_extensions):
            for k, U in enumerate(U_values):
                merged_model_top1_accuracies = []
                merged_model_top5_accuracies = []
                
                # M1 ve M2 modellerini yükle
                model_M1 = GPT2LMHeadModel.from_pretrained(model_extension[0])
                model_M2 = GPT2LMHeadModel.from_pretrained(model_extension[1])
                
                for sentence in dataset:
                    total_top1_accuracies_M1, total_top5_accuracies_M1 = next_word_stepwise_accuracy(sentence, model_M1, tokenizer)
                    total_top1_accuracies_M2, total_top5_accuracies_M2 = next_word_stepwise_accuracy(sentence, model_M2, tokenizer)
                    
                    # Birleştirilmiş modeli oluştur
                    merged_model_top1_accuracies.extend(np.array(total_top1_accuracies_M1) * U + np.array(total_top1_accuracies_M2) * (1 - U))
                    merged_model_top5_accuracies.extend(np.array(total_top5_accuracies_M1) * U + np.array(total_top5_accuracies_M2) * (1 - U))
                
                # Ortalama doğrulukları hesapla
                average_merged_top1_accuracy = sum(merged_model_top1_accuracies) / (len(merged_model_top1_accuracies)+1)
                average_merged_top5_accuracy = sum(merged_model_top5_accuracies) / (len(merged_model_top5_accuracies)+1)
                
                # Skorları yazdır
                ax = axs[i, j * len(U_values) + k]
                ax.bar(["Top-1", "Top-5"], [average_merged_top1_accuracy, average_merged_top5_accuracy], color=["#ff0000", "#00ff00"])
                ax.set_title(f"Dataset {i+1}, Model {j+1}, U={U}")
                ax.set_ylim([0, 1])
                ax.set_ylabel("Accuracy")
                
                # Accuracy değerlerini blokların üzerine yazdır
                for index, value in enumerate([average_merged_top1_accuracy, average_merged_top5_accuracy]):
                    ax.text(index, value + 0.05, f"{value:.2f}", ha='center')
    
    # Dosyayı kaydet
    plt.tight_layout()
    plt.savefig("merged_models_evaluation-0.125-0.875.png")
    plt.close()

# Örnek model uzantıları listesi (M1 ve M2 modelleri)
model_extensions = ["eminAydin/turkish-gpt2-mini-M1-cleaned-sports720k-10ep", "eminAydin/turkish-gpt2-mini-M2-cleaned-politics720k-10ep"]


# Veri setini yükle ve her satır için işlem yap
def load_dataset(file_path, start_index, filter):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip(filter) for line in file.readlines()]

    return [line.strip() for line in lines[start_index:start_index+1000]]


# Örnek veri seti
dataset_paths = [
    'sports_843648_lines.txt',
    'politics_737166_lines.txt'
]
datasets = [
    load_dataset(dataset_paths[0], 800000, "__label__sports"),
    load_dataset(dataset_paths[1], 730000, "__label__politics")
]

# U değerlerini belirle
U_values = list(np.linspace(0.125, 0.875, 7))

# Modelleri değerlendir ve görselleştir
evaluate_merged_models(model_extensions, tokenizer, datasets, U_values)
