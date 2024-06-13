import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import numpy as np

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

def plot_heatmap(accuracies_matrix, title, save_path, model_extensions):
    plt.figure(figsize=(10, 6))
    plt.imshow(accuracies_matrix, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Accuracy')
    plt.xticks(np.arange(len(model_extensions)), [model_extension.split('mini-')[-1] for model_extension in model_extensions], rotation=45)
    plt.yticks(np.arange(len(datasets)), [f'Dataset {i}' for i in range(len(datasets))])
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel('Datasets')

    # Accuracy değerlerini hücrelerin içine yazma
    for i in range(len(datasets)):
        for j in range(len(model_extensions)):
            plt.text(j, i, f'{accuracies_matrix[i, j]:.3f}', ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_models(model_extensions, datasets, tokenizer, save_path):
    top1_accuracies_matrix = []
    top5_accuracies_matrix = []

    for dataset in datasets:
        top1_row_accuracies = []
        top5_row_accuracies = []
        for model_extension in model_extensions:
            model = GPT2LMHeadModel.from_pretrained(model_extension)
            total_top1_accuracies = []
            total_top5_accuracies = []

            for sentence in dataset:
                top1_accuracies, top5_accuracies = next_word_stepwise_accuracy(sentence, model, tokenizer)
                total_top1_accuracies.extend(top1_accuracies)
                total_top5_accuracies.extend(top5_accuracies)

            average_top1_accuracy = sum(total_top1_accuracies) / len(total_top1_accuracies)
            average_top5_accuracy = sum(total_top5_accuracies) / len(total_top5_accuracies)

            top1_row_accuracies.append(average_top1_accuracy)
            top5_row_accuracies.append(average_top5_accuracy)

        top1_accuracies_matrix.append(top1_row_accuracies)
        top5_accuracies_matrix.append(top5_row_accuracies)

    top1_accuracies_matrix = np.array(top1_accuracies_matrix)
    top5_accuracies_matrix = np.array(top5_accuracies_matrix)

    plot_heatmap(top1_accuracies_matrix, 'Top-1 Accuracy Heatmap', f"{save_path}_top1.png", model_extensions)
    plot_heatmap(top5_accuracies_matrix, 'Top-5 Accuracy Heatmap', f"{save_path}_top5.png", model_extensions)

# Örnek model uzantıları listesi

model_extensions =["eminAydin/turkish-gpt2-mini-M1-cleaned-sports720k-10ep", 
                   "eminAydin/turkish-gpt2-mini-M2-cleaned-politics720k-10ep",
                   "eminAydin/Turkish-GPT2-mini-ModelA-v2",
                   "eminAydin/Turkish-GPT2-mini-ModelB-v2",
                   "eminAydin/Turkish-GPT2-mini-ModelC-v2",
                   "eminAydin/Turkish-GPT2-mini-ModelD"]
# Veri setini yükle ve her satır için işlem yap
def load_dataset(file_path, start_index, filter):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip(filter) for line in file.readlines()]

    return [line.strip() for line in lines[start_index:start_index+1000]]

# Veri setinin yolunu belirt
dataset_path = 'sports_843648_lines.txt'
dataset1 = load_dataset(dataset_path, 800000, "__label__sports")
dataset_path = 'politics_737166_lines.txt'
dataset2 = load_dataset(dataset_path, 730000, "__label__politics")

# Örnek veri seti listesi
datasets = [dataset1, dataset2]

# Modelleri ve veri setlerini değerlendir
evaluate_models(model_extensions, datasets, tokenizer, save_path="Model1-2_evaluation")
