import re
from bs4 import BeautifulSoup
import nltk

nltk.download('punkt')

# Veri okuma ve temel temizleme
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()

def basic_cleaning(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text

# HTML/XML etiketlerini kaldırma
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Gereksiz içerikleri filtreleme ve paragrafların sonuna nokta ekleme
def remove_unwanted_content(text):
    # Paragrafların sonuna nokta ekleme
    text = re.sub(r'([^\.\?!])\s*(==+)', r'\1. \2', text)
    
    # Başlıkları kaldır
    text = re.sub(r'==.*?==+', '', text)
    # Tabloları kaldır
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Liste elemanlarını kaldır
    text = re.sub(r'\*.*?\n', '', text)
    return text

# Özel durumları koruma ve cümlelere bölme
def split_into_sentences(text):
    # Özel isimleri ve tarih formatlarını korumak için geçici değişiklikler yapma
    text = re.sub(r'(\d+)\. (Yüzyıl|Yıl|cm|gr|m|kg|l|ml|km)', r'\1<UNIT> \2', text)
    text = re.sub(r'([A-ZÇĞİÖŞÜ]\.)', r'\1<PERIOD>', text)



    # Nokta ve kelimeler arasındaki hatalı bölünmeleri önlemek için düzenleme
    text = re.sub(r'(\d+)\. ([a-zçğıöşü])', r'\1.<NOPBREAK> \2', text)

    # Cümlelere bölme
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)

    # Geçici değişiklikleri geri alma
    sentences = [re.sub(r'<PERIOD>', '.', sentence) for sentence in sentences]
    sentences = [re.sub(r'<UNIT>', '.', sentence) for sentence in sentences]

    # Ardışık noktalama işaretlerini kaldırma
    sentences = [re.sub(r'([.?!]){2,}', r'\1', sentence) for sentence in sentences]

    # <NOPBREAK> etiketlerini kaldırma
    sentences = [sentence.replace('<NOPBREAK>', '') for sentence in sentences]

    # Cümle sonunu kontrol etme ve yanlış cümleleri kaldırma
    valid_endings = {'.', '...', '?', '!'}
    sentences = [sentence for sentence in sentences if sentence[-1] in valid_endings]

    # İki noktayı tek noktaya çevirme
    sentences = [sentence.replace('..', '.') for sentence in sentences]

    # Tek kelimelik cümleleri kaldırma
    sentences = [sentence for sentence in sentences if len(sentence.split()) > 1]

    return sentences

# Özel durumu düzeltme fonksiyonu
def fix_special_cases(sentences):
    fixed_sentences = []
    for sentence in sentences:
        # Noktalama işaretlerinden sonra başka bir noktalama işareti kontrolü
        sentence = re.sub(r'([,;:\.\?!])\s*([,;:\.\?!])', r'\2', sentence)
        fixed_sentences.append(sentence)
    return fixed_sentences

# Ana fonksiyon
def process_wikipedia_data(file_path):
    sentences = []
    for line in read_data(file_path):
        cleaned_data = basic_cleaning(line)
        cleaned_data = remove_html_tags(cleaned_data)
        cleaned_data = remove_unwanted_content(cleaned_data)
        new_sentences = split_into_sentences(cleaned_data)
        sentences.extend(new_sentences)
    sentences = fix_special_cases(sentences)
    return sentences

# Dosya yolu
file_path = 'trwiki-67.train.txt'
sentences = process_wikipedia_data(file_path)

# Sonuçları dosyaya yazma
output_file = 'trwiki-67-train-cleaned.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for sentence in sentences:
        file.write(sentence + '\n')

print(f"Veri başariyla '{output_file}' dosyasina kaydedildi.")