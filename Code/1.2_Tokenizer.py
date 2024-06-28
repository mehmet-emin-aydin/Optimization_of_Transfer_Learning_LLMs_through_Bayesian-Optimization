# Açıklama: Bu dosya, veri kümesi üzerinde eğitilmiş bir tokenizer oluşturur ve kaydeder.
import datasets
from transformers import AutoTokenizer

def batch_iterator(dataset, batch_size=1_000):
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]

if __name__ == "__main__":

    ds_id = "musabg/wikipedia-tr"
    clone_from_name = "gpt2"
    vocab_size = 32_768

    clone_from_tokenizer = AutoTokenizer.from_pretrained(clone_from_name)
    ds_train = datasets.load_dataset(ds_id, split="train")
    # remove non text columns
    ds_train = ds_train.remove_columns([
        col for col in ds_train.column_names if col != "text"
    ])

    tokenizer = clone_from_tokenizer.train_new_from_iterator(
        batch_iterator(ds_train),
        vocab_size=vocab_size,
    )
    tokenizer.save_pretrained("my_new-retrained-gpt2-v32k-tokenizer")