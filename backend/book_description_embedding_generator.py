from transformers import AutoTokenizer, AutoModel
import torch
import json
import functools  # Import functools for lru_cache
from typing import Dict, Optional

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embedding(text):
    # Ensure tokenizer and model are loaded on the correct device
    tokenizer, model = _load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.to(device)
    embedding_vector = embeddings[:, 0, :]

    embedding_list = embedding_vector.squeeze().cpu().numpy().tolist()
    return embedding_list


@functools.lru_cache()
def _load_model():
    # Load the model and tokenizer
    model_name = "avsolatorio/NoInstruct-small-Embedding-v0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    model.eval()
    return tokenizer, model


EMBEDDING_BATCH_SIZE = 8


@torch.no_grad()
def embed(chunks: list[str], mode: str) -> Optional[list[list[float]]]:
    if mode not in ["sentence", "query"]:
        print('Mode has to either be "sentence" or "query", but got', mode)
        return None

    tokenizer, model = _load_model()
    ret = []
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i: i + EMBEDDING_BATCH_SIZE]

        # Move input to the correct device
        inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        output = model(**inp)

        if mode == "query":
            vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
            vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
        else:
            vectors = output.last_hidden_state[:, 0, :]

        ret.extend(vectors.cpu().numpy().tolist())
    return ret


def generate_description_embedding(file_path: str) -> Dict[str, list[float]]:
    result = {}
    descriptions = []
    book_ids = []

    # Collect descriptions and book_ids for batching
    with open(file_path, 'r') as file:
        for line in file:
            try:
                row = json.loads(line)
                description = row.get("description")
                book_id = row.get("book_id")

                if description == "":
                    continue

                descriptions.append(description)
                book_ids.append(book_id)

                # Process in batches when we reach EMBEDDING_BATCH_SIZE
                if len(descriptions) == EMBEDDING_BATCH_SIZE:
                    embeddings = embed(descriptions, mode="sentence")
                    result.update(dict(zip(book_ids, embeddings)))
                    descriptions.clear()
                    book_ids.clear()

            except json.JSONDecodeError:
                print("Invalid JSON format in line:", line)

    # Process any remaining descriptions
    if descriptions:
        embeddings = embed(descriptions, mode="sentence")
        result.update(dict(zip(book_ids, embeddings)))

    return result


if __name__ == '__main__':
    # Test the embedding generation function
    res = generate_description_embedding("books_1000.json")
    print(res)
