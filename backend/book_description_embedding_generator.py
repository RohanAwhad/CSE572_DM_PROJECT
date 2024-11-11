from transformers import AutoTokenizer, AutoModel
import torch
import json

# Load the model and tokenizer
model_name = "avsolatorio/NoInstruct-small-Embedding-v0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.to(device)
    # print("Output is on device:", embeddings.device)
    # Mean pool over the token embeddings to get a single vector for the input
    embedding_vector = embeddings.mean(dim=1)

    embedding_list = embedding_vector.squeeze().cpu().numpy().tolist()
    return embedding_list


def generate_description_embedding(file_path: str) -> dict:
    result = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                row = json.loads(line)
                description = row.get("description")
                book_id = row.get("book_id")

                if description == "":
                    # print(row)
                    continue
                description_embedding = get_embedding(description)
                result[book_id] = description_embedding
            except json.JSONDecodeError:
                print("Invalid JSON format in line:", line)

    return result


if __name__ == '__main__':
    res = generate_description_embedding("books_1000.json")
    print(res)
