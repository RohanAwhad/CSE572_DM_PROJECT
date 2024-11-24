from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import json
import numpy as np
import pickle
import functools
from typing import Dict, List
import os

torch.set_float32_matmul_precision('high')
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cached model loading
@functools.lru_cache()
def _load_model():
    model_name = "avsolatorio/NoInstruct-small-Embedding-v0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def embed_batch(texts: List[str], batch_size: int, mode: str = "sentence") -> List[List[float]]:
    """
    Embed a batch of text descriptions and return their embeddings.
    """
    tokenizer, model = _load_model()
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), total=len(texts)//batch_size, desc='Embedding', leave=True):
        batch = texts[i:i + batch_size]
        
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", padding="longest", truncation=True).to(device)
            outputs = model(**inputs)

            # CLS token embedding
            if mode == "sentence":
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif mode == "query":
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                batch_embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                batch_embeddings = batch_embeddings.cpu().numpy()
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            embeddings.extend(batch_embeddings)
        torch.cuda.empty_cache()
    
    return embeddings


def process_file_in_batches(file_path: str, save_dir: str, file_batch_size: int, embed_batch_size: int):
    """
    Process the JSON file in batches, generate embeddings, and save results to disk.
    
    Args:
    - file_path (str): Path to the input JSON file.
    - save_dir (str): Directory to save processed embeddings and IDs.
    - file_batch_size (int): Number of lines to read from the file in each batch.
    - embed_batch_size (int): Number of descriptions to embed in each GPU batch.
    """

    os.makedirs(save_dir, exist_ok=True)


    tokenizer, _ = _load_model()
    total_embeddings = 0
    batch_count = 0
    
    with open(file_path, 'r') as f:
        batch_descriptions = []
        batch_book_ids = []
        
        for line in tqdm(f, desc="Reading File in Batches"):
            try:
                row = json.loads(line)
                description = row.get("description", "").strip()
                book_id = row.get("book_id")

                if description:
                    batch_descriptions.append(description)
                    batch_book_ids.append(book_id)
                
                if len(batch_descriptions) == file_batch_size:  # Process file batch
                    # Generate embeddings for the batch
                    embeddings = embed_batch(batch_descriptions, embed_batch_size, mode="sentence")

                    # Save results
                    save_results(batch_book_ids, embeddings, save_dir, batch_count)
                    batch_count += 1
                    total_embeddings += len(batch_descriptions)

                    # Clear batch buffers
                    batch_descriptions.clear()
                    batch_book_ids.clear()
            
            except json.JSONDecodeError:
                print("Invalid JSON line skipped.")

        # Process the remaining batch
        if batch_descriptions:
            embeddings = embed_batch(batch_descriptions, embed_batch_size, mode="sentence")
            save_results(batch_book_ids, embeddings, save_dir, batch_count)
            total_embeddings += len(batch_descriptions)

    print(f"Total embeddings generated: {total_embeddings}")


def save_results(book_ids: List[str], embeddings: List[List[float]], save_dir: str, batch_index: int):
    """
    Save book IDs and embeddings to files in chunks.
    
    Args:
    - book_ids (List[str]): List of book IDs for the current batch.
    - embeddings (List[List[float]]): List of embeddings for the current batch.
    - save_dir (str): Directory to save results.
    - batch_index (int): Index of the current batch for naming.
    """
    id_file = f"{save_dir}/batch_{batch_index}_ids.pkl"
    embedding_file = f"{save_dir}/batch_{batch_index}_embeddings.npy"
    
    # Save book IDs
    with open(id_file, "wb") as f:
        pickle.dump(book_ids, f)
    
    # Save embeddings
    np.save(embedding_file, np.array(embeddings))
    print(f"Saved batch {batch_index}: {len(book_ids)} entries")


if __name__ == "__main__":
    # Paths and parameters
    FILE_PATH = "/scratch/rawhad/CSE572/project/data/goodreads_books.json"
    SAVE_DIR = "/scratch/rawhad/CSE572/project/data/embeddings"
    FILE_BATCH_SIZE = 1048576  # Number of lines read from the file in each batch
    EMBED_BATCH_SIZE = 2048  # Number of descriptions embedded per GPU batch

    process_file_in_batches(FILE_PATH, SAVE_DIR, FILE_BATCH_SIZE, EMBED_BATCH_SIZE)
