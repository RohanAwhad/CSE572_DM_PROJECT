import os
import pickle
import numpy as np

def merge_files(embedding_dir, output_ids_file, output_embeddings_file):
  """
  Merges all batch IDs and embeddings files in the given directory into single consolidated files.
  
  Args:
  - embedding_dir (str): Path to the directory containing the batch files.
  - output_ids_file (str): Path to save the merged book IDs.
  - output_embeddings_file (str): Path to save the merged embeddings.
  """
  all_book_ids = []
  all_embeddings = []

  for file_name in sorted(os.listdir(embedding_dir)):
    file_path = os.path.join(embedding_dir, file_name)
    if file_name.endswith("_ids.pkl"):
      # Load and append IDs
      with open(file_path, "rb") as f:
        book_ids = pickle.load(f)
        all_book_ids.extend(book_ids)
    elif file_name.endswith("_embeddings.npy"):
      # Load and append embeddings
      embeddings = np.load(file_path)
      all_embeddings.append(embeddings)

  # Save consolidated IDs
  with open(output_ids_file, "wb") as f:
    pickle.dump(all_book_ids, f)
  print(f"Merged book IDs saved to: {output_ids_file}")

  # Save consolidated embeddings
  all_embeddings = np.vstack(all_embeddings)
  np.save(output_embeddings_file, all_embeddings)
  print(f"Merged embeddings saved to: {output_embeddings_file}")

if __name__ == "__main__":
  # Paths and output file names
  EMBEDDING_DIR = "/scratch/rawhad/CSE572/project/data/embeddings"  # Directory containing batch files
  OUTPUT_IDS_FILE = "/scratch/rawhad/CSE572/project/data/all_book_ids.pkl"
  OUTPUT_EMBEDDINGS_FILE = "/scratch/rawhad/CSE572/project/data/all_embeddings.npy"

  # Merge files
  merge_files(EMBEDDING_DIR, OUTPUT_IDS_FILE, OUTPUT_EMBEDDINGS_FILE)
