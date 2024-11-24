# ===
# Books Project
# ===

from pyspark import SparkContext
import json
import os
import pandas as pd
import pickle
import random

BOOK_JSON: str = '/scratch/rawhad/CSE572/project/data/goodreads_books.json'
TRAIN_BOOK_IDS_FP: str = '/scratch/rawhad/CSE572/project/data/train_book_ids.pkl'
TEST_BOOK_IDS_FP: str = '/scratch/rawhad/CSE572/project/data/test_book_ids.pkl'

if os.path.exists(TRAIN_BOOK_IDS_FP) and os.path.exists(TEST_BOOK_IDS_FP):
    # load
    with open(TRAIN_BOOK_IDS_FP, 'rb') as f: train_book_set = pickle.load(f)
    with open(TEST_BOOK_IDS_FP, 'rb') as f: test_book_set = pickle.load(f)
else:
    # Initialize Spark context
    sc: SparkContext = SparkContext(appName="BookSplit")

    print('getting all_book ids')
    book_data = sc.textFile(BOOK_JSON)
    all_book_ids = book_data.map(lambda x: x.strip()) \
                            .filter(lambda x: len(x) > 0) \
                            .map(json.loads) \
                            .filter(lambda x: 'book_id' in x) \
                            .map(lambda x: x['book_id']) \
                            .collect()

    print('splitting into train test sets')
    random.shuffle(all_book_ids)
    num_test_books: int = int(0.2 * len(all_book_ids))
    test_book_set: list = all_book_ids[:num_test_books]
    train_book_set: list = all_book_ids[num_test_books:]

    # save
    with open(TRAIN_BOOK_IDS_FP, 'wb') as f:
      pickle.dump(train_book_set, f)

    with open(TEST_BOOK_IDS_FP, 'wb') as f:
      pickle.dump(test_book_set, f)

    # Stop Spark context
    sc.stop()


# ===
# Interactions DF
# ===
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
# select 20% users
INTERACTION_PARQUET = '/scratch/rawhad/CSE572/project/data/goodreads_interactions.parquet'
TRAIN_INTERACTION_PARQUET = '/scratch/rawhad/CSE572/project/data/train_interactions.parquet'
TRAIN_USER_IDS_FP = '/scratch/rawhad/CSE572/project/data/train_user_ids.pkl'
TEST_USER_IDS_FP = '/scratch/rawhad/CSE572/project/data/test_used_ids.pkl'
USER2BOOKS_FP = '/scratch/rawhad/CSE572/project/data/user_to_books.parquet'
BOOK2USERS_FP = '/scratch/rawhad/CSE572/project/data/book_to_users.parquet'


if os.path.exists(TRAIN_USER_IDS_FP) and os.path.exists(TEST_USER_IDS_FP) and os.path.exists(USER2BOOKS_FP) and os.path.exists(BOOK2USERS_FP):
    with open(TRAIN_USER_IDS_FP, 'rb') as f: train_user_ids = pickle.load(f)
    with open(TEST_USER_IDS_FP, 'rb') as f: test_user_ids = pickle.load(f)
    # Load the Parquet files into Pandas DataFrames
    user_to_books_df = pd.read_parquet(USER2BOOKS_FP)
    book_to_users_df = pd.read_parquet(BOOK2USERS_FP)

else:
    spark = SparkSession.builder \
            .appName('GoodreadsDataProcessing') \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.dynamicAllocation.minExecutors", "1") \
            .config("spark.dynamicAllocation.maxExecutors", "10") \
            .config("spark.sql.shuffle.partitions", "1200") \
            .config("spark.executor.memory", "32g") \
            .config("spark.driver.memory", "32g") \
            .config("spark.memory.offHeap.enabled",True) \
            .config("spark.memory.offHeap.size","32g") \
            .config("spark.shuffle.compress", True) \
            .config("spark.shuffle.spill.compress", True) \
            .config("spark.executor.memoryOverhead", "10g")  \
            .config("spark.driver.maxResultSize", "10g") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
            .getOrCreate()

    interactions_df = spark.read.parquet(INTERACTION_PARQUET).select('user_id', 'book_id')
    print('# getting unique users')
    unique_user_ids = interactions_df.select('user_id').distinct().rdd.flatMap(lambda x: x).collect()
    print('# Shuffle and split user IDs')
    random.shuffle(unique_user_ids)
    num_test_users = int(0.2 * len(unique_user_ids))
    test_user_ids = unique_user_ids[:num_test_users]
    train_user_ids = unique_user_ids[num_test_users:]
    #print('# Filter out rows with book_id in test_book_set')
    #print('#   Broadcast sets')
    #train_book_set_bc = spark.sparkContext.broadcast(set(train_book_set))
    #train_user_ids_bc = spark.sparkContext.broadcast(set(train_user_ids))
    #train_interactions_df = interactions_df.filter((col('book_id').isin(train_book_set_bc.value)) & (col('user_id').isin(train_user_ids_bc.value)))
    #print('# Save')
    #train_interactions_df.write.parquet(TRAIN_INTERACTION_PARQUET)
    with open(TRAIN_USER_IDS_FP, 'wb') as f: pickle.dump(train_user_ids, f)
    with open(TEST_USER_IDS_FP, 'wb') as f: pickle.dump(test_user_ids, f)



    from pyspark.sql.functions import collect_list

    print('#  Creating user 2 books dict')
    user_to_books_df = interactions_df.groupBy('user_id').agg(collect_list('book_id').alias('books'))
    print('#  Creating book 2 users dict')
    book_to_users_df = interactions_df.groupBy('book_id').agg(collect_list('user_id').alias('users'))
    print('#  collapsing partitioned parquet files into 1')
    user_to_books_df.coalesce(1).write.mode("overwrite").parquet(USER2BOOKS_FP)
    book_to_users_df.coalesce(1).write.mode("overwrite").parquet(BOOK2USERS_FP)
    spark.stop()



# ===
# Book Description Embeddings Matrix 
# ===
import numpy as np

BOOK_IDS_PKL = '/scratch/rawhad/CSE572/project/data/all_book_ids.pkl'  # book_ids with embeddings of description
EMBEDDINGS_NPY = '/scratch/rawhad/CSE572/project/data/all_embeddings.npy'  # embeddings[idx] are embeddings of book[book_ids[idx]]['desc']

# create an embedding matrix for books in training set
# laod book_ids_pkl => list[book_ids] and EMBEDDINGS_NPY => np.array(len(book_ids), 384)
# then get embedding_indices for train set and then claw out the train books embedding matrix
# ignore book ids not present

# Load book IDs and embeddings
with open(BOOK_IDS_PKL, 'rb') as f: book_ids_with_embeddings = pickle.load(f)
embeddings = np.load(EMBEDDINGS_NPY)
# Create a mapping of book IDs to their embedding index
book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids_with_embeddings)}
# Get embedding indices for train set
train_embedding_indices = [book_id_to_index[book_id] for book_id in train_book_set if book_id in book_id_to_index]
# Extract embeddings for the train set
train_book_embeddings = embeddings[train_embedding_indices]


# ===
# Popularity based on Average Rating
# ===
sc: SparkContext = SparkContext(appName="PopularitySort")
BOOK_RATINGS = sc.textFile(BOOK_JSON) \
               .map(json.loads) \
               .filter(lambda x: 'book_id' in x and 'average_rating' in x and x['average_rating'] != '') \
               .map(lambda x: (x['book_id'], float(x['average_rating']))) \
               .sortBy(lambda x: x[1], ascending=False) \
               .map(lambda x: x[0])


# ===
# Recommendations
# ===
# create a new function `reccommendations` that will take in user_id, list[book_ids] # these are previously read book_ids
# and it will return top-10 recommendations

# it will check if the len(book_ids) == 0, if yes, then return top-10 books of all time.
# if user has previously read a few books, then find the top-10 books of all time, which user hasn't read.
# Collaborative Filtering: then find books read by other users of the same books, and then rank them according to number of reads, limit to 10
# Content Based Filtering: then find books which are similar to the books given based on cosine similarity and then rank similar books based on how close they are to the previous books, limit to 10
# rerank books with weight of sigmoid(len(book_ids)/20) for popularity and (1-popularity_weight) for both Collaborative and Content based filtering.
# The rerank function is Reciprocal Rerank Fusion
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter


def reciprocal_rank_fusion(rankings, k=60):
  fused_scores = Counter()
  for ranking in rankings:
    for rank, item in enumerate(ranking, 1):
      fused_scores[item] += 1 / (rank + k)
  return [item for item, _ in fused_scores.most_common()]

def recommendations(user_id, book_ids, n):
  if len(book_ids) == 0:
    return get_top_books(book_ids, m)
  
  top_books = [book for book in get_top_books(book_ids, n) if book not in book_ids]
  cf_books = collaborative_filtering(user_id, book_ids, n)
  cb_books = content_based_filtering(book_ids, n)
  
  popularity_weight = 1 / (1 + np.exp(-len(book_ids)/20))
  
  rankings = [
    top_books,
    cf_books,
    cb_books
  ]
  
  weights = [
    popularity_weight,
    (1 - popularity_weight) / 2,
    (1 - popularity_weight) / 2
  ]
  
  fused_ranking = reciprocal_rank_fusion(rankings)
  
  return fused_ranking[:n]


def get_top_books(read_book_ids, n=10) -> list[str]:
  return BOOK_RATINGS.filter(lambda x: x not in read_book_ids).take(n)

from sklearn.metrics.pairwise import cosine_distances

def content_based_filtering(book_ids, n) -> list[str]:
  # Check if any of the book_ids have embeddings
  valid_book_ids = [book_id for book_id in book_ids if book_id in book_id_to_index]
  
  if not valid_book_ids:
    return []

  # Get embeddings for input book IDs
  book_embeddings = embeddings[[book_id_to_index[book_id] for book_id in valid_book_ids]]

  # Calculate cosine distances between input books and all books with embeddings
  distances = cosine_distances(book_embeddings, embeddings)

  # Get similar books, excluding already read ones
  similar_books = []
  for dist in distances:
    similar_indices = np.argsort(dist)[:10 + len(book_ids)]
    for idx in similar_indices:
      similar_id = book_ids_with_embeddings[idx]
      if similar_id not in book_ids:
        similar_books.append((similar_id, dist[idx]))

  # Sort and get unique similar books
  similar_books = sorted(similar_books, key=lambda x: x[1])
  unique_similar_books = []
  seen = set()
  
  for book_id, _ in similar_books:
    if book_id not in seen:
      unique_similar_books.append(book_id)
      seen.add(book_id)
      if len(unique_similar_books) == n:
        break

  return unique_similar_books



def collaborative_filtering(user_id, book_ids, n) -> list[str]:
  # Find similar users who have read the same books
  similar_users = set()
  for book_id in book_ids:
    if book_id in book_to_users_df.index:
      similar_users.update(book_to_users_df.loc[book_id])

  # Collect books read by similar users excluding those already read
  candidate_books = Counter()
  for similar_user in similar_users:
    if similar_user != user_id and similar_user in user_to_books_df.index:
      for book_id in user_to_books_df.loc[similar_user]:
        if book_id not in book_ids:
          candidate_books[book_id] += 1

  # Rank books by the number of similar users who read them
  ranked_books = [book_id for book_id, _ in candidate_books.most_common(n)]
  return ranked_books


if __name__ == '__main__':
  test_book_ids = list(test_book_set)
  print(recommendations(test_user_ids[0], []), 10)
  print(recommendations(test_user_ids[2], test_book_ids[:3]), 10)

  import numpy as np
  from sklearn.metrics import ndcg_score

  def calculate_ndcg(true_items, predicted_items, k):
    relevance = [1 if item in true_items else 0 for item in predicted_items[:k]]
    ideal_relevance = sorted(relevance, reverse=True)
    return ndcg_score([ideal_relevance], [relevance])

  def calculate_recall(true_items, predicted_items, k):
    relevant_items = set(true_items) & set(predicted_items[:k])
    return len(relevant_items) / len(true_items) if true_items else 0

  def calculate_precision(true_items, predicted_items, k):
    relevant_items = set(true_items) & set(predicted_items[:k])
    return len(relevant_items) / k if k > 0 else 0

  def evaluate_recommendations(true_items, predicted_items):
    results = {}
    
    # NDCG
    for k in [10, 20]:
      results[f'NDCG@{k}'] = calculate_ndcg(true_items, predicted_items, k)
    
    # Recall
    for k in [10, 20, 50, 100]:
      results[f'Recall@{k}'] = calculate_recall(true_items, predicted_items, k)
    
    # Precision
    for k in [1, 2, 5, 10]:
      results[f'Precision@{k}'] = calculate_precision(true_items, predicted_items, k)
    
    return results

  def evaluate_model(test_users, num_users_to_evaluate=1000):
    all_results = {metric: [] for metric in ['NDCG@10', 'NDCG@20', 'Recall@10', 'Recall@20', 'Recall@50', 'Recall@100', 'Precision@1', 'Precision@2', 'Precision@5', 'Precision@10']}
    
    for user_id in test_users[:num_users_to_evaluate]:
      if user_id in user_to_books_df.index:
        true_items = user_to_books_df.loc[user_id]
        
        # Split true items into history and test set
        history = true_items[:len(true_items)//2]
        test_set = true_items[len(true_items)//2:]
        
        # Get recommendations
        predicted_items = recommendations(user_id, history, n=100)
        
        # Evaluate
        results = evaluate_recommendations(test_set, predicted_items)
        
        for metric, value in results.items():
          all_results[metric].append(value)
    
    # Calculate average results
    avg_results = {metric: np.mean(values) for metric, values in all_results.items()}
    return avg_results

  # Evaluate the model
  evaluation_results = evaluate_model(test_user_ids)

  # Print results
  for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")


  with open('evaluation_results.pkl', 'wb') as f: pickle.dump(evaluation_results, f)

sc.stop()

