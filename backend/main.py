# ===
# Books Project
# ===

import time
from pyspark import SparkContext
from tqdm import tqdm
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

    # set user_id in user_to_books_df as str and books as list[str]
    #user_to_books_df['user_id'] = user_to_books_df['user_id'].astype(str)
    #user_to_books_df['books'] = user_to_books_df['books'].apply(lambda x: list(map(str, x)))
    user_to_books_df = user_to_books_df.set_index('user_id', drop=True)

    # set book_id in book_to_users_df as str and users as list[str]
    #book_to_users_df['book_id'] = book_to_users_df['book_id'].astype(str)
    #book_to_users_df['users'] = book_to_users_df['users'].apply(lambda x: list(map(str, x)))
    book_to_users_df = book_to_users_df.set_index('book_id', drop=True)


    print(user_to_books_df.head())
    print(book_to_users_df.head())


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
    with open(TRAIN_USER_IDS_FP, 'wb') as f: pickle.dump(train_user_ids, f)
    with open(TEST_USER_IDS_FP, 'wb') as f: pickle.dump(test_user_ids, f)



    from pyspark.sql.functions import collect_list

    print('#  Creating user 2 books dict')
    user_to_books_df = interactions_df.groupBy('user_id').agg(collect_list('book_id').alias('books'))
    user_to_books_df = user_to_books_df.withColumn('user_id', col('user_id').cast('string'))
    #user_to_books_df = user_to_books_df.withColumn('books', col('books').cast('array<string>'))

    print('#  Creating book 2 users dict')
    book_to_users_df = interactions_df.groupBy('book_id').agg(collect_list('user_id').alias('users'))
    book_to_users_df = book_to_users_df.withColumn('book_id', col('book_id').cast('string'))
    #book_to_users_df = book_to_users_df.withColumn('users', col('users').cast('array<string>'))

    print('#  collapsing partitioned parquet files into 1')
    user_to_books_df.coalesce(1).write.mode("overwrite").parquet(USER2BOOKS_FP)
    book_to_users_df.coalesce(1).write.mode("overwrite").parquet(BOOK2USERS_FP)

    # Load the Parquet files into Pandas DataFrames
    user_to_books_df = pd.read_parquet(USER2BOOKS_FP)
    book_to_users_df = pd.read_parquet(BOOK2USERS_FP)

    # Set index for faster lookups
    user_to_books_df = user_to_books_df.set_index('user_id', drop=True)
    book_to_users_df = book_to_users_df.set_index('book_id', drop=True)

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
# Write BOOK_RATINGS to disk
BOOK_RATINGS_FP = '/scratch/rawhad/CSE572/project/data/book_ratings.pkl'

if not os.path.exists(BOOK_RATINGS_FP):
  sc: SparkContext = SparkContext(appName="PopularitySort")

  BOOK_RATINGS = sc.textFile(BOOK_JSON) \
                 .map(json.loads) \
                 .filter(lambda x: 'book_id' in x and 'average_rating' in x and x['average_rating'] != '') \
                 .map(lambda x: (x['book_id'], float(x['average_rating']))) \
                 .sortBy(lambda x: x[1], ascending=False) \
                 .map(lambda x: x[0])
  BOOK_RATINGS_LIST = BOOK_RATINGS.collect()
  with open(BOOK_RATINGS_FP, 'wb') as f:
    pickle.dump(BOOK_RATINGS_LIST, f)
  sc.stop()
else:
  # Load BOOK_RATINGS from disk
  with open(BOOK_RATINGS_FP, 'rb') as f:
    BOOK_RATINGS_LIST = pickle.load(f)
  print(BOOK_RATINGS_LIST[:10])


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

def recommendations(user_id: str, book_ids: list[str], n):
  if len(book_ids) > 0: print('type of book_ids[0]:', type(book_ids[0]))

  if len(book_ids) == 0:
    return get_top_books(book_ids, n)
  
  start = time.monotonic()
  top_books = [book for book in get_top_books(book_ids, n) if book not in book_ids]
  top_end = time.monotonic()
  cf_books = collaborative_filtering(user_id, book_ids, n)
  cf_end = time.monotonic()
  cb_books = content_based_filtering(book_ids, n)
  cb_end = time.monotonic()
  print(f'Time taken to get top books:', (top_end - start) * 1e3, 'ms')
  print(f'Time taken to get cf books:', (cf_end - top_end) * 1e3, 'ms')
  print(f'Time taken to get cb books:', (cb_end - cf_end) * 1e3, 'ms')
  
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


def get_top_books(read_book_ids: list[str], n=10) -> list[str]:
  ret = []
  for x in BOOK_RATINGS_LIST:
    if x not in read_book_ids: ret.append(x)
    if len(ret) == n: return ret
  return ret


from sklearn.metrics.pairwise import cosine_distances

def content_based_filtering(book_ids: list[str], n) -> list[str]:
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



def collaborative_filtering(user_id: str, book_ids: list[str], n) -> list[str]:
  # Find similar users who have read the same books
  similar_users = set()
  for book_id in book_ids:
    if book_id in book_to_users_df.index:
      similar_users.update(list(map(str, book_to_users_df.loc[book_id, 'users'])))

  # Collect books read by similar users excluding those already read
  candidate_books = Counter()
  for similar_user in similar_users:
    if similar_user != user_id and similar_user in user_to_books_df.index:
      for book_id in map(str, user_to_books_df.loc[similar_user, 'books']):
        if book_id not in book_ids:
          candidate_books[book_id] += 1

  # Rank books by the number of similar users who read them
  ranked_books = [book_id for book_id, _ in candidate_books.most_common(n)]
  return ranked_books



if __name__ == '__main__':
  test_book_ids = list(test_book_set)
  print('Recommendation 1:', recommendations(str(test_user_ids[0]), [], 10))
  print('Recommendation 2:', recommendations(str(test_user_ids[2]), test_book_ids[:3], 10))

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

  def evaluate_model(test_users, num_users_to_evaluate=10):
    all_results = {metric: [] for metric in ['NDCG@10', 'NDCG@20', 'Recall@10', 'Recall@20', 'Recall@50', 'Recall@100', 'Precision@1', 'Precision@2', 'Precision@5', 'Precision@10']}
    
    for user_id in tqdm(map(str, test_users[:num_users_to_evaluate]), total=min(len(test_users), num_users_to_evaluate), desc='Evaluating'):
      if user_id in user_to_books_df.index:
        true_items = list(map(str, user_to_books_df.loc[user_id, 'books']))
        
        # Split true items into history and test set
        history = true_items[:len(true_items)//2]
        test_set = true_items[len(true_items)//2:]
        
        # Get recommendations
        predicted_items = recommendations(user_id, history, n=100)
        
        # Evaluate
        results = evaluate_recommendations(test_set, predicted_items)
        
        for metric, value in results.items():
          all_results[metric].append(value)
      else:
        print(user_id, 'not found in user_to_books_df')
    
    # Calculate average results
    avg_results = {metric: np.mean(values) for metric, values in all_results.items()}
    return avg_results

  # Evaluate the model
  evaluation_results = evaluate_model(test_user_ids)

  # Print results
  for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")


  with open('evaluation_results.pkl', 'wb') as f: pickle.dump(evaluation_results, f)


  # also evaluate for cold starts
  def evaluate_cold_start(test_users, num_users_to_evaluate=10):
    all_results = {metric: [] for metric in ['NDCG@10', 'NDCG@20', 'Recall@10', 'Recall@20', 'Recall@50', 'Recall@100', 'Precision@1', 'Precision@2', 'Precision@5', 'Precision@10']}
    
    for user_id in tqdm(map(str, test_users[:num_users_to_evaluate]), total=min(len(test_users), num_users_to_evaluate), desc='Evaluating'):
      if user_id in user_to_books_df.index:
        true_items = list(map(str, user_to_books_df.loc[user_id, 'books']))
        
        # Get recommendations for cold start (empty history)
        predicted_items = recommendations(user_id, [], n=100)
        
        # Evaluate
        results = evaluate_recommendations(true_items, predicted_items)
        
        for metric, value in results.items():
          all_results[metric].append(value)
    
    # Calculate average results
    avg_results = {metric: np.mean(values) for metric, values in all_results.items()}
    return avg_results

  # Evaluate the model for cold start
  cold_start_results = evaluate_cold_start(test_user_ids)

  # Print cold start results
  print("\nCold Start Evaluation Results:")
  for metric, value in cold_start_results.items():
    print(f"{metric}: {value:.4f}")

  # Save cold start results
  with open('cold_start_results.pkl', 'wb') as f:
    pickle.dump(cold_start_results, f)

  import multiprocessing as mp
  from functools import partial

  def evaluate_single_user(user_id: str, user_to_books_df, is_cold_start=False):
      """Evaluate recommendations for a single user"""
      if user_id in user_to_books_df.index:
          true_items = list(map(str, user_to_books_df.loc[user_id, 'books']))
          
          if is_cold_start:
              # Cold start evaluation
              predicted_items = recommendations(user_id, [], n=100)
              results = evaluate_recommendations(true_items, predicted_items)
          else:
              # Regular evaluation
              history = true_items[:len(true_items)//2]
              test_set = true_items[len(true_items)//2:]
              predicted_items = recommendations(user_id, history, n=100)
              results = evaluate_recommendations(test_set, predicted_items)
              
          return results
      return None

  def parallel_evaluate_model(test_users, user_to_books_df, num_users_to_evaluate=10, is_cold_start=False):
      """Parallel evaluation of the recommendation model"""
      # Initialize metrics dictionary
      all_metrics = ['NDCG@10', 'NDCG@20', 'Recall@10', 'Recall@20', 'Recall@50', 
                    'Recall@100', 'Precision@1', 'Precision@2', 'Precision@5', 'Precision@10']
      all_results = {metric: [] for metric in all_metrics}
      
      # Create a partial function with fixed arguments
      evaluate_func = partial(evaluate_single_user, 
                            user_to_books_df=user_to_books_df, 
                            is_cold_start=is_cold_start)
      
      # Create process pool with 14 cores
      with mp.Pool(processes=14) as pool:
          # Evaluate users in parallel
          user_subset = list(map(str, test_users[:num_users_to_evaluate]))
          results = list(tqdm(
              pool.imap(evaluate_func, user_subset),
              total=len(user_subset),
              desc='Evaluating'
          ))
          
          # Aggregate results
          for user_result in results:
              if user_result is not None:
                  for metric in all_metrics:
                      all_results[metric].append(user_result[metric])
      
      # Calculate average results
      avg_results = {metric: np.mean(values) for metric, values in all_results.items()}
      return avg_results

  #evaluation_results = parallel_evaluate_model(
  #    test_users=test_user_ids,
  #    user_to_books_df=user_to_books_df,
  #    num_users_to_evaluate=10,
  #    is_cold_start=False
  #)
  #print("\nEvaluation Results:")
  #for metric, value in cold_start_results.items():
  #  print(f"{metric}: {value:.4f}")

  # Cold start evaluation with parallel processing
  #cold_start_results = parallel_evaluate_model(
  #    test_users=test_user_ids,
  #    user_to_books_df=user_to_books_df,
  #    num_users_to_evaluate=10,
  #    is_cold_start=True
  #)

  # Save results
  #with open('evaluation_results.pkl', 'wb') as f:
  #    pickle.dump(evaluation_results, f)

  #with open('cold_start_results.pkl', 'wb') as f:
  #    pickle.dump(cold_start_results, f)

  #print("\nCold Start Evaluation Results:")
  #for metric, value in cold_start_results.items():
  #  print(f"{metric}: {value:.4f}")

  # Function to compare regular and cold start results
  def compare_results(regular_results, cold_start_results):
    print("\nComparison of Regular and Cold Start Results:")
    print("{:<15} {:<15} {:<15} {:<15}".format("Metric", "Regular", "Cold Start", "Difference"))
    print("-" * 60)
    for metric in regular_results.keys():
      regular = regular_results[metric]
      cold = cold_start_results[metric]
      diff = regular - cold
      print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format(metric, regular, cold, diff))

  # Compare regular and cold start results
  compare_results(evaluation_results, cold_start_results)

  # Visualize the comparison
  import matplotlib.pyplot as plt

  def plot_comparison(regular_results, cold_start_results):
    metrics = list(regular_results.keys())
    regular_values = [regular_results[m] for m in metrics]
    cold_start_values = [cold_start_results[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], regular_values, width, label='Regular')
    ax.bar([i + width/2 for i in x], cold_start_values, width, label='Cold Start')

    ax.set_ylabel('Score')
    ax.set_title('Comparison of Regular and Cold Start Results')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    plt.close()

  # Plot the comparison
  plot_comparison(evaluation_results, cold_start_results)

  print("\nComparison plot saved as 'comparison_plot.png'")

  # Save all results
  all_results = {
    'regular_evaluation': evaluation_results,
    'cold_start_evaluation': cold_start_results,
  }

  with open('all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

  print("\nAll results saved in 'all_results.pkl'")

'''

i want to use 14 of my cores to generate recommendations for the evaluation functions : `evaluate_model` and `evaluate_cold_start`
how do I use multiprocessing pools for this
'''
