# ===
# Books Project
# ===

from pyspark import SparkContext
import json
import pickle
import random

BOOK_JSON: str = '/scratch/rawhad/CSE572/project/data/goodreads_books.json'
TRAIN_BOOK_IDS_FP: str = '/scratch/rawhad/CSE572/project/data/train_book_ids.pkl'
TEST_BOOK_IDS_FP: str = '/scratch/rawhad/CSE572/project/data/test_book_ids.pkl'

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
USER2BOOKS_FP = '/scratch/rawhad/CSE572/project/data/user_to_books_dict.pkl'
BOOK2USERS_FP = '/scratch/rawhad/CSE572/project/data/book_to_users_dict.pkl'

spark = SparkSession.builder \
        .appName('GoodreadsDataProcessing') \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", "1") \
        .config("spark.dynamicAllocation.maxExecutors", "10") \
        .config("spark.sql.shuffle.partitions", "1200") \
        .config("spark.executor.memory", "250g") \
        .config("spark.driver.memory", "250g") \
        .config("spark.memory.offHeap.enabled",True) \
        .config("spark.memory.offHeap.size","256g") \
        .config("spark.shuffle.compress", True) \
        .config("spark.shuffle.spill.compress", True) \
        .config("spark.executor.memoryOverhead", "10g")  \
        .config("spark.driver.maxResultSize", "10g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .getOrCreate()
'''
spark = SparkSession.builder \
  .appName('GoodreadsDataProcessing') \
  .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
  .config("spark.dynamicAllocation.enabled", "false") \
  .config("spark.executor.instances", "8") \
  .config("spark.executor.memory", "32g") \
  .config("spark.executor.cores", "4") \
  .config("spark.driver.memory", "16g") \
  .config("spark.sql.shuffle.partitions", "256") \
  .config("spark.memory.offHeap.enabled", True) \
  .config("spark.memory.offHeap.size", "32g") \
  .config("spark.executor.memoryOverhead", "4g") \
  .config("spark.driver.maxResultSize", "4g") \
  .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
  .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
  .getOrCreate()
'''

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




'''
from collections import defaultdict

# Convert interactions_df to dictionaries
user_to_books = defaultdict(list)
book_to_users = defaultdict(list)

for row in tqdm(interactions_df.select('user_id', 'book_id').rdd.collect()):
  user_to_books[row['user_id']].append(row['book_id'])
  book_to_users[row['book_id']].append(row['user_id'])

#how can I do this using pyspark?
'''

from pyspark.sql.functions import collect_list
USER2BOOKS_FP = '/scratch/rawhad/CSE572/project/data/user_to_books_dict.parquet'

print('#  Creating user 2 books dict')
user_to_books_df = interactions_df.groupBy('user_id').agg(collect_list('book_id').alias('books'))
user_to_books_df.write.parquet(USER2BOOKS_FP)
exit(0)

'''
user_to_books = {row['user_id']: row['books'] for row in user_to_books_df.collect()}
batch_size = 10000  # Adjust as needed
user_to_books = {}

for start in range(0, user_to_books_df.count(), batch_size):
  chunk = user_to_books_df.limit(batch_size).collect()
  for row in chunk:
    user_to_books[row['user_id']] = row['books']

print('#  Creating book 2 users dict')
book_to_users_df = interactions_df.groupBy('book_id').agg(collect_list('user_id').alias('users'))
book_to_users = {row['book_id']: row['users'] for row in book_to_users_df.collect()}
'''

with open(USER2BOOKS_FP, 'wb') as f: pickle.dump(user_to_books, f)
with open(BOOK2USERS_FP, 'wb') as f: pickle.dump(book_to_users, f)

spark.stop()


# ===
# Book Description Embeddings Matrix 
# ===
BOOK_IDS_PKL = '/scratch/rawhad/CSE572/project/data/all_book_ids.pkl'  # book_ids with embeddings of description
EMBEDDINGS_NPY = '/scratch/rawhad/CSE572/project/data/all_embeddings.npy'  # embeddings[idx] are embeddings of book[book_ids[idx]]['desc']

# create an embedding matrix for books in training set
# laod book_ids_pkl => list[book_ids] and EMBEDDINGS_NPY => np.array(len(book_ids), 384)
# then get embedding_indices for train set and then claw out the train books embedding matrix
# ignore book ids not present
import numpy as np

# Load book IDs and embeddings
with open(BOOK_IDS_PKL, 'rb') as f: book_ids_with_embeddings = pickle.load(f)
embeddings = np.load(EMBEDDINGS_NPY)
# Create a mapping of book IDs to their embedding index
book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids_with_embeddings)}
# Get embedding indices for train set
train_embedding_indices = [book_id_to_index[book_id] for book_id in train_book_set if book_id in book_id_to_index]
# Extract embeddings for the train set
train_book_embeddings = embeddings[train_embedding_indices]
# Save train book embeddings
TRAIN_BOOK_EMBEDDINGS_NPY = '/scratch/rawhad/CSE572/project/data/train_book_embeddings.npy'
np.save(TRAIN_BOOK_EMBEDDINGS_NPY, train_book_embeddings)


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

def recommendations(user_id, book_ids):
  if len(book_ids) == 0:
    return get_top_books()
  
  top_books = [book for book in get_top_books() if book not in book_ids]
  cf_books = collaborative_filtering(user_id, book_ids)
  cb_books = content_based_filtering(book_ids)
  
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
  
  return fused_ranking[:10]


def get_top_books(read_book_ids, n=10) -> list[str]:
  book_ratings = sc.textFile(BOOK_JSON) \
                   .map(json.loads) \
                   .filter(lambda x: 'book_id' in x and 'average_rating' in x and x['book_id'] not in read_book_ids) \
                   .map(lambda x: (x['book_id'], float(x['average_rating']))) \
                   .sortBy(lambda x: x[1], ascending=False) \
                   .map(lambda x: x[0]) \
                   .take(n)
  return book_ratings



from sklearn.metrics.pairwise import cosine_distances

def content_based_filtering(book_ids) -> list[str]:
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
      if len(unique_similar_books) == 10:
        break

  return unique_similar_books



'''
def collaborative_filtering(user_id, book_ids) -> list[str]:
  # Example function to perform collaborative filtering
  # Use RDDs or DataFrames to fetch similar users and find books
  similar_users = interactions_df.filter(col('book_id').isin(book_ids) &
                                         ~col('user_id').isin([user_id])) \
                                  .select('user_id') \
                                  .distinct() \
                                  .rdd.flatMap(lambda x: x).collect()
  recommended_books = interactions_df.filter(col('user_id').isin(similar_users) &
                                             ~col('book_id').isin(book_ids)) \
                                     .groupBy('book_id') \
                                     .count() \
                                     .orderBy('count', ascending=False) \
                                     .select('book_id') \
                                     .rdd.flatMap(lambda x: x).take(10)
  return recommended_books


i would first like to convert interactions_df into 2 dictionaries, one will have user_id as key and list of book_ids as values and the other will have book_id as key and list[user_ids] as values.
And then use those to do collaborative filtering
'''


def collaborative_filtering(user_id, book_ids) -> list[str]:
  # Find similar users who have read the same books
  similar_users = set()
  for book_id in book_ids:
    if book_id in book_to_users:
      similar_users.update(book_to_users[book_id])
  similar_users.discard(user_id)  # Remove the input user

  # Count book recommendations from similar users
  book_recommendations = Counter()
  for similar_user in similar_users:
    for book in user_to_books.get(similar_user, []):
      if book not in book_ids:
        book_recommendations[book] += 1
  
  # Get top 10 recommended books
  top_recommended_books = [book for book, _ in book_recommendations.most_common(10)]
  
  return top_recommended_books
