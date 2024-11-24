from pyspark.sql import SparkSession

# Initialize Spark session
spark: SparkSession = SparkSession.builder \
    .appName("CSV to Parquet Converter") \
    .getOrCreate()

# Load CSV file
df = spark.read.csv("/scratch/rawhad/CSE572/project/data/goodreads_interactions.csv", header=True, inferSchema=True)

# Select rows where is_read is true
df_filtered = df.filter(df.is_read == 1)

# print head
df_filtered.show()

# Write to Parquet
df_filtered.write.parquet("/scratch/rawhad/CSE572/project/data/goodreads_interactions.parquet")

# Stop the Spark session
spark.stop()
