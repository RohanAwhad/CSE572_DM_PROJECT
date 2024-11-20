### Dataset Information

1. authors_1000.csv
    - Cols: author_id,name,average_rating,text_reviews_count,ratings_count

2. interactions_1000.csv
    - Cols: user_id,book_id,is_read,rating,is_reviewed

3. books_1000.json
    {
      "isbn": "0312853122",
      "text_reviews_count": "1",
      "series": [],
      "country_code": "US",
      "language_code": "",
      "popular_shelves": [
        {"count": "3", "name": "to-read"},
        {"count": "1", "name": "p"},
        {"count": "1", "name": "collection"},
        {"count": "1", "name": "w-c-fields"},
        {"count": "1", "name": "biography"}
      ],
      "asin": "",
      "is_ebook": "false",
      "average_rating": "4.00",
      "kindle_asin": "",
      "similar_books": [],
      "description": "",
      "format": "Paperback",
      "link": "https://www.goodreads.com/book/show/5333265-w-c-fields",
      "authors": [{"author_id": "604031", "role": ""}],
      "publisher": "St. Martin's Press",
      "num_pages": "256",
      "publication_day": "1",
      "isbn13": "9780312853129",
      "publication_month": "9",
      "edition_information": "",
      "publication_year": "1984",
      "url": "https://www.goodreads.com/book/show/5333265-w-c-fields",
      "image_url": "https://images.gr-assets.com/books/1310220028m/5333265.jpg",
      "book_id": "5333265",
      "ratings_count": "3",
      "work_id": "5400751",
      "title": "W.C. Fields: A Life on Film",
      "title_without_series": "W.C. Fields: A Life on Film"
    }

4. genre_1000.json
    - {"book_id": "5333265", "genres": {"history, historical fiction, biography": 1}}


### Ideas

1. What if join everything and flatten the entire dataset, and then do analysis on that?
    - Do we have enough time to do that?
    - Is it worth it? Would doing that help us save time?
    - Can we then process everything in parallel? Meaning can we generate book description embedding, thumbnail embedding, tf-idf vectors all in parallel?
    - At this phase learning is more important and getting hands on knowledge with Spark is also important.
    - Also, what kind of a machine would we require? From a hardware perspective?
    - Can we connect all of our machines together to run distributed Spark Jobs?  


2. Expanded Dataset would have everything in Books
    - authors information will be filled in authors
    - genres dict will also be moved to the appropriate books -> this has to use either TF-IDF or UNION of embeddings that fall within a radius of query embedding or DBSCAN with it.
    - interatctions will be another key in books, which will be a list of interaction_dict. Each dict will basically be the interaction csv with cols as keys

* This would allow us to create a master dataset. From which we can create smaller datasets as needed. Because this might not be consumed directly. Also the data pipelines would also be worth some points right.
