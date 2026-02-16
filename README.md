### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 14-02-2026
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```py
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from tabulate import tabulate

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')   # IMPORTANT FIX
nltk.download('stopwords')

# -------------------------------
# Sample Documents
# -------------------------------
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        token for token in tokens
        if token not in stopwords.words("english")
        and token not in string.punctuation
    ]
    return " ".join(tokens)

# Preprocess Documents
preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}

# -------------------------------
# Vectorizers
# -------------------------------
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(preprocessed_docs.values())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

terms = tfidf_vectorizer.get_feature_names_out()

# -------------------------------
# Term Frequency (TF) Table
# -------------------------------
print("\n--- Term Frequencies (TF) ---\n")
tf_table = count_matrix.toarray()

print(tabulate(
    [["Doc ID"] + list(terms)] +
    [[list(preprocessed_docs.keys())[i]] + list(row)
     for i, row in enumerate(tf_table)],
    headers="firstrow",
    tablefmt="grid"
))

# -------------------------------
# Document Frequency (DF) & IDF
# -------------------------------
df = np.sum(count_matrix.toarray() > 0, axis=0)
idf = tfidf_vectorizer.idf_

df_idf_table = []
for i, term in enumerate(terms):
    df_idf_table.append([term, df[i], round(idf[i], 4)])

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")
print(tabulate(
    df_idf_table,
    headers=["Term", "Document Frequency (DF)", "Inverse Document Frequency (IDF)"],
    tablefmt="grid"
))

# -------------------------------
# TF-IDF Table
# -------------------------------
print("\n--- TF-IDF Weights ---\n")
tfidf_table = tfidf_matrix.toarray()

print(tabulate(
    [["Doc ID"] + list(terms)] +
    [[list(preprocessed_docs.keys())[i]] +
     list(map(lambda x: round(x, 4), row))
     for i, row in enumerate(tfidf_table)],
    headers="firstrow",
    tablefmt="grid"
))

# -------------------------------
# Manual Cosine Similarity
# -------------------------------
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm_vec1 * norm_vec2) \
        if norm_vec1 != 0 and norm_vec2 != 0 else 0.0

    return dot_product, norm_vec1, norm_vec2, similarity

# -------------------------------
# Search Function
# -------------------------------
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform(
        [preprocessed_query]
    ).toarray()[0]

    results = []

    for idx, doc_vector in enumerate(tfidf_matrix.toarray()):
        doc_id = list(preprocessed_docs.keys())[idx]
        doc_text = documents[doc_id]

        dot, norm_q, norm_d, sim = cosine_similarity_manual(
            query_vector,
            doc_vector
        )

        results.append([
            doc_id,
            doc_text,
            round(dot, 4),
            round(norm_q, 4),
            round(norm_d, 4),
            round(sim, 4)
        ])

    results.sort(key=lambda x: x[5], reverse=True)
    return results, query_vector

# -------------------------------
# Get User Query
# -------------------------------
query = input("\nEnter your query: ")

# Perform Search
results_table, query_vector = search(
    query,
    tfidf_matrix,
    tfidf_vectorizer
)

# -------------------------------
# Display Cosine Similarity Table
# -------------------------------
print("\n--- Search Results and Cosine Similarity ---\n")

headers = [
    "Doc ID",
    "Document",
    "Dot Product",
    "Query Magnitude",
    "Doc Magnitude",
    "Cosine Similarity"
]

print(tabulate(results_table, headers=headers, tablefmt="grid"))

# -------------------------------
# Query TF-IDF Weights
# -------------------------------
print("\n--- Query TF-IDF Weights ---\n")

query_weights = [
    (terms[i], round(query_vector[i], 4))
    for i in range(len(terms))
    if query_vector[i] > 0
]

print(tabulate(
    query_weights,
    headers=["Term", "Query TF-IDF Weight"],
    tablefmt="grid"
))

# -------------------------------
# Ranking Table
# -------------------------------
print("\n--- Ranked Documents ---\n")

ranked_docs = []
for idx, res in enumerate(results_table, start=1):
    ranked_docs.append([
        idx,
        res[0],
        res[1],
        res[5]
    ])

print(tabulate(
    ranked_docs,
    headers=["Rank", "Document ID", "Document Text", "Cosine Similarity"],
    tablefmt="grid"
))

# -------------------------------
# Highest Rank Document
# -------------------------------
highest_doc = max(results_table, key=lambda x: x[5])

print(f"\nThe highest rank cosine score is: {highest_doc[5]} (Document ID: {highest_doc[0]})")
```
### Output:

<img width="1126" height="340" alt="image" src="https://github.com/user-attachments/assets/98038c85-c12e-4329-9ff1-f338f1c03306" />

<img width="925" height="369" alt="image" src="https://github.com/user-attachments/assets/c8c48c7c-b9d1-4316-b320-84da7e4e7bd9" />

<img width="722" height="292" alt="image" src="https://github.com/user-attachments/assets/93f1beed-401b-49bb-aaf7-6cde69a9b565" />

<img width="1156" height="369" alt="image" src="https://github.com/user-attachments/assets/9247cc69-51f5-471b-9f2f-95c7644677e5" />

<img width="1116" height="308" alt="image" src="https://github.com/user-attachments/assets/418a675e-f250-4017-8283-2a6ccaddc19c" />

<img width="933" height="318" alt="image" src="https://github.com/user-attachments/assets/59e7f26d-5256-43eb-a48e-9cabd671a392" />


### Result:

Thus, the Information Retrieval system using the Vector Space Model was successfully implemented in Python, and documents were ranked based on cosine similarity using TF-IDF weighting.
