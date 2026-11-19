
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language",
    "Java is used for software development",
    "Machine learning is a part of artificial intelligence",
    "I enjoy learning Python programming",
    "Deep learning uses neural networks",
    "Python is easy to learn"
]

embeddings = model.encode(sentences)

query = input("Enter your search text: ")

query_embedding = model.encode([query])

similarity_scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]

results = sorted(
    zip(sentences, similarity_scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nSearch Results:\n")

for text, score in results:
    print("Text:", text)
    print("Similarity Score:", round(float(score), 4))
    print()
