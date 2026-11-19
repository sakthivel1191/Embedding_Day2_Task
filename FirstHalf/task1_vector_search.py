import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
np.random.seed(42)
data = np.random.random((50000, 128)).astype("float32")
query = np.random.random((1, 128)).astype("float32")
print("Total vectors:", len(data))
print("Vector dimensions:", data.shape[1])
start = time.time()
distances = np.sum((data - query) ** 2, axis=1)
exact_top10 = np.argsort(distances)[:10]
numpy_time = (time.time() - start) * 1000
print("\nNumPy Exact Search")
print("Time:", round(numpy_time, 3), "ms")
print("Top 10:", exact_top10)
index_flat = faiss.IndexFlatL2(128)
index_flat.add(data)
start = time.time()
D, I = index_flat.search(query, 10)
flat_time = (time.time() - start) * 1000
print("\nFAISS Exact Search")
print("Time:", round(flat_time, 3), "ms")
print("Top 10:", I[0])
index_hnsw = faiss.IndexHNSWFlat(128, 32)
index_hnsw.hnsw.efConstruction = 100
index_hnsw.hnsw.efSearch = 64
index_hnsw.add(data)
start = time.time()
D_hnsw, I_hnsw = index_hnsw.search(query, 10)
hnsw_time = (time.time() - start) * 1000
print("\nFAISS HNSW Search")
print("Time:", round(hnsw_time, 3), "ms")
print("Top 10:", I_hnsw[0])
exact_set = set(I[0])
hnsw_set = set(I_hnsw[0])
recall = len(exact_set.intersection(hnsw_set)) / 10
print("\nRecall@10:", recall)
M_values = [8, 16, 32, 48]
ef_values = [16, 32, 64, 128]
times = []
recalls = []
for M in M_values:
    index = faiss.IndexHNSWFlat(128, M)
    index.hnsw.efConstruction = 100
    index.add(data)
    for ef in ef_values:
        index.hnsw.efSearch = ef
        start = time.time()
        D, I = index.search(query, 10)
        search_time = (time.time() - start) * 1000
        result_set = set(I[0])
        recall_value = len(exact_set.intersection(result_set)) / 10
        times.append(search_time)
        recalls.append(recall_value)
        print("M:", M,"efSearch:", ef,"Time:", round(search_time, 3), "ms","Recall:", recall_value)
plt.figure(figsize=(8, 5))
plt.scatter(times, recalls, color="blue")
plt.xlabel("Search Time (milliseconds)")
plt.ylabel("Recall@10")
plt.title("HNSW Recall vs Search Time")
plt.grid(True)
plt.savefig("recall_vs_time.png")
plt.show()
