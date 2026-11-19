import numpy as np
import faiss
import time
np.random.seed(42)
data = np.random.random((50000, 128)).astype("float32")
query = np.random.random((1, 128)).astype("float32")
k = 10
data_normalized = data.copy()
query_normalized = query.copy()
faiss.normalize_L2(data_normalized)
faiss.normalize_L2(query_normalized)
cosine_index = faiss.IndexFlatIP(128)
cosine_index.add(data_normalized)
start = time.time()
cosine_D, cosine_I = cosine_index.search(query_normalized, k)
cosine_time = (time.time() - start) * 1000
print("\nCosine Similarity")
print("Time:", round(cosine_time, 3), "ms")
print("Top 10:", cosine_I[0])
l2_index = faiss.IndexFlatL2(128)
l2_index.add(data)
start = time.time()
l2_D, l2_I = l2_index.search(query, k)
l2_time = (time.time() - start) * 1000
print("\nL2 Distance")
print("Time:", round(l2_time, 3), "ms")
print("Top 10:", l2_I[0])
ip_index = faiss.IndexFlatIP(128)
ip_index.add(data)
start = time.time()
ip_D, ip_I = ip_index.search(query, k)
ip_time = (time.time() - start) * 1000
print("\nInner Product")
print("Time:", round(ip_time, 3), "ms")
print("Top 10:", ip_I[0])
print("\nComparison")
print("Cosine Time:", round(cosine_time, 3), "ms")
print("L2 Time:", round(l2_time, 3), "ms")
print("Inner Product Time:", round(ip_time, 3), "ms")
