import pandas as pd
import numpy as np
import faiss



# Load DataFrame
df = pd.read_pickle("data/embeddings.pkl")
index = faiss.read_index("data/vector_index.faiss")


def cosine_similar_vectors(question, top_k=10):
    question = np.asarray(question, dtype=np.float32).reshape(1, -1)
    D, I = index.search(question, k=top_k)
    return df.iloc[I[0]].copy()  # Return a copy of the DataFrame rows corresponding to the indices
    # for i, idx in enumerate(I[0]):
    #     print("-" * 50)
    #     print("Row:", idx)
    #     print("Distance:", D[0][i])
    #     print(df.iloc[idx][["title","text"]])
    #print(df.iloc[I[0]].shape)
    