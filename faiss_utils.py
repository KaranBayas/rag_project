import numpy as np
import faiss

def dataframe_to_faiss(df):
    
    embedding_matrix = np.array(df["embedding"].tolist()).astype("float32")
    dimension = 768
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    faiss.write_index(index, "data/vector_index.faiss")
    print("[dataframe -> faiss] mission completed!!!!!!")
