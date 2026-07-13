import json
import os
import pandas as pd
from embedding import create_embedding


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    batch_size = 100
    embeddings = []
    count=0
    for i in range(0, len(content["chunks"]), batch_size):
        print(f"batch {count}")
        count+=1
        batch = [c["text"] for c in content["chunks"][i:i + batch_size]]
        batch_embeddings = create_embedding(batch)
        embeddings.extend(batch_embeddings)
        
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
df.to_pickle("data/embeddings.pkl")
print("successfully created embeddings.pkl file")