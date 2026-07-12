from ollama import embed

def create_embedding(text_list):
    response = embed(
        model="nomic-embed-text",
        input=text_list
    )

    #print(response)

    return response["embeddings"]