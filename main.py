from prompt import create_prompt
from embedding import create_embedding
from cosine_similar_vectors import cosine_similar_vectors
from llm_response import generate_response

if __name__ == "__main__":
   
    question = input("\nAsk a question : ")

    question_embedding = create_embedding([question])[0]

    new_df = cosine_similar_vectors(question_embedding)
    # print(new_df)
    prompt = create_prompt(question,new_df,10)
    
    response = generate_response(prompt)
    
    print("\nResponse : ", response)