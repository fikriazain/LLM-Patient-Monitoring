from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from llm_client_chat import AlpacaLLM
from llm_client import AlpacaLLM

import time

# CHROMA_PATH = "langchain-loader/chroma-multilingual_e5_large-semantic-split"
# CHROMA_PATH = "unstructured-loader/chroma-multilingual_e5_basic_split"
CHROMA_PATH = "final_test/chroma_test"
CHROMA_PATH = "final_test/chroma_test2"

# DATA_PATH = "data/pdfs"
# os.environ['OPENAI_API_KEY'] = "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"

PROMPT_TEMPLATE = """
Jawab pertanyaan hanya menggunakan informasi di bawah ini, suatu informasi mungkin tidak mengandung informasi yang relevan dengan pertanyaan, masing-masing informasi dipisahkan oleh '---':

Pertanyaan : {question}
---

{context}

---
"""

def load_embedding_model():
    start_time = time.time()
    model_path="intfloat/multilingual-e5-large"   
    encode_kwargs = {"normalize_embeddings": True}
    local_embedding = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder="./models",
        encode_kwargs=encode_kwargs
    )
    end_time = time.time()
    print(f'model load time {round(end_time - start_time, 0)} second')
    return local_embedding

embedding = load_embedding_model()
# embedding = OpenAIEmbeddings()

chat = True
while chat:
    # Accept user input
    user_prompt = input("User > ")
    if user_prompt == "exit":
        chat = False
        continue
    else :
        pass
    
    # load DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(user_prompt, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Similarity too low.", end="\n")
        # return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=user_prompt)

    # print(prompt)
    counter = 1
    print("-----------------------------------------------------")
    print(user_prompt)
    for doc, _score in results:
        print(f"{counter}. doc: \n{doc.page_content}, \nmetadata: {doc.metadata}, \nscore: {_score}\n")
        counter+=1
    print("-----------------------------------------------------")
    print()

    # model = AlpacaLLM()
    # response_text = model.invoke(prompt)

    # # model = ChatOpenAI()
    # # response_text = model.predict(prompt)

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # print()
    # formatted_response = f"AI > Response: {response_text}\nSources: {sources}"
    # # formatted_response = f"Response: {response_text}"
    # print(formatted_response)