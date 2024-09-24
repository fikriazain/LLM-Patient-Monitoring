from langchain.tools import BaseTool, StructuredTool, tool
from langchain.utilities import GoogleSerperAPIWrapper
import requests
import random
import json
from tools import db, reranker_model, model_llm_rag
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from deep_translator import GoogleTranslator
from langchain.prompts import ChatPromptTemplate


PROMPT_TEMPLATE = """
You are an assistant that can summarize a document and answer questions based on the document. You will be given three documents that separate by '---'. You will be asked a question based on the information in the documents. You must answer the question using only the information in the documents.
---
These is the documents:

{context}

---

Question: {question}
Answer:
"""

@tool
def send_emergency_message_to_medic(query: str) -> str: 
    """This function is used to send a message containing user symptoms to the medic where the symptoms are related to emergency cases. You must give the query semantically the same with the user input,
    You can ONLY run this function ONE time, then you must run the 'search_hemodialysis_information' tools to get user symptoms explanation."""
    url = "http://127.0.0.1:8000/message/get_message/"

    user_id = str(random.randint(1, 100))

    data = {
        "message": query,
        "user_id": user_id
    }

    #Turn data into json for the request
    data = json.dumps(data)

    response = requests.post(url, data=data)
    return ' Success sending message. Please provide search query for the symtomps that patient has.\n'

@tool
def search_information_for_question(query: str) -> str:
    """Function that searches for information based on the user query. You must use this function if there are questions related to medical topics. The query is the message that the patient send to Panda, YOU MUST NOT CHANGE IT."""
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=db
    )

    query_translate = GoogleTranslator(source='english', target='id').translate(query)
    results = compression_retriever.invoke(query)
    target = "\n\n---\n\n".join([doc.page_content for doc in results])
    context_text = GoogleTranslator(source='id', target='english').translate(target)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_translate)
    print(target)
    result = model_llm_rag.invoke(prompt)
    return GoogleTranslator(source='id', target='english').translate(result)

# @tool
# def search_medic_info(query: str) -> str:
#     results = db.similarity_search_with_relevance_scores(query, k=3)
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     return context_text

# @tool
# def search_medic_info(query: str) -> str: 
#     """Function that searches for medical information based on the user query. The query is the message that the patient send to Panda, YOU MUST NOT CHANGE IT."""
#     return GoogleSerperAPIWrapper().run(query)