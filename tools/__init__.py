import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import time
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from bot.llm_client import Mistral


os.environ["SERPER_API_KEY"] = 'TOKEN'
# CHROMA_PATH = "final_test/chroma_test"
CHROMA_PATH = "final_test/chroma_test2"


def load_embedding_model(model_path : str):
    start_time = time.time()
    encode_kwargs = {"normalize_embeddings": True}
    local_embedding = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder="./models",
        encode_kwargs=encode_kwargs
    )
    end_time = time.time()
    print(f'model load time {round(end_time - start_time, 0)} second')
    return local_embedding

embedding = load_embedding_model(model_path="intfloat/multilingual-e5-large")

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding).as_retriever(search_kwargs={"k": 20})

model_llm_rag = Mistral()

