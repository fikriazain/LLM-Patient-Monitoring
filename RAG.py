from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import time
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

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
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# satu cell sendiri di ipynb
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
# ------------------------------------------------------------------------------------

CHROMA_PATH = "final_test/chroma_test2"
retriever = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding).as_retriever(search_kwargs={"k": 20})

def rag_with_reranking(query : str):
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    results = compression_retriever.invoke(query)

    print(results)
    return results

# example
results = rag_with_reranking("apa penyabab kerusakan ginjal pada anak?")
for doc in results:
    print(doc.page_content)
    print(doc.metadata)