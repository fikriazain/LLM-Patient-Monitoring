import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma


os.environ["SERPER_API_KEY"] = 'TOKEN'
CHROMA_PATH = "final_test/chroma_test"
CHROMA_PATH = "final_test/chroma_test2"


def load_embedding_model():
    # start_time = time.time()
    model_path="intfloat/multilingual-e5-large"   
    encode_kwargs = {"normalize_embeddings": True}
    local_embedding = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder="./models",
        encode_kwargs=encode_kwargs
    )
    return local_embedding

embedding = load_embedding_model()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)



