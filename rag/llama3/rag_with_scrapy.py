import os
import argparse
import requests
from bs4 import BeautifulSoup
from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Document,  StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Workaround for HuggingFace FastTokenizers

def fetch_html_content(url):
    """Fetches and returns the text content of the HTML from the given URL."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="URL to fetch documents from")
    parser.add_argument("--persist_dir", type=str, default="./vector_index/", help="Path to store the serialized VectorStore")
    args = parser.parse_args()

    print(f"Fetching data from {args.url}")
    html_text = fetch_html_content(args.url)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = Ollama(
        model="llama3",
        is_chat_model=True,
        temperature=0.6,
        request_timeout=120.0,
    )

    Settings.llm = llm
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    Settings.embed_model = embed_model

    # Initialize VectorStore
    if not os.path.exists(args.persist_dir):
        os.makedirs(args.persist_dir)
    document = Document(text=html_text)
    
    storage_context = StorageContext.from_defaults(
        vector_store=DeepLakeVectorStore(dataset_path="./vector_index/")
    )

    vector_store = VectorStoreIndex.from_documents([document], storage_context=storage_context)

    print("VectorStore initialized and persisted.")

    retriever = VectorIndexRetriever(vector_store)
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
    chat_engine = ContextChatEngine.from_defaults(retriever=retriever, query_engine=query_engine, verbose=True)

    # Start interactive chat session
    chat_engine.chat_repl()

if __name__ == "__main__":
    main()