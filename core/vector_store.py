import os 
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

CHROMA_DIR = "Vector_DB"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": 'cpu'}
    )

def build_vector_store(transcript: str) -> Chroma:
    print("Building Vector Store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store

def get_retriever(vector_store: Chroma, transcript: str = None, k: int = 6):
    """Upgrades retrieval to Hybrid Search by combining BM25 Keyword matching and Dense Vector matching."""
    chroma_retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )
    
    # If the transcript text is passed, build a high-performance Hybrid Ensemble
    if transcript:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = splitter.split_text(transcript)
        
        bm25_retriever = BM25Retriever.from_texts(chunks)
        bm25_retriever.k = k
        
        # Mix BM25 Keyword Search (40% weight) with Semantic Vector Search (60% weight)
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.4, 0.6]
        )
        return hybrid_retriever
        
    return chroma_retriever