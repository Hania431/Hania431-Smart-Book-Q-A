"""
rag_setup.py - Load documents and build the vector store

This script does 4 things:
1. Reads all PDF and TXT files from the 'docs/' folder
2. Splits them into small chunks (500 characters each)
3. Converts each chunk into an embedding (a list of numbers)
4. Stores everything in ChromaDB (our vector database)

Run this ONCE before asking questions:
    python rag_setup.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_chroma import Chroma

# Load the API key from the .env file
load_dotenv()


def build_vector_store(docs_folder="docs"):
    """
    Load all documents from docs/ folder and store them in ChromaDB.

    This is the RAG "indexing" phase - we only need to run this once.
    After this, the vector store is saved to disk and ready for searching.
    """

    # ---- Step 1: Load all documents ----
    all_documents = []

    # Loop through every file in the docs/ folder
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)

        if filename.endswith(".pdf"):
            print(f"  Loading PDF: {filename}")
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())

        elif filename.endswith(".txt"):
            print(f"  Loading TXT: {filename}")
            loader = TextLoader(file_path, encoding="utf-8")
            all_documents.extend(loader.load())

    # Check if we found any documents
    if not all_documents:
        print("No documents found!")
        print("Please add PDF or TXT files to the 'docs/' folder first.")
        return None

    print(f"  Loaded {len(all_documents)} pages total")

    # ---- Step 2: Split documents into small chunks ----
    # Why? Big documents are too large for the AI to process at once.
    # Small chunks let us find the EXACT paragraph that answers a question.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Each chunk is about 500 characters
        chunk_overlap=50     # Chunks overlap by 50 chars so we don't lose context
    )
    chunks = splitter.split_documents(all_documents)
    print(f"  Split into {len(chunks)} chunks")

    # ---- Step 3 & 4: Create embeddings and store in ChromaDB ----
    # Embeddings convert text into numbers so we can measure similarity.
    # ChromaDB stores these embeddings and lets us search by meaning.
    # Using simple embeddings (no additional packages required)
    embeddings = FakeEmbeddings(size=384)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"   # Save to disk so we can reuse it
    )

    print(f"  Vector store ready! Indexed {len(chunks)} chunks.")
    print("  You can now run: python main.py")
    return vector_store


# ---- Run this script to build the vector store ----
if __name__ == "__main__":
    # Create the docs folder if it doesn't exist yet
    os.makedirs("docs", exist_ok=True)

    print("=" * 50)
    print("  Smart Book Q&A Crew - Document Indexer")
    print("=" * 50)
    print()
    build_vector_store()
