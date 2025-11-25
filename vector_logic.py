# vector_logic.py - Core RAG Logic and Document Processing

import os
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Conditional/Optional Imports for File Reading ---
# These libraries are often required for professional document handling.
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import pandas as pd
except ImportError:
    pd = None


class TfidfRetriever:
    """
    A TF-IDF based document chunk retriever for RAG systems.
    
    Manages document vectorization and performs cosine similarity search 
    to retrieve the most relevant context for a given query.
    """
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.texts = [doc.page_content for doc in docs]
        self.vectorizer = TfidfVectorizer(stop_words='english')

        if not self.texts or all(not text.strip() for text in self.texts):
            raise ValueError("No meaningful text found for TF-IDF vectorization.")

        try:
            self.doc_vectors = self.vectorizer.fit_transform(self.texts)
            if not hasattr(self.vectorizer, 'stop_words_'):
                self.vectorizer.stop_words_ = self.vectorizer.get_stop_words() or set()
        except ValueError as e:
            raise ValueError(f"Error during TF-IDF vectorization: {e}. Check document text quality.")

    def get_relevant_docs(self, query: str, top_k: int = 3) -> list[Document]:
        """
        Retrieves the top_k most relevant documents for a given query.
        
        Args:
            query (str): The user's input question.
            top_k (int): The number of top documents to retrieve.
            
        Returns:
            list[Document]: A list of the most relevant LangChain Document objects.
        """
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
            return []

        # Pre-process query to match vectorizer's treatment of stop words
        current_stop_words = self.vectorizer.stop_words_ if hasattr(self.vectorizer, 'stop_words_') else set()
        processed_query_words = [word for word in query.lower().split() if word not in current_stop_words]
        processed_query = ' '.join(processed_query_words)

        if not processed_query.strip():
            return []

        query_vec = self.vectorizer.transform([processed_query])
        
        if query_vec.sum() == 0:
            # Query contains only OOC vocabulary or stop words
            return []

        # Calculate cosine similarity
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [self.docs[i] for i in top_indices]

def read_file(file_path: str) -> str:
    """Reads content from various document types based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            if pdfplumber:
                with pdfplumber.open(file_path) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            else:
                raise ImportError("`pdfplumber` is required for PDF files.")
        elif ext == ".docx":
            if DocxDocument:
                doc = DocxDocument(file_path)
                return "\n".join(p.text for p in doc.paragraphs)
            else:
                raise ImportError("`python-docx` is required for DOCX files.")
        elif ext in [".csv", ".xlsx"]:
            if pd:
                return pd.read_csv(file_path).to_string() if ext == ".csv" else pd.read_excel(file_path).to_string()
            else:
                raise ImportError("`pandas` is required for CSV/XLSX files.")
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return f"Error: Unsupported file type '{ext}'."
    except ImportError as ie:
        return f"Error: Missing dependency for {ext} file type. {ie}"
    except Exception as e:
        return f"Error reading file: {e}"

def create_vector_store(file_path: str) -> TfidfRetriever:
    """Reads file, chunks content, and initializes the TfidfRetriever."""
    text = read_file(file_path)
    if not text or text.startswith("Error"):
        # Re-raise the error for the Streamlit app to catch
        raise ValueError(text if text.startswith("Error") else "No text extracted from document.")

    # Chunk the text using LangChain's RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Create LangChain Document objects with source metadata
    docs = splitter.create_documents([text], metadatas=[{"source": os.path.basename(file_path)}])

    return TfidfRetriever(docs)