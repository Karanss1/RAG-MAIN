# app.py - Streamlit Interface and Ollama Client

import streamlit as st
import requests
import json
import os
import tempfile
import time

# --- Import Core Logic ---
try:
    from vector_logic import create_vector_store, TfidfRetriever
except ImportError:
    st.error("Error: Could not import core RAG logic. Ensure 'vector_logic.py' is in the same directory.")
    st.stop()
    
# --- Conditional Import for Voice Input ---
try:
    from streamlit_mic_recorder import speech_to_text
    STT_ENABLED = True
except ImportError:
    def speech_to_text(*args, **kwargs):
        # Placeholder function for graceful degradation
        return ""
    STT_ENABLED = False


# --- CONFIGURATION CONSTANTS ---
class Config:
    OLLAMA_MODEL = "llama3.2"
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    PAGE_TITLE = "Professional RAG Q&A Interface"
    FILE_TYPES = ["pdf", "docx", "txt", "csv", "xlsx"]
    MAX_DOC_CHUNKS = 3

# --- BACKEND INTERACTION CLASS ---

class RAGClient:
    """Encapsulates the core logic for communicating with Ollama and performing RAG."""
    def __init__(self, retriever: TfidfRetriever = None):
        self.retriever = retriever

    def call_ollama_api_stream(self, prompt_text: str, model_name: str = Config.OLLAMA_MODEL):
        """Streams text generation from the local Ollama API."""
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": True,
            "options": {"temperature": 0.05}
        }
        headers = {'Content-Type': 'application/json'}

        try:
            with requests.post(Config.OLLAMA_API_URL, headers=headers, json=payload, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        try:
                            json_chunk = json.loads(chunk.decode('utf-8'))
                            if "response" in json_chunk:
                                yield json_chunk["response"]
                            if json_chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            pass
        except requests.exceptions.ConnectionError:
            yield f"Ollama Error: Could not connect to Ollama server at {Config.OLLAMA_API_URL}. Is Ollama running and is '{model_name}' pulled?"
        except requests.exceptions.RequestException as e:
            yield f"Ollama API Error: Failed request: {e}"
        except Exception as e:
            yield f"An unexpected error occurred during Ollama call: {e}"

    def get_rag_context(self, query: str) -> str:
        """Retrieves relevant document chunks and formats them for the LLM."""
        if not self.retriever:
            raise ValueError("Document index is not loaded. Please upload a file first.")
            
        docs = self.retriever.get_relevant_docs(query, top_k=Config.MAX_DOC_CHUNKS)
        
        if not docs:
            return "No relevant context found in the uploaded document."

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            context_parts.append(f"--- Context Chunk {i+1} (Source: {source}) ---\n{doc.page_content}")
            
        return "\n\n".join(context_parts)

    def generate_rag_answer(self, query: str, context: str):
        """Generates the RAG answer based strictly on the retrieved context."""
        rag_prompt = f"""
You are a helpful and meticulous RAG assistant. Your sole purpose is to answer the user's question, which is provided below. You MUST base your answer ONLY on the provided context. 
If the answer cannot be found in the context, you must politely state that the information is not available in the provided document.

Context:
{context}

Question: {query}

Answer:"""
        return self.call_ollama_api_stream(rag_prompt)

    def generate_general_insights(self, query: str):
        """Generates general, creative insights not restricted by the document."""
        general_prompt = f"""
Based on the question: '{query}', provide related general knowledge, alternative perspectives, or actionable insights. 
Do NOT reference the document content. Be creative, informative, and helpful."""

        return self.call_ollama_api_stream(general_prompt)

# --- STREAMLIT UI AND STATE MANAGEMENT ---

def init_session_state():
    """Initializes all necessary session state variables."""
    if "rag_client" not in st.session_state:
        st.session_state.rag_client = RAGClient()
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = "Awaiting document upload."
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    if "stt_text" not in st.session_state:
        st.session_state.stt_text = ""
    if "stt_enabled" not in st.session_state:
        st.session_state.stt_enabled = STT_ENABLED


def handle_file_upload(uploaded_file):
    """Processes the uploaded file, creates the vector store, and updates state."""
    if uploaded_file is None:
        st.session_state.upload_status = "Please select a file to upload."
        return

    st.session_state.upload_status = f"Processing {uploaded_file.name}..."
    status_placeholder = st.empty()
    status_placeholder.info(st.session_state.upload_status)
    
    file_path = None
    try:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        # Process file (create vector store) via imported logic
        retriever = create_vector_store(file_path)
        
        # Update session state
        st.session_state.rag_client = RAGClient(retriever)
        st.session_state.upload_status = f"Document indexed successfully. Ready to answer questions based on **{uploaded_file.name}**."

    except Exception as e:
        st.session_state.upload_status = f"Error during file indexing: {e}"
    finally:
        status_placeholder.empty()
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
    st.session_state.file_uploader_key += 1

def process_query(final_query: str):
    """Executes the RAG and General Insight queries using the RAGClient."""
    if not final_query.strip():
        return

    client = st.session_state.rag_client
    
    if client.retriever is None:
        st.error("Please upload and index a document first to enable Q&A.")
        return

    st.session_state.query_history.append({"role": "user", "content": final_query})
    
    with st.status("Generating Answers...", expanded=True) as status_box:
        try:
            # 1. Retrieval
            status_box.update(label="Retrieving relevant document context...")
            context = client.get_rag_context(final_query)
            
            response_container = st.container()

            # 2. RAG Generation
            status_box.update(label="Generating RAG-based answer (Streaming from Ollama)...")
            with response_container.expander("Document-Grounded Answer", expanded=True):
                rag_response_text = st.write_stream(client.generate_rag_answer(final_query, context))
                st.session_state.query_history.append({"role": "rag", "content": rag_response_text})
            
            # 3. General Insights Generation
            status_box.update(label="Generating General Knowledge Insights (Streaming from Ollama)...")
            
            with response_container.expander("General Knowledge Insights"):
                insights_response_text = st.write_stream(client.generate_general_insights(final_query))
                st.session_state.query_history.append({"role": "general", "content": insights_response_text})
                
            status_box.update(label="Answers Generated.", state="complete")
            
        except ValueError as e:
            st.error(f"Configuration Error: {e}")
            status_box.update(label=f"Failed: {e}", state="error")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            status_box.update(label=f"Failed: {e}", state="error")


def render_chat_history():
    """Renders the previous conversation history."""
    for message in st.session_state.query_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "rag":
            with st.expander("**Document-Grounded Answer** (Click to view)", expanded=True):
                st.info(message["content"])
        elif message["role"] == "general":
            with st.expander("**General Knowledge Insights** (Click to view)"):
                st.markdown(message["content"])


def main_app():
    """The main Streamlit application function."""
    init_session_state()

    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    st.title(Config.PAGE_TITLE)
    st.markdown("---")

    # --- Sidebar: Document Management ---
    with st.sidebar:
        st.header("Document Indexing")
        
        uploaded_file = st.file_uploader(
            "Upload your document (.pdf, .docx, .xlsx, .csv, .txt)",
            type=Config.FILE_TYPES,
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )

        if st.button("Index Document", type="primary", use_container_width=True):
            handle_file_upload(uploaded_file)
                
        st.markdown(f"**Status:** {st.session_state.upload_status}")
        st.markdown("---")
        st.caption(f"LLM: **{Config.OLLAMA_MODEL}** | API: `{Config.OLLAMA_API_URL}`")

    # --- Main Content: Q&A Interface ---
    
    render_chat_history()
    
    # 1. Voice Input Component (conditional)
    st.subheader("Voice Input")
    col_mic, col_disp = st.columns([1, 4])
    
    if st.session_state.stt_enabled:
        with col_mic:
            st.session_state.stt_text = speech_to_text(
                language='en',
                start_prompt="Start Recording",
                stop_prompt="Stop Recording",
                just_once=True,
                use_container_width=True,
                key='STT_Recorder'
            )
        
        with col_disp:
            if st.session_state.stt_text:
                st.info(f"**Transcribed Text:** {st.session_state.stt_text}")
            else:
                st.markdown("*(Use microphone for transcription)*")
    else:
        st.warning("Voice input is disabled (missing `streamlit-mic-recorder`).")


    # 2. Text Input (Main Chat Interface)
    st.subheader("Query Input")
    typed_query = st.chat_input("Enter your question about the document...")

    # --- Query Execution Logic ---
    final_query = ""
    
    if typed_query:
        final_query = typed_query
        st.session_state.stt_text = ""
    elif st.session_state.stt_text:
        final_query = st.session_state.stt_text
        st.session_state.stt_text = ""

    if final_query:
        process_query(final_query)


if __name__ == "__main__":
    main_app()