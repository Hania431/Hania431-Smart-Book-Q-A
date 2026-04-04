"""
app.py - Streamlit UI for Smart Book Q&A CrewAI RAG System

This application provides a user-friendly interface for:
1. Uploading PDF or TXT documents
2. Automatically building the vector store
3. Asking questions about uploaded documents
4. Viewing answers and agent thought processes

Usage:
    streamlit run app.py
"""

import os
from dotenv import load_dotenv

# CRITICAL: Load environment variables FIRST before any other imports
load_dotenv()

# Load from Streamlit secrets if available
try:
    import streamlit as st
   os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
   os.environ["GEMINI_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
   os.environ["GOOGLE_GEMINI_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    os.environ["GEMINI_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

# Now import after environment variables are set
import tempfile
import shutil
import streamlit as st
from rag_setup import build_vector_store


def get_crew_result(question):
    """Lazy import of run_crew to avoid initialization issues."""
    from main import run_crew
    return run_crew(question)

# Page configuration
st.set_page_config(
    page_title="Smart Book Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "vector_store_built" not in st.session_state:
    st.session_state.vector_store_built = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "crew_result" not in st.session_state:
    st.session_state.crew_result = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []


def save_uploaded_file(uploaded_file, docs_folder="docs"):
    """Save uploaded file to the docs folder."""
    os.makedirs(docs_folder, exist_ok=True)
    file_path = os.path.join(docs_folder, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def clear_docs_folder():
    """Clear all files from the docs folder."""
    docs_folder = "docs"
    if os.path.exists(docs_folder):
        for filename in os.listdir(docs_folder):
            file_path = os.path.join(docs_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.error(f"Error deleting {file_path}: {e}")


def build_index():
    """Build the vector store and update session state."""
    with st.spinner("Building document index... This may take a moment."):
        result = build_vector_store()
        if result is not None:
            st.session_state.vector_store_built = True
            st.success("✅ Vector store built successfully!")
        else:
            st.error("Failed to build vector store.")


def process_question(question):
    """Process the question through the crew and capture results."""
    with st.spinner("🤖 AI agents are working on your question..."):
        # Capture verbose output by redirecting print statements
        import io
        import sys
        
        log_capture = io.StringIO()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = log_capture
            result = get_crew_result(question)
            logs = log_capture.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return result, logs


# ============================================================
#  MAIN UI LAYOUT
# ============================================================

st.title("📚 Smart Book Q&A")
st.markdown("""
Ask questions about your PDF or TXT documents and get AI-powered answers!
""")

# Create two columns for the sidebar and main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Upload Documents")
    st.markdown("Upload PDF or TXT files to build your knowledge base.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Select one or more PDF or TXT files to upload"
    )
    
    # Process uploaded files
    if uploaded_files:
        new_files = [f.name for f in uploaded_files if f.name not in st.session_state.uploaded_files]
        
        if new_files:
            st.info(f"Found {len(new_files)} new file(s) to process.")
            
            if st.button("📥 Process Files & Build Index", key="process_btn"):
                # Clear previous files and rebuild
                clear_docs_folder()
                st.session_state.uploaded_files = []
                st.session_state.vector_store_built = False
                
                # Save all uploaded files
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Build the vector store
                build_index()
                progress_bar.empty()
        
        elif st.button("🔄 Rebuild Index", key="rebuild_btn"):
            build_index()
    
    # Show status
    if st.session_state.vector_store_built:
        st.success("✅ Index ready!")
        st.markdown(f"**Files indexed:** {len(st.session_state.uploaded_files)}")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- {fname}")
    else:
        st.warning("⚠️ Please upload files and build the index first.")

with col2:
    st.subheader("❓ Ask Questions")
    st.markdown("Enter your question about the uploaded documents.")
    
    if not st.session_state.vector_store_built:
        st.warning("Please upload documents and build the index first!")
        question = st.text_area(
            "Your Question",
            height=100,
            placeholder="Upload documents first to ask questions...",
            disabled=True
        )
    else:
        question = st.text_area(
            "Your Question",
            height=100,
            placeholder="E.g., What is the main topic of this document?"
        )
        
        if st.button("🚀 Run Crew", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                # Process the question
                result, logs = process_question(question)
                
                # Store results in session state
                st.session_state.crew_result = result
                st.session_state.agent_logs = logs
                
                # Display the answer
                st.success("✅ **Final Answer:**")
                st.markdown(result)
                
                # Show agent thought process in expandable section
                with st.expander("🔍 View Agent Thought Process"):
                    st.markdown("**Agent Execution Logs:**")
                    if logs:
                        st.code(logs, language="text")
                    else:
                        st.info("No detailed logs available.")
                    
                    st.markdown("---")
                    st.markdown("**How it works:**")
                    st.markdown("""
                    1. **Document Retriever** searches the vector store for relevant chunks
                    2. **Answer Writer** creates a clear answer from the retrieved information
                    3. **Quality Checker** verifies the answer against source chunks
                    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by CrewAI + LangChain + Google Gemini</small>
</div>
""", unsafe_allow_html=True)

