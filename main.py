import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Document Loader & Vector Store
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Model Providers (Gemini 2.5 Flash & HuggingFace Embeddings)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Core Components (Bebas dari langchain.chains)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Multi-Doc Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Document Research Assistant")
st.caption("Unggah beberapa dokumen PDF dan tanyakan informasi spesifik tentang informasi yang tersedia.")

# ---------------------------------------------------------
# Session State & Embedding Cache
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def get_embeddings():
    """Cache model embedding lokal agar cepat."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def process_pdfs(uploaded_files):
    """Membaca PDF, memotong teks (chunking), dan menyimpan ke Vector Store."""
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        
        documents.extend(docs)
        os.remove(tmp_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    return vector_store

def run_rag_pipeline(vector_store, user_query, chat_history):
    """Eksekusi RAG menggunakan LCEL Pipeline."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # 1. Retrieval: Cari 4 dokumen paling relevan
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(user_query)
    
    # mengabungkan teks dokumen menjadi konteks
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 2. Prompting
    system_prompt = (
        "Anda adalah asisten peneliti profesional. Gunakan potongan konteks berikut "
        "untuk menjawab pertanyaan. Jika Anda tidak tahu jawabannya, katakan bahwa "
        "Anda tidak tahu. Gunakan maksimal 3 kalimat agar jawaban tetap ringkas dan padat.\n\n"
        "Konteks Dokumen:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # 3. Chain Building dengan LCEL (| Operator)
    chain = prompt | llm | StrOutputParser()
    
    # 4. Invoke Chain
    answer = chain.invoke({
        "context": context_text,
        "chat_history": chat_history,
        "input": user_query
    })
    
    return answer, relevant_docs

# ---------------------------------------------------------
# Sidebar: Upload PDF
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 Dokumen")
    uploaded_files = st.file_uploader(
        "Unggah berkas PDF:",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Proses Dokumen", type="primary"):
        if uploaded_files:
            with st.spinner("Memproses & Indeks Dokumen..."):
                st.session_state.vector_store = process_pdfs(uploaded_files)
                st.success("Dokumen berhasil diproses!")
        else:
            st.warning("Silakan unggah minimal satu berkas PDF.")

# ---------------------------------------------------------
# Main UI: Chat Area
# ---------------------------------------------------------
# Menampilkan Riwayat Percakapan
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input Pertanyaan Baru
if user_query := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
    if not st.session_state.vector_store:
        st.error("Silakan unggah dan proses dokumen PDF terlebih dahulu di sidebar!")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban dari dokumen..."):
                answer, sources = run_rag_pipeline(
                    st.session_state.vector_store,
                    user_query,
                    st.session_state.chat_history
                )
                
                st.markdown(answer)
                
                # Tampilkan Sumber Kutipan
                with st.expander("📌 Lihat Sumber Kutipan"):
                    for i, doc in enumerate(sources, 1):
                        source_name = doc.metadata.get("source", "Tidak diketahui")
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**[{i}] Berkas:** `{source_name}` | **Halaman:** `{page_num}`")
                        st.caption(f'"{doc.page_content[:200]}..."')
        
        # Simpan ke Riwayat Percakapan
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))