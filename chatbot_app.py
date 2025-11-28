import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from groq import Groq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List, Optional, Any, Iterator

class ChatGroq(BaseChatModel):
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    max_tokens: int = 512
    groq_api_key: Optional[str] = None

    class Config:
        extra = "allow"

    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.7,
                 max_tokens: int = 512, groq_api_key: Optional[str] = None, **kwargs):

        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY không được tìm thấy")

        super().__init__(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=groq_api_key,
            **kwargs
        )

        object.__setattr__(self, '_client', Groq(api_key=self.groq_api_key))

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None, **kwargs) -> ChatResult:
        groq_messages = []
        for msg in messages:
            role = "user" if not isinstance(msg, AIMessage) else "assistant"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            groq_messages.append({"role": role, "content": content})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=groq_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            **kwargs
        )

        ai_message = AIMessage(content=response.choices[0].message.content)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "groq"

import time

load_dotenv()

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def load_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=512,
            groq_api_key=groq_key
        )

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    custom_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.

INSTRUCTIONS:
- Answer questions based on the information provided in the context below
- If the question is clearly NOT related to the context or if you cannot find any relevant information in the context, respond with: "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu. Vui lòng hỏi về nội dung có trong tài liệu."
- If the context contains relevant information (even if partial), try to answer based on what is available
- DO NOT make up information that is not in the context
- Always answer in Vietnamese

Context: {context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=custom_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        if not docs or len(docs) == 0:
            return "NO_RELEVANT_CONTEXT_FOUND"
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    st.session_state.retriever = retriever

    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

def add_message(role, content):
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

def clear_chat():
    st.session_state.chat_history = []

def display_chat():
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("Xin chào! Tôi là AI assistant. Hãy upload file PDF và bắt đầu đặt câu hỏi về nội dung tài liệu nhé! 😊")

def main():
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("PDF RAG Assistant")
    # st.logo("./logo.png", size="large")

    with st.sidebar:
        st.title("⚙️ Cài đặt")

        if not st.session_state.models_loaded:
            st.info("🔄 Đang kết nối API...")
            with st.spinner("Đang khởi tạo API connections..."):
                st.session_state.embeddings = load_embeddings()
                st.session_state.llm = load_llm()
                st.session_state.models_loaded = True
            st.success("✅ API đã sẵn sàng!")
            st.rerun()
        else:
            st.success("✅ API đã sẵn sàng!")

        st.markdown("---")

        st.subheader("📄 Upload tài liệu")
        uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")

        if uploaded_file:
            if st.button("🔄 Xử lý PDF", use_container_width=True):
                with st.spinner("Đang xử lý PDF..."):
                    st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    clear_chat()
                    add_message("assistant", f"✅ Đã xử lý thành công file **{uploaded_file.name}**!\n\n📊 Tài liệu được chia thành {num_chunks} phần. Bạn có thể bắt đầu đặt câu hỏi về nội dung tài liệu.")
                st.rerun()

        if st.session_state.pdf_processed:
            st.success(f"📄 Đã tải: {st.session_state.pdf_name}")
        else:
            st.info("📄 Chưa có tài liệu")

        st.markdown("---")

        st.subheader("💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            clear_chat()
            st.rerun()

        st.markdown("---")

        st.subheader("📋 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. **Upload PDF** - Chọn file và nhấn "Xử lý PDF"
        2. **Đặt câu hỏi** - Nhập câu hỏi trong ô chat
        3. **Nhận trả lời** - AI sẽ trả lời dựa trên nội dung PDF
        """)

    st.markdown("*Trò chuyện với Chatbot để trao đổi về nội dung tài liệu PDF của bạn*")

    chat_container = st.container()

    with chat_container:
        display_chat()

    if st.session_state.models_loaded:
        if st.session_state.pdf_processed:
            user_input = st.chat_input("Nhập câu hỏi của bạn...")

            if user_input:
                add_message("user", user_input)

                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        try:
                            relevant_docs = st.session_state.retriever.invoke(user_input)

                            output = st.session_state.rag_chain.invoke(user_input)

                            if 'Answer:' in output:
                                answer = output.split('Answer:')[1].strip()
                            else:
                                answer = output.strip()

                            st.write(answer)

                            add_message("assistant", answer)

                        except Exception as e:
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info("🔄 Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info("⏳ Đang tải AI models, vui lòng đợi...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled=True)

if __name__ == "__main__":
    main()
