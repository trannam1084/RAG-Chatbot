# PDF RAG Chatbot

Ứng dụng chatbot thông minh sử dụng RAG (Retrieval-Augmented Generation) để trả lời câu hỏi về nội dung tài liệu PDF bằng tiếng Việt.

## 📋 Mô tả

Dự án này là một chatbot được xây dựng bằng Streamlit, sử dụng:

- **LangChain**: Framework để xây dựng ứng dụng LLM
- **Groq API**: LLM model (Llama 3.1 8B Instant) để tạo phản hồi
- **Chroma**: Vector database để lưu trữ embeddings
- **HuggingFace Embeddings**: Model tiếng Việt để tạo embeddings

## ✨ Tính năng

- 📄 Upload và xử lý file PDF
- 🔍 Semantic chunking để chia nhỏ tài liệu
- 💬 Chatbot trả lời câu hỏi dựa trên nội dung PDF
- 🇻🇳 Hỗ trợ tiếng Việt
- 🎯 RAG (Retrieval-Augmented Generation) để tìm kiếm thông tin chính xác

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8 trở lên
- Groq API Key (lấy tại [https://console.groq.com/](https://console.groq.com/))

### Các bước cài đặt

1. **Clone repository**

```bash
git clone https://github.com/trannam1084/RAG-Chatbot.git
cd "RAG-Chatbot"
```

2. **Tạo virtual environment (khuyến nghị)**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt dependencies**

```bash
pip install -r requirements.txt
```

4. **Tạo file .env**
   Tạo file `.env` trong thư mục gốc với nội dung:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. **Chạy ứng dụng**

```bash
streamlit run chatbot_app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

## 📖 Hướng dẫn sử dụng

1. **Upload PDF**: Chọn file PDF từ sidebar và nhấn "Xử lý PDF"
2. **Đặt câu hỏi**: Nhập câu hỏi về nội dung tài liệu vào ô chat
3. **Nhận trả lời**: Chatbot sẽ trả lời dựa trên thông tin trong PDF

## 🛠️ Công nghệ sử dụng

- **Streamlit**: Framework web app
- **LangChain**: LLM framework
- **Groq**: LLM API
- **Chroma**: Vector database
- **HuggingFace**: Embedding models

## 📄 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

## 👥 Tác giả

- Trần Hải Nam

---
