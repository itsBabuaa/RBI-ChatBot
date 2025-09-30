# RBI Q&A ChatBot

A Streamlit-based chatbot that answers questions related to the Reserve Bank of India (RBI) regulations, guidelines, and FAQs. It uses LangChain, Groq LLM, and HuggingFace embeddings to provide context-aware responses from RBI documents.

---

## Features

- Ask questions related to RBI regulations and guidelines.
- Contextualized question reformulation for precise FAQ-style answers.
- Uses PDF documents as the source of truth.
- Maintains session-based chat history.
- Formal RBI-style response tone with bullet points and references.
- Built with Streamlit for an interactive web interface.

---

## Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Large Language Model**: [Groq ChatGroq](https://groq.com/)
- **Embeddings**: [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **RAG (Retrieval-Augmented Generation)**: LangChain
- **PDF Loader & Text Splitter**: LangChain Community
- **Environment Management**: Python `dotenv` for API keys
