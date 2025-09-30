import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

# API Keys
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
#groq_api_key = os.getenv("GROQ_API_KEY")
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
#os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "QnA ChatBot For RBI"

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM
llm = ChatGroq(
    groq_api_key = groq_api_key,
    model = "llama-3.1-8b-instant"
)

# Streamlit UI 
st.set_page_config(page_title="RBI Q&A ChatBot",page_icon="💸", layout="wide")
# Chat History for Stateful management
username = st.sidebar.text_input("Enter your name").replace(" ", "")
if not username:
     username = "default_user"
session_id = st.sidebar.text_input("Session ID", value= f"{username}_session")
if "store" not in st.session_state:
        st.session_state.store = {}

# Title
st.title("🗨️RBI Q&A ChatBot")
st.markdown("Ask questions related to Reserve Bank of India (Non-Banking Financial Company – Scale Based Regulation)")

# Loading the given dataset
loader = PyPDFLoader("RBI.pdf")
documents = loader.load()

# Splitting and Embeddings for documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size= 2500, chunk_overlap= 300)
splits = text_splitter.split_documents(documents= documents)
vectorStore = FAISS.from_documents(documents= splits, embedding= embeddings)
# Retriever with more results
retriever = vectorStore.as_retriever(search_kwargs={"k": 8})

# Contextualized Prompt
contextualize_que_system_prompt = (
    "You are a query reformulation assistant for an RBI chatbot. "
    "Your task is to take the most recent user query and rewrite it as a clear, "
    "standalone question in the official FAQ style of the Reserve Bank of India.\n\n"
    
    "Guidelines:\n"
    "- Always use a formal, regulatory tone, similar to RBI FAQs.\n"
    "- If the user query is vague or informal, rewrite it into a precise and self-contained "
    "RBI-style FAQ question.\n"
    "- Explicitly include RBI context terms such as 'NBFCs', 'public deposits', "
    "'Certificate of Registration','guidelines' if they are implied but missing.\n"
    "- Ensure the reformulated question stands alone without needing prior chat history.\n"
    "- Do not answer the question. Only output the reformulated FAQ-style question.\n\n"
    
    "Output Example:\n"
    "User asks: 'can nbfc take deposits?'\n"
    "You reformulate as: 'Can all NBFCs accept deposits? What conditions are prescribed by the Reserve Bank of India for acceptance of public deposits?'"
)

contextualize_que_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_que_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

# History aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_que_prompt
)

# QnA Prompt
system_prompt = (
    "You are an AI assistant specializing in Reserve Bank of India (RBI) regulations, "
    "FAQs, circulars, and guidelines. Answer user questions using the retrieved RBI context "
    "as the primary source.\n\n"

    "Answering Style:\n"
    "- Respond in the official FAQ tone of the RBI.\n"
    "- Use short bullet points where helpful.\n"
    "- Include references to specific provisions, "
    "credit rating requirements, deposit limits, or timelines, when available.\n"
    "- Keep responses formal, regulatory, and precise.\n\n"

    "Rules:\n"
    "1. Always try to answer using the retrieved RBI context first.\n"
    "2. If the retrieved context does not contain the information, then use your broader internet knowledge "
    "to provide an authoritative RBI-style answer.\n"
    "3. If neither the context nor broader knowledge provides the answer, reply exactly: "
    "'I could not find this information in the RBI documents or online sources.'\n"
    "4. Never copy text verbatim; paraphrase in clear RBI language.\n"
    "5. Do not mention 'retrieved context' or 'documents'.\n"
    "6. Always keep your response concise, do not give too large response.\n"
    "7. Never include filler phrases such as 'based on my knowledge'—keep the official FAQ tone.\n\n"

    "Retrieved RBI Context:\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

# Create Chain
qna_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)


# Session History fn
def get_session_history(session:str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Create Conversational RAG Chain
conversational_rag_chain=RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key= "input",
    history_messages_key= "chat_history",
    output_messages_key= "answer"
    )

# User Input
if user_input := st.chat_input("Ask your query:"):
    session_history = get_session_history(session_id)
    res = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id":session_id}
        }
    )
    for msg in session_history.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)

               
# User Input without chat history
#if user_input := st.text_input("Ask your query:"):
#    res = rag_chain.invoke({"input": user_input})

#    st.markdown(f"**Answer:** {res['answer']}")

