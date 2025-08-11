import streamlit as st
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nomic import NomicEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
llm_model = "gemini-2.5-pro"  
embedding_model_name = "nomic-embed-text-v1.5"
db_path = "./lancedb"

# --- Initialize LLM and Embedder ---
llm = ChatGoogleGenerativeAI(model=llm_model)
embedder = NomicEmbeddings(model=embedding_model_name)

# --- Connect to Vector Store ---
db = lancedb.connect(db_path)
vectorstore = LanceDB(
    connection=db,
    table_name="vectorstore",
    embedding=embedder
)
retriever = vectorstore.as_retriever()

# --- Build RAG Chain ---
rag_template = """YOUR NAME IS Viswamithra. Answer the question based only on the following context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- Streamlit UI ---
st.title("Viswamithra")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about saptharushi:")

if question:
    with st.spinner("Getting answer..."):
        answer = rag_chain.invoke(question)
    st.session_state.history.append((question, answer))

if st.session_state.history:
    st.markdown("### Conversation History")
    for q, a in st.session_state.history:
        st.markdown(f"**Question:** {q}")
        st.markdown(f"**Answer:** {a}")

# User can ask next question after seeing the answer