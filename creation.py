import os
import lancedb
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import LanceDB
from langchain_nomic import NomicEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai

# --- Configuration ---
pdf_file_path = "saptharushi.pdf"
embedding_model_name = "nomic-embed-text-v1.5"
db_path = "./lancedb"

# --- Check for PDF ---
if not os.path.exists(pdf_file_path):
    print(f"Error: The file '{pdf_file_path}' was not found.")
    exit()

# --- Check for Gemini API Key ---
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit()

# --- 1. Load and Chunk the Document ---
print(f"Loading and chunking document: {pdf_file_path}")
loader = UnstructuredPDFLoader(file_path=pdf_file_path)
doc_splits = loader.load()
print(f"Document split into {len(doc_splits)} chunks.")

# Optional: Print original chunks
for idx, chunk in enumerate(doc_splits, start=1):
    print(f"Original chunk{idx}: {chunk.page_content}")

# --- 2. Contextualize and Embed Chunks ---
def contextualize_and_embed_chunks(doc_splits, full_document, embedder):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    contextual_chunks = []
    for idx, chunk in enumerate(doc_splits, start=1):
        prompt = (
            f"Here is the full document:\n{full_document}\n\n"
            f"Here is chunk{idx}:\n{chunk.page_content}\n\n"
            "Please add relevant context from the document to this chunk, "
            "making it more informative and self-contained. Return only the improved chunk text."
        )
        response = model.generate_content(prompt)
        improved_chunk = response.text.strip()
        embedding = embedder.embed_query(improved_chunk)
        contextual_chunks.append((improved_chunk, embedding))
        print(f"Contextualized and embedded chunk{idx}: {improved_chunk}")
    return contextual_chunks

full_document = "\n".join([chunk.page_content for chunk in doc_splits])
print(f"Initializing Nomic Embedding Model via API: {embedding_model_name}")
embedder = NomicEmbeddings(model=embedding_model_name)

contextualized = contextualize_and_embed_chunks(doc_splits, full_document, embedder)
contextualized_documents = [Document(page_content=chunk) for chunk, _ in contextualized]

# Optional: Print contextualized chunks
for idx, (chunk, _) in enumerate(contextualized, start=1):
    print(f"Contextualized chunk{idx}: {chunk}")

print(f"Embedding contextualized chunks and storing them in LanceDB at: {db_path}")
vectorstore = LanceDB.from_documents(
    documents=contextualized_documents,
    embedding=embedder,
    connection=lancedb.connect(db_path)
)
print("Vector store has been created and saved successfully using contextualized chunks.")
