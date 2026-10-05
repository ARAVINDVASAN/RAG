from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

loader = PyMuPDF4LLMLoader("uploads/resume.pdf")

docs = loader.load()
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

texts = text_splitter.split_documents(docs)
from langchain_chroma import Chroma


from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",google_api_key=settings.gemini_api_key
)

vectorstore = Chroma(
    collection_name="resume",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.add_documents(texts)

results = vectorstore.similarity_search(
    "What are my skills?",
    k=3
)
for doc in results:
    print(doc.page_content,"resultt")
