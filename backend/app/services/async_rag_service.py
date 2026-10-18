from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from app.core.config import settings


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.gemini_api_key,
)


async def run_rag(question: str):

    # 1. Load PDF
    loader = PyMuPDF4LLMLoader("uploads/resume.pdf")
    docs = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    texts = text_splitter.split_documents(docs)

    # 3. Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=settings.gemini_api_key,
    )

    # 4. Create vector store
    vectorstore = Chroma(
        collection_name="resume",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    # 5. Add chunks
    vectorstore.add_documents(texts)

    # 6. Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # 7. Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    # 8. Prompt
    prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}
""")

    # 9. Build RAG chain
    rag_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # 10. Stream the answer
    async for chunk in rag_chain.astream(question):
        yield chunk