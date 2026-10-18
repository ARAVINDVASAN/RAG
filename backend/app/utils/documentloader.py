from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.gemini_api_key
)

loader = PyMuPDF4LLMLoader("uploads/resume.pdf")

docs = loader.load()
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

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
#similarity search
# similarity_search_results = vectorstore.similarity_search(
#     "What are my skills?",
#     k=3
# )
# print(similarity_search_results,"similarity_Search")


###########using retriver
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


results = retriever.invoke("What are my skills?")


prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
""")
# context = "\n\n".join(
#     doc.page_content for doc in results
# )


# prompt_value = prompt.invoke({
#     "context": context,
#     "question": "What are my skills?"
# })

# response = llm.invoke(prompt_value)
# parser = StrOutputParser()
# parser_result=parser.invoke(response)
# print(response.content,"resssponeee",parser_result)

# for doc in results:
#     print(doc.page_content,"resultt")

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

format_docs_runnable = RunnableLambda(format_docs)
rag_chain = (
    RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

# result = rag_chain.invoke("What are my skills?")


#using streaming
result = rag_chain.stream("Where did he study?")
for chunks in result:
    print(chunks,end=" ",flush=True)

print(result)

