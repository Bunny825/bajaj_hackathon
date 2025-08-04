import os
import asyncio
import httpx 
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Cassandra
# --- Imports for the Re-ranker ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

import cassio

# --- Initialization (runs only once on startup) ---
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
COHERE_API_KEY = os.getenv("COHERE_API_KEY") # Your new Production Key

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize components that don't change per request
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="bajaj_insurance_policy_prod",
    session=None,
    keyspace=ASTRA_DB_KEYSPACE,
)

# --- Core Asynchronous Function ---
async def insurance_answer(url: str, queries: list[str]) -> list[str]:
    """
    Asynchronously processes a document and answers questions with a fresh,
    non-cached re-ranker retriever to guarantee data isolation and accuracy.
    """
    # --- DATA INGESTION AND RETRIEVER SETUP ON EVERY CALL ---
    # This is the only way to be 100% sure there is no data leakage.
    print(f"Processing document: {url}. Building fresh retriever...")
    
    # 1. Clear the remote vector store to ensure no data leakage from previous runs
    await astra_vector_store.aclear()
    
    # 2. Download and process the new document
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()
        pdf_content = response.content
    file_path = f"/tmp/{uuid4()}.pdf"
    with open(file_path, "wb") as f: f.write(pdf_content)
    loader = UnstructuredFileLoader(file_path)
    docs = await loader.aload()
    os.remove(file_path)
    
    if not docs:
        raise ValueError("Failed to load or parse the document content.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)
    
    # 3. Add the new documents to the now-empty vector store
    await astra_vector_store.aadd_documents(final_docs)
    
    # --- BUILD THE RETRIEVER FROM SCRATCH ---
    base_retriever = astra_vector_store.as_retriever(search_kwargs={"k": 10})
    compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    print("Fresh retriever has been built.")

    # QUESTION ANSWERING PIPELINE
    qa_prompt = ChatPromptTemplate.from_template(
        """
        **Persona:** You are a meticulous and precise Insurance Policy Analyst. Your sole function is to answer questions based on the provided policy document context. Your responses must be formal, objective, and strictly factual.

        **Core Task:** Analyze the 'Context' below and provide a clear, factual answer to the user's 'Question'.

        **Critical Rules of Engagement:**
        1.  **Strictly Grounded in Context:** Your answer MUST be derived exclusively from the text within the 'Context' section. Do not use any external knowledge, make assumptions, not explicitly stated.

        2.  **Best-Effort Answering:** If the context does not contain a perfect, direct answer, you must still attempt to provide the most relevant information available. If you are providing an answer that is related but not a direct answer, you can state that. If no relevant information exists at all, then you may state that the information could not be found.

        3.  **Precision and Detail:** When the answer is available, you must include all relevant, specific details such as numbers, percentages, time periods (e.g., 30 days, 24 months), and named conditions or clauses mentioned in the context.

        4.  **Concise and Direct Output:** Provide a direct answer to the question. Avoid unnecessary introductory phrases. The answer should be a single, well-formed paragraph. Do not add concluding summaries or elaborate on topics not directly asked about.

        ---
        **Context:**
        {context}
        ---
        **Question:**
        {input}
        ---
        **Answer:**
        """
    )
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    
    # --- UNLEASH MAXIMUM SPEED (NO BATCHING) ---
    # With a production key, we can process all questions at once.
    tasks = [retrieval_chain.ainvoke({"input": query}) for query in queries]
    results = await asyncio.gather(*tasks)
    answers = [result.get("answer", "Error: Could not find an answer.") for result in results]
        
    return answers
