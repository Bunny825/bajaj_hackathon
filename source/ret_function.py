import os
import asyncio
import httpx # Using httpx for async requests
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

import cassio

# --- Initialization (runs only once on startup) ---
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
# COHERE_API_KEY is no longer needed

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

# --- SETUP A ROBUST, SIMPLE RETRIEVER ---
# CRITICAL FIX: Reduced k from 15 to 8 to prevent exceeding the context window.
# This is the sweet spot between providing enough context and staying within the model's limit.
retriever = astra_vector_store.as_retriever(search_kwargs={"k": 8})


# A simple in-memory cache to track processed URLs
processed_urls = set()

# --- Core Asynchronous Function ---
async def insurance_answer(url: str, queries: list[str]) -> list[str]:
    """
    Asynchronously processes a document and answers questions about it with
    a robust retriever and maximum concurrency.
    """
    global processed_urls

    # DATA INGESTION (only if document is new)
    if url not in processed_urls:
        print(f"New document URL received: {url}. Processing...")
        await astra_vector_store.aclear()
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
            print("ERROR: UnstructuredFileLoader returned no documents.")
            raise ValueError("Failed to load or parse the document content.")

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)
        
        batch_size = 20
        tasks = [astra_vector_store.aadd_documents(final_docs[i:i + batch_size]) for i in range(0, len(final_docs), batch_size)]
        await asyncio.gather(*tasks)
        processed_urls.add(url)
        print("Document processing and ingestion complete.")

    # QUESTION ANSWERING PIPELINE
    qa_prompt = ChatPromptTemplate.from_template(
        """
        **Persona:** You are a meticulous and precise Insurance Policy Analyst...
        (Your full, detailed prompt goes here)
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
    
    # Process all queries at maximum speed
    tasks = [retrieval_chain.ainvoke({"input": query}) for query in queries]
    results = await asyncio.gather(*tasks)
    answers = [result.get("answer", "Error: Could not find an answer.") for result in results]
        
    return answers
