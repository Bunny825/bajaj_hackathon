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
from langchain.document_loaders import UnstructuredFileLoader # Using UnstructuredFileLoader for in-memory processing
from langchain_community.vectorstores import Cassandra

import cassio

# --- Initialization (runs only once on startup) ---
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize components that don't change per request
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo") # Using a faster model can also help
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="bajaj_insurance_policy_prod", # Use a more descriptive table name
    session=None,
    keyspace=ASTRA_DB_KEYSPACE,
)
retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})

# A simple in-memory cache to track processed URLs
processed_urls = set()

# --- Core Asynchronous Function ---
async def insurance_answer(url: str, queries: list[str]) -> list[str]:
    """
    Asynchronously processes a document and answers questions about it.
    """
    global processed_urls

    # 1. DATA INGESTION (only if document is new)
    if url not in processed_urls:
        print(f"New document URL received: {url}. Processing...")
        # For this specific use case, we clear the store for each new document.
        # In a real-world app, you'd manage different documents differently.
        await astra_vector_store.aclear()

        # Asynchronously download the PDF content
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            pdf_content = response.content

        # Process the document in memory
        file_path = f"/tmp/{uuid4()}.pdf"
        with open(file_path, "wb") as f:
            f.write(pdf_content)
        
        loader = UnstructuredFileLoader(file_path)
        docs = await loader.aload() # Async loading
        os.remove(file_path) # Clean up the temp file

        # Split documents
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)

        # Asynchronously add documents to the vector store in batches
        batch_size = 20
        tasks = []
        for i in range(0, len(final_docs), batch_size):
            batch = final_docs[i:i + batch_size]
            tasks.append(astra_vector_store.aadd_documents(batch))
        
        await asyncio.gather(*tasks)
        processed_urls.add(url)
        print("Document processing and ingestion complete.")

    # 2. QUESTION ANSWERING (run concurrently)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert assistant that answers questions about insurance policies with precise, fact-based information. "
        "Only answer based on the given context. Do not make up information. "
        "Always include exact time periods, percentages, exclusions, and conditions as mentioned in the policy. "
        "Keep the answer short, formal, and directly addressing the user's question.\n\n"
        "Context: {context}\n\n"
        "Question: {input}"
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    # Create a list of async tasks for each query
    answer_tasks = [retrieval_chain.ainvoke({"input": query}) for query in queries]
    
    # Execute all queries concurrently
    results = await asyncio.gather(*answer_tasks)

    # Extract the answer from each result
    answers = [result.get("answer", "Error: Could not find an answer.") for result in results]

    return answers