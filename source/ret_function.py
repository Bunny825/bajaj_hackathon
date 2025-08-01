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
# --- Imports for Hybrid Search ---
from langchain.retrievers import BM25Retriever, EnsembleRetriever

import cassio

# --- Initialization (runs only once on startup) ---
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize components that don't change per request
embeddings = OpenAIEmbeddings()
# --- Reverted to OpenAI as requested ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="bajaj_insurance_policy_prod",
    session=None,
    keyspace=ASTRA_DB_KEYSPACE,
)

# A simple in-memory cache to store the fully initialized retriever for each document
retriever_cache = {}

# --- Core Asynchronous Function ---
async def insurance_answer(url: str, queries: list[str]) -> list[str]:
    """
    Asynchronously processes a document and answers questions about it with
    a robust, cached hybrid search retriever to ensure speed and accuracy.
    """
    global retriever_cache

    # --- CRITICAL FIX: Check if we have already built a retriever for this URL ---
    if url not in retriever_cache:
        print(f"New document URL received: {url}. Building retriever for the first time...")
        # 1. Clear the remote vector store to ensure no data leakage
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
        
        # --- BUILD THE HYBRID RETRIEVER ONCE ---
        # 4. Create the keyword retriever from the new document's chunks
        bm25_retriever = BM25Retriever.from_documents(final_docs)
        bm25_retriever.k = 4
        
        # 5. Create the vector retriever (which now correctly points to the new data)
        vector_retriever = astra_vector_store.as_retriever(search_kwargs={"k": 4})
        
        # 6. Combine them into the final ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
        )
        
        # 7. Store the fully-built retriever in the cache
        retriever_cache[url] = ensemble_retriever
        print("Retriever for this document has been built and cached.")

    # Retrieve the ready-to-use, fast retriever from the cache
    retriever = retriever_cache[url]

    # QUESTION ANSWERING PIPELINE
    qa_prompt = ChatPromptTemplate.from_template(
        """
        **Persona:** You are a meticulous and precise Insurance Policy Analyst. Your sole function is to answer questions based on the provided policy document context. Your responses must be formal, objective, and strictly factual.

        **Core Task:** Analyze the 'Context' below and provide a clear, factual answer to the user's 'Question'.

        **Critical Rules of Engagement:**
        1.  **Strictly Grounded in Context:** Your answer MUST be derived exclusively from the text within the 'Context' section. Do not use any external knowledge, make assumptions, or infer information not explicitly stated.

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
    
    # --- CONTROLLED CONCURRENCY (To handle OpenAI's rate limits) ---
    final_answers = []
    # Process in small batches to avoid overwhelming the OpenAI TPM limit.
    batch_size = 3
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_tasks = [retrieval_chain.ainvoke({"input": query}) for query in batch_queries]
        results = await asyncio.gather(*batch_tasks)
        answers = [result.get("answer", "Error: Could not find an answer.") for result in results]
        final_answers.extend(answers)

        # Add a small delay between batches to respect time-based rate limits
        if i + batch_size < len(queries):
            print(f"Batch complete. Waiting for 1 second to respect rate limit...")
            await asyncio.sleep(1)
        
    return final_answers
