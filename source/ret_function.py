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
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

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

# --- SETUP THE ADVANCED RETRIEVER WITH RE-RANKING ---
# 1. The base retriever fetches a larger number of initial documents (k=20).
#    This "casts a wider net" to increase the chances of finding the right context.
base_retriever = astra_vector_store.as_retriever(search_kwargs={"k": 20})

# 2. The CohereRerank compressor takes these 20 documents and intelligently finds the top 3.
compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")

# 3. The final retriever combines these two steps into a single component.
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# A simple in-memory cache to track processed URLs
processed_urls = set()

# --- Core Asynchronous Function ---
async def insurance_answer(url: str, queries: list[str]) -> list[str]:
    """
    Asynchronously processes a document and answers questions about it with
    a robust re-ranking retriever and controlled concurrency.
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
        **Persona:** You are a meticulous and precise Insurance Policy Analyst. Your sole function is to answer questions based on the provided policy document context. Your responses must be formal, objective, and strictly factual.

        **Core Task:** Analyze the 'Context' below and provide a clear, factual answer to the user's 'Question'.

        **Critical Rules of Engagement:**
        1.  **Strictly Grounded in Context:** Your answer MUST be derived exclusively from the text within the 'Context' section. Do not use any external knowledge or make assumptions not explicitly stated.

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
    
    # CONTROLLED CONCURRENCY (To handle ALL rate limits)
    final_answers = []
    # Process in small batches to avoid overwhelming API rate limits.
    batch_size = 5
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_tasks = [retrieval_chain.ainvoke({"input": query}) for query in batch_queries]
        results = await asyncio.gather(*batch_tasks)
        answers = [result.get("answer", "Error: Could not find an answer.") for result in results]
        final_answers.extend(answers)

        # Add a small delay between batches to respect time-based rate limits
        if i + batch_size < len(queries):
            print(f"Batch complete. Waiting for 2 seconds to respect rate limits...")
            await asyncio.sleep(2)
        
    return final_answers
