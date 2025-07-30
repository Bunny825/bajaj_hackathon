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
base_retriever = astra_vector_store.as_retriever(search_kwargs={"k": 10})
compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# A simple in-memory cache to track processed URLs
processed_urls = set()

# --- 1. QUERY TRANSFORMATION (To improve accuracy) ---
# This prompt asks the LLM to rewrite the user's question into a more
# precise query that is more likely to match the document's language.
query_transform_prompt = ChatPromptTemplate.from_template(
    "You are an expert at rewriting user questions into precise search queries for a vector database. "
    "Based on the user's question about an insurance policy, generate a new query that is optimized for semantic search. "
    "Focus on using keywords and phrases that would likely appear in a formal policy document, such as 'coverage limitations', 'exclusion clauses', 'waiting periods', 'co-payment responsibilities', etc. "
    "User Question: {question}"
)
query_transformer = query_transform_prompt | llm

# --- Core Asynchronous Function ---
async def insurance_answer(url: str, queries: list[str]) -> list[str]:
    """
    Asynchronously processes a document and answers questions about it with
    query transformation and controlled concurrency.
    """
    global processed_urls

    # DATA INGESTION (only if document is new)
    if url not in processed_urls:
        print(f"New document URL received: {url}. Processing...")
        await astra_vector_store.aclear()
        # ... (rest of your data ingestion code is perfect, no changes needed)
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            pdf_content = response.content
        file_path = f"/tmp/{uuid4()}.pdf"
        with open(file_path, "wb") as f: f.write(pdf_content)
        loader = UnstructuredFileLoader(file_path)
        docs = await loader.aload()
        os.remove(file_path)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)
        batch_size = 20
        tasks = [astra_vector_store.aadd_documents(final_docs[i:i + batch_size]) for i in range(0, len(final_docs), batch_size)]
        await asyncio.gather(*tasks)
        processed_urls.add(url)
        print("Document processing and ingestion complete.")

    # 2. QUESTION ANSWERING PIPELINE
    qa_prompt = ChatPromptTemplate.from_template(
        """
        **Persona:** You are a meticulous and precise Insurance Policy Analyst. Your sole function is to answer questions based on the provided policy document context. Your responses must be formal, objective, and strictly factual.

        **Core Task:** Analyze the 'Context' below and provide a clear, factual answer to the user's 'Question'.

        **Critical Rules of Engagement:**
        1. **Strictly Grounded in Context:** Your answer MUST be derived exclusively from the text within the 'Context' section. Do not use any external knowledge, make assumptions, or infer information not explicitly stated.
        2. **Handle Missing Information:** If the context does not contain the information needed to answer the question, you MUST respond with the exact phrase: "The information required to answer this question is not available in the provided document context." Do not apologize or try to find a related answer.
        3. **Precision and Detail:** When the answer is available, you must include all relevant, specific details such as numbers, percentages, time periods (e.g., 30 days, 24 months), and named conditions or clauses mentioned in the context.
        4. **Concise and Direct Output:** Provide a direct answer to the question. Avoid unnecessary introductory phrases. The answer should be a single, well-formed paragraph. Do not add concluding summaries or elaborate on topics not directly asked about.
        5. **Interpret Ambiguous Queries:** If a user's question is vague or incomplete, interpret it logically based on the most significant and relevant information in the context. Your answer should clarify the aspect you are addressing.

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
    
    # --- 3. CONTROLLED CONCURRENCY (To handle rate limits) ---
    # We process the queries in small batches to avoid hitting the API rate limit.
    final_answers = []
    batch_size = 5 # Process 5 queries at a time (safely below the 10/min limit)
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        
        # Create a list of tasks for the current batch
        batch_tasks = []
        for query in batch_queries:
            # First, transform the query
            transformed_query_result = await query_transformer.ainvoke({"question": query})
            transformed_query = transformed_query_result.content
            print(f"Original Query: '{query}' -> Transformed Query: '{transformed_query}'")
            
            # Then, create the task for the full QA chain with the transformed query
            task = retrieval_chain.ainvoke({"input": transformed_query})
            batch_tasks.append(task)
        
        # Execute the current batch of tasks concurrently
        results = await asyncio.gather(*batch_tasks)
        
        # Extract and store the answers
        answers = [result.get("answer", "Error: Could not find an answer.") for result in results]
        final_answers.extend(answers)
        
        # Optional: Add a small delay between batches if needed, though batching is usually sufficient
        # await asyncio.sleep(1) 

    return final_answers
