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
# CORRECTED IMPORT: CohereRerank is now in the langchain_cohere package
from langchain_cohere import CohereRerank

import cassio

# --- Initialization (runs only once on startup) ---
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
# IMPORTANT: You will need a Cohere API key for the re-ranker
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
# 1. The base retriever fetches a larger number of initial documents (e.g., 10).
#    This increases the chance of finding the correct information.
base_retriever = astra_vector_store.as_retriever(search_kwargs={"k": 10})

# 2. The CohereRerank compressor takes these documents and intelligently re-orders them
#    based on relevance to the query. It will return the top 3.
compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3)

# 3. The final retriever combines these two steps into a single component.
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

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
        await astra_vector_store.aclear()

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            pdf_content = response.content

        file_path = f"/tmp/{uuid4()}.pdf"
        with open(file_path, "wb") as f:
            f.write(pdf_content)
        
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

    # 2. QUESTION ANSWERING (run concurrently)
    prompt = ChatPromptTemplate.from_template(
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

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    answer_tasks = [retrieval_chain.ainvoke({"input": query}) for query in queries]
    results = await asyncio.gather(*answer_tasks)
    answers = [result.get("answer", "Error: Could not find an answer.") for result in results]

    return answers
