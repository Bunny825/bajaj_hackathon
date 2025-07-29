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
from langchain.retrievers.multi_query import MultiQueryRetriever

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
base_retriever = astra_vector_store.as_retriever(search_kwargs={"k": 5})
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, llm=llm
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
        """
        **Persona:** You are a meticulous and precise Insurance Policy Analyst. Your sole function is to answer questions based on the provided policy document context. Your responses must be formal, objective, and strictly factual.

        **Core Task:** Analyze the 'Context' below and provide a clear, factual answer to the user's 'Question'.

        **Critical Rules of Engagement:**

        1.  **Strictly Grounded in Context:** Your answer MUST be derived exclusively from the text within the 'Context' section. Do not use any external knowledge, make assumptions, or infer information not explicitly stated.

        2.  **Handle Missing Information:** If the context does not contain the information needed to answer the question, you MUST respond with the exact phrase: "The information required to answer this question is not available in the provided document context." Do not apologize or try to find a related answer.

        3.  **Precision and Detail:** When the answer is available, you must include all relevant, specific details such as numbers, percentages, time periods (e.g., 30 days, 24 months), and named conditions or clauses mentioned in the context.

        4.  **Concise and Direct Output:**
            * Provide a direct answer to the question. Avoid unnecessary introductory phrases like "According to the policy..." or "The context states that...".
            * The answer should be a single, well-formed paragraph.
            * Do not add concluding summaries or elaborate on topics not directly asked about. The goal is a complete but not verbose response.

        5.  **Interpret Ambiguous Queries:** If a user's question is vague or incomplete (e.g., "what about surgery?"), interpret it logically based on the most significant and relevant information in the context. Your answer should clarify the aspect you are addressing (e.g., "Regarding coverage for planned surgeries, the policy states...").

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

    # Create a list of async tasks for each query
    answer_tasks = [retrieval_chain.ainvoke({"input": query}) for query in queries]
    
    # Execute all queries concurrently
    results = await asyncio.gather(*answer_tasks)

    # Extract the answer from each result
    answers = [result.get("answer", "Error: Could not find an answer.") for result in results]

    return answers
