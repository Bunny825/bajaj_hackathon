import os
import asyncio
import httpx 
import pandas as pd
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
# --- Imports for the XLSX Data Analyst ---
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

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
llm2 = ChatOpenAI(model="gpt-4o", temperature=0)


astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="bajaj_insurance_policy_prod",
    session=None,
    keyspace=ASTRA_DB_KEYSPACE,
)

# --- Core Asynchronous Function ---
async def insurance_answer(url: str, queries: list[str]) -> list[str]:
    """
    Asynchronously processes a document and answers questions by routing
    to the correct engine (Text RAG or Data Analysis Agent) based on file type.
    """
    
    print(f"Received queries: {queries}")

    # --- SMART ROUTER: Check if the file is an XLSX spreadsheet ---
    # CRITICAL FIX: Check for the extension before any query parameters
    if '.xlsx' in url.lower().split('?')[0]:
        print(f"XLSX file detected: {url}. Using Data Analysis Engine...")
        
        # 1. Download the XLSX file
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            excel_content = response.content

        # 2. Load the spreadsheet into a Pandas DataFrame
        # We use a temporary file to load it with pandas
        file_path = f"/tmp/{uuid4()}.xlsx"
        with open(file_path, "wb") as f:
            f.write(excel_content)
        
        df = pd.read_excel(file_path)
        os.remove(file_path)
        
        # 3. Create the Pandas DataFrame Agent
        # This agent can write and execute Python code to answer questions about the data
        agent = create_pandas_dataframe_agent(
            llm2, 
            df, 
            agent_type="openai-tools", 
            verbose=True, 
            allow_dangerous_code=True
        )
        
        # 4. Answer all questions concurrently
        tasks = [agent.ainvoke({"input": query}) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # Extract the text output from the agent's response
        answers = [result.get("output", "Error: Could not process the query for the spreadsheet.") for result in results]
        return answers

    # --- DEFAULT BRANCH: Use the Text RAG Engine for all other document types ---
    else:
        print(f"Text document detected: {url}. Using RAG Engine...")
        # This is your proven, non-cached re-ranker pipeline for text
        
        await astra_vector_store.aclear()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            pdf_content = response.content
        file_path = f"/tmp/{uuid4()}.tmp" # Use a generic extension
        with open(file_path, "wb") as f: f.write(pdf_content)
        loader = UnstructuredFileLoader(file_path)
        docs = await loader.aload()
        os.remove(file_path)
        
        if not docs:
            raise ValueError("Failed to load or parse the document content.")

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)
        
        await astra_vector_store.aadd_documents(final_docs)
        
        base_retriever = astra_vector_store.as_retriever(search_kwargs={"k": 10})
        compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        
        qa_prompt = ChatPromptTemplate.from_template(
            """
            **Persona:** You are a diligent and precise Research Analyst. Your sole function is to answer questions based on the provided document context. Your responses must be formal, objective, and strictly factual.

            **Core Task:** Analyze the 'Context' below and provide a clear, factual answer to the user's 'Question'.

            **Critical Rules of Engagement:**

            1.  **Strictly Grounded in Context:** Your answer MUST be derived exclusively from the text within the 'Context' section. Do not use any external knowledge, make assumptions, or infer information not explicitly stated.

            2.  **Handle All Data Formats:** The provided 'Context' can be prose from a book, legal text from a constitution, technical specifications from a manual, or structured data from a spreadsheet (e.g., rows and columns). Your task is to interpret the provided format literally and extract the answer. For tabular data, answer based on the specific rows and columns provided.

            3.  **Best-Effort Answering:** If the context does not contain a perfect, direct answer, you must still attempt to provide the most relevant information available. If you are providing an answer that is related but not a direct answer, you can state that. If no relevant information exists at all, then you may state that the information could not be found.

            4.  **Precision and Detail:** When the answer is available, you must include all relevant, specific details: direct quotes from text, specific numbers from tables, or exact clauses from legal documents.

            5.  **Concise and Direct Output:** Provide a direct answer to the question. Avoid unnecessary introductory phrases. The answer should be a single, well-formed paragraph. Do not add concluding summaries or elaborate on topics not directly asked about.

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
        
        tasks = [retrieval_chain.ainvoke({"input": query}) for query in queries]
        results = await asyncio.gather(*tasks)
        answers = [result.get("answer", "Error: Could not find an answer.") for result in results]
        return answers
