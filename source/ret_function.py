import os
import io
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
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Cassandra
from cassandra.cluster import ConsistencyLevel
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import cassio

load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
COHERE_API_KEY = os.getenv("COHERE_API_KEY") # Your Production Key

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
session = cassio.config.resolve_session()


embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="bajaj_insurance_policy_prod",
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
)

async def insurance_answer(url: str, queries: list[str]) -> list[str]:

    file_path_main = url.lower().split('?')[0]
    
    if file_path_main.endswith('.xlsx'):
        print(f"XLSX file detected: {url}. Using Data Analysis Engine...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            df = pd.read_excel(io.BytesIO(response.content))


        prefix_prompt = "The data is based on India, so all the results must be based on India."
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            agent_type="openai-tools", 
            verbose=True, 
            allow_dangerous_code=True,
            prefix=prefix_prompt
        )
        
        tasks = [agent.ainvoke({"input": query}) for query in queries]
        results = await asyncio.gather(*tasks)
        
        answers = [result.get("output", "Error: Could not process the query for the spreadsheet.") for result in results]
        return answers

    elif file_path_main.endswith('.pptx'):
        print(f"PPTX file detected: {url}. Using Presenter Engine (Slide-by-Slide RAG)...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            content = response.content
        file_path = f"/tmp/{uuid4()}.pptx"
        with open(file_path, "wb") as f: f.write(content)
        # Using strategy="hi_res" to get one document per slide
        loader = UnstructuredFileLoader(file_path, strategy="hi_res")
        final_docs = await loader.aload()
        os.remove(file_path)

    else:
        print(f"Text document detected: {url}. Using Librarian Engine (Standard RAG)...")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            content = response.content
        file_path = f"/tmp/{uuid4()}.tmp"
        with open(file_path, "wb") as f: f.write(content)
        loader = UnstructuredFileLoader(file_path)
        docs = await loader.aload()
        os.remove(file_path)
        
        if not docs:
            raise ValueError("Failed to load or parse the document content.")

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)

    await astra_vector_store.aclear()
    await astra_vector_store.aadd_documents(final_docs)
    
    retriever_kwargs = {
    "search_kwargs": {"k": 12},
    "consistency_level": ConsistencyLevel.LOCAL_ONE,
    }
    base_retriever = astra_vector_store.as_retriever(**retriever_kwargs)

    compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    
    print("Fresh RAG retriever has been built.")

    qa_prompt = ChatPromptTemplate.from_template(
        """
        **Persona:** You are a diligent and precise Research Analyst. Your sole function is to answer questions based on the provided document context. Your responses must be formal, objective, and strictly factual.

        **Core Task:** Analyze the 'Context' below and provide a clear, factual answer to the user's 'Question'.

        **Critical Rules of Engagement:**
        1.  **Strictly Grounded in Context:** Your answer MUST be derived exclusively from the text within the 'Context' section. Do not use any external knowledge, make assumptions, not explicitly stated.
        2.  **Handle All Data Formats:** The provided 'Context' can be prose from a book, legal text, technical specifications, or structured data from a spreadsheet that has been converted to text. Your task is to interpret the provided format literally and extract the answer.
        3.  **Best-Effort Answering:** If the context does not contain a perfect, direct answer, you must still attempt to provide the most relevant information available. If no relevant information exists at all, then you may state that the information could not be found.
        4.  **Precision and Detail:** When the answer is available, you must include all relevant, specific details: direct quotes from text, specific numbers from tables, or exact clauses from legal documents.
        5.  **Concise and Direct Output:** Provide a direct answer to the question. Avoid unnecessary introductory phrases. The answer should be a single, well-formed paragraph.

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
