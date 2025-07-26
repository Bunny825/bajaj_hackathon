import os
import requests
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

import cassio

load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID
)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="bajaj_insurance",
    session=None,
    keyspace=ASTRA_DB_KEYSPACE
)

last_url = None


def insurance_answer(url, queries):
    global last_url

    if url != last_url:
        astra_vector_store.clear() 
        last_url = url

        file_path = f"/tmp/{uuid4()}.pdf"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)

            loader = UnstructuredLoader(file_path)
            docs = loader.load()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000,
            chunk_overlap=200
        )
        final_docs = splitter.split_documents(docs)

        for i in range(0, len(final_docs), 200):
            astra_vector_store.add_documents(final_docs[i:i+200])

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    retriever = astra_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}, 
        timeout=20
    )

    prompt = ChatPromptTemplate.from_template(
        "You are an expert assistant that answers questions about insurance policies with precise, fact-based information. "
        "Only answer based on the given context. Do not make up information. "
        "Always include exact time periods, percentages, exclusions, and conditions as mentioned in the policy. "
        "Keep the answer short, formal, and directly addressing the user's question.\n\n"
        "Context: {context}\n\n"
        "Question: {input}"
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    final_chain = create_retrieval_chain(retriever, doc_chain)

    answers = []
    for query in queries:
        try:
            result = final_chain.invoke({"input": query})

            # Most likely structure returned
            if isinstance(result, dict) and "answer" in result:
                answers.append(result["answer"])
            else:
                answers.append(str(result))

        except Exception as e:
            answers.append(f"An unexpected error occurred: {str(e)}")


    return answers
