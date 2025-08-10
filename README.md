# Intelligent Document Query System (HackRx 6.0)

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)
![Container](https://img.shields.io/badge/Container-Docker-blue.svg)

An advanced, intelligent query-retrieval system designed to provide precise answers from complex documents like insurance policies, legal contracts, and financial reports. This project was developed as a submission for the **HackRx 6.0 Hackathon**.

## 📝 Project Overview

This solution is a multi-format query engine that dynamically selects the optimal processing strategy based on the input document's format. It features an **Adaptive Engine** that can pivot between a sophisticated Retrieval-Augmented Generation (RAG) pipeline for text-based documents (PDF, DOCX, PPTX) and a powerful data agent for structured spreadsheets (.xlsx). This dual-pathway approach ensures high accuracy and relevance across diverse data types, delivering structured JSON responses through a secure and scalable API.

---

## ✨ Key Features

* **Adaptive Dual-Pipeline Architecture**: Intelligently switches between a RAG pipeline and a Pandas DataFrame Agent based on the document type.
* **Multi-Format Document Support**: Natively processes PDFs, DOCX, PPTX, and XLSX files.
* **High-Precision Retrieval**: Implements a two-stage retrieval process using a standard vector search followed by a `Cohere Re-ranker` to significantly enhance context quality and answer accuracy.
* **Analytical Capabilities**: The Pandas Agent can answer complex analytical queries on spreadsheet data (e.g., averages, sums, correlations) beyond simple text lookups.
* **Asynchronous & Scalable**: Built with FastAPI for high-performance, non-blocking I/O, ready for concurrent requests.
* **Containerized & Portable**: Fully containerized with Docker for easy deployment and consistent environments.

---

## 🏗️ Architecture & Data Flow

The system is built on a modular, asynchronous architecture. An API Gateway built with FastAPI handles authentication and routes requests to the core Adaptive Engine. This engine inspects the document type and directs the query to the appropriate workflow.

```mermaid
graph TD
    A[Client API Call] --> B{API Gateway<br/>FastAPI};
    B --> C{Adaptive Engine<br/>Route by File Type};
    C --> |Text/PDF| D[RAG Workflow<br/>Parse → Embed → Retrieve → Re-rank];
    C --> |Spreadsheet| E[Agentic Workflow<br/>Pandas Agent Executes Code];
    D --> F(Aggregate Answers);
    E --> F;
    F --> G[JSON Response];
    G --> A;
```

---

## 📂 File Structure & Execution Flow

This section describes the role of each key file and the overall execution flow from request to response.

### File Functionality

* **`main_api.py`**: This is the main entry point of the application. It creates the FastAPI server, defines the `/hackrx/run` API endpoint, and handles HTTP requests, bearer token authentication, and final JSON response formatting.
* **`source/ret_function.py`**: This is the core logic engine of the project. It contains the `insurance_answer` function which implements the adaptive logic to switch between the RAG pipeline (for text) and the Agentic pipeline (for spreadsheets). All document processing, embedding, retrieval, and AI generation happens here.
* **`basemodel/hackrx.py`**: Defines the Pydantic data model (`HackRxRequest`) for incoming API requests. This ensures that any incoming JSON body is validated against a predefined schema, preventing invalid data from being processed.
* **`Dockerfile`**: A multi-stage Dockerfile that provides instructions to build a lightweight, portable, containerized version of the application. It handles system dependencies, Python environment setup, and package installation, ensuring the application runs consistently anywhere.
* **`requirements.txt`**: Lists all the Python packages and their specific versions required for the project to run correctly.
* **`.env`**: A crucial file for security that stores confidential information like API keys and database credentials. It is listed in `.gitignore` to prevent secrets from being committed to version control.

### Execution Flow

1.  A user sends a `POST` request to the `/hackrx/run` endpoint.
2.  The `uvicorn` server receives the request and passes it to the FastAPI application instance defined in **`main_api.py`**.
3.  The `/hackrx/run` endpoint function in **`main_api.py`** is triggered. It first validates the `Authorization: Bearer <token>` header.
4.  The JSON request body is automatically parsed and validated against the `HackRxRequest` Pydantic model defined in **`basemodel/hackrx.py`**.
5.  The endpoint then calls the `insurance_answer` function from **`source/ret_function.py`**, passing the `documents` URL and `questions` list.
6.  Inside **`ret_function.py`**, the function checks the file extension.
    * If it's an `.xlsx` file, it initializes a Pandas Agent.
    * If it's any other text-based file, it initializes the RAG pipeline (parsing with Unstructured, embedding with OpenAI, indexing in Astra DB, retrieving, and re-ranking with Cohere).
7.  After processing, the `insurance_answer` function returns a list of answers.
8.  This list is received back in **`main_api.py`**, which then constructs a final `JSONResponse` and sends it back to the client with a `200 OK` status code.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI |
| **AI Orchestration** | LangChain |
| **LLMs & Embeddings** | OpenAI (GPT-3.5-Turbo) |
| **Retrieval Enhancement**| Cohere (Re-rank Model) |
| **Vector Database** | DataStax Astra DB (Cassandra) |
| **Data Analysis** | Pandas |
| **Document Parsing** | Unstructured.io |
| **Containerization** | Docker |

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.11+
* Docker (Optional, for containerized deployment)
* An IDE like VS Code

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the root directory of the project and add the following keys. Replace `<your-key-here>` with your actual credentials.

```env
# OpenAI API Key
OPENAI_API_KEY="<your-key-here>"

# Cohere API Key
COHERE_API_KEY="<your-key-here>"

# Astra DB Credentials
ASTRA_DB_APPLICATION_TOKEN="<your-key-here>"
ASTRA_DB_ID="<your-database-id-here>"
ASTRA_DB_KEYSPACE="<your-keyspace-name-here>"

# Hackathon API Key for authentication
HACKRX_API_KEY="975f3d8badaa281b00e3629eaeb9ea4900aea2047d026ecbd10381c1d5940255"
```

---

## ⚙️ Usage

### Running the Application

Once the setup is complete, you can start the FastAPI server using Uvicorn.

```bash
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at `http://localhost:8000`.

### API Endpoint

The primary endpoint for interacting with the service is:

**`POST /hackrx/run`**

#### Sample Request (`cURL`)

You can test the endpoint using `cURL` or any API client like Postman.

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 975f3d8badaa281b00e3629eaeb9ea4900aea2047d026ecbd10381c1d5940255" \
-d '{
    "documents": "[https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D](https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D)",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}'
```

#### Sample Response

The API will return a JSON object with the answers.

```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
    ]
}
```

---

## 🐳 Running with Docker

You can also build and run the application as a Docker container.

### 1. Build the Docker Image

```bash
docker build -t hackrx-query-system .
```

### 2. Run the Docker Container

Make sure your `.env` file is present in the root directory.

```bash
docker run -p 8000:8000 --env-file .env hackrx-query-system
```

The containerized application will be accessible at `http://localhost:8000`.
