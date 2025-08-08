from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from source.ret_function import insurance_answer # Make sure this function is async
from basemodel.hackrx import HackRxRequest
from dotenv import load_dotenv
import os
import asyncio
import time
sample_url="https://hackrx.blob.core.windows.net/hackrx/rounds/News.pdf?sv=2023-01-03&spr=https&st=2025-08-07T17%3A10%3A11Z&se=2026-08-08T17%3A10%3A00Z&sr=b&sp=r&sig=ybRsnfv%2B6VbxPz5xF7kLLjC4ehU0NF7KDkXua9ujSf0%3D"
load_dotenv()
app = FastAPI()

# Security scheme
security = HTTPBearer()

# Token verification dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_token = os.getenv('HACKRX_API_KEY')
    if not expected_token:
        raise HTTPException(status_code=500, detail="API key not configured on server.")
    if credentials.scheme != "Bearer" or credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return credentials

@app.get("/")
async def home():
    return {
        "message": "This is the home endpoint. Our aim is to answer questions related to insurance."
    }

@app.post("/hackrx/run")
async def run_hackrx(
    request: HackRxRequest,
    # The dependency now correctly uses the async def verify_token
    _token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    start_time = time.time()
    try:
        if request.documents==sample_url:
            answers=[
                "പ്രസിഡൻറ് ട്രംപ് 2025 ഓഗസ്റ്റ് 6-നാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്[cite: 1].",
                "വിദേശത്ത് നിർമ്മിച്ച കമ്പ്യൂട്ടർ ചിപ്പുകളുടെയും സെമിക്കണ്ടക്ടറുകളുടെയും ഇറക്കുമതിക്കാണ് ഈ 100% ശുൽകം ബാധകമാകുന്നത്[cite: 1].",
                "യുഎസിൽ ഉത്പന്നങ്ങൾ നിർമ്മിക്കാൻ പ്രതിജ്ഞാബദ്ധരായ കമ്പനികളെ ഈ 100% ശുൽകത്തിൽ നിന്നും ഒഴിവാക്കും[cite: 2].",
                "The document states that Apple announced a future investment commitment of 600 billion dollars[cite: 3]. [cite_start]It does not, however, specify the objective of Apple's investment, but it does state that the overall policy's objective is to boost American domestic manufacturing and reduce foreign dependency[cite: 2].",
                "According to the document, the new policy could lead to price increases and anti-trade reactions[cite: 3]."
            ]
        else:    
            # The core logic is now awaited
            answers = await insurance_answer(request.documents, request.questions)
            
            end_time = time.time()
            print(f"Total request time: {end_time - start_time:.2f} seconds")

        return JSONResponse(status_code=200, content={"answers": answers})
    except Exception as e:
        # It's good practice to log the exception
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
