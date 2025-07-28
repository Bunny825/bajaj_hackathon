from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from source.ret_function import insurance_answer # Make sure this function is async
from basemodel.hackrx import HackRxRequest
from dotenv import load_dotenv
import os
import asyncio
import time

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
        # The core logic is now awaited
        answers = await insurance_answer(request.documents, request.questions)
        
        end_time = time.time()
        print(f"Total request time: {end_time - start_time:.2f} seconds")

        return JSONResponse(status_code=200, content={"answers": answers})
    except Exception as e:
        # It's good practice to log the exception
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))