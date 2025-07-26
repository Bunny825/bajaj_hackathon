from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from source.ret_function import insurance_answer
from basemodel.hackrx import HackRxRequest
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

# Set up security scheme
security = HTTPBearer()

# Token verification dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_token = os.getenv('HACKRX_API_KEY')
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/")
def home():
    return {
        "message": "This is the home endpoint. Our aim is to answer questions related to insurance."
    }

@app.post("/hackrx/run")
async def run_hackrx(
    request: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    try:
        url = request.documents
        queries = request.questions
        answers = insurance_answer(url, queries)
        return JSONResponse(status_code=200, content={"answers": answers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
