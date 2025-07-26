from pydantic import BaseModel

class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]