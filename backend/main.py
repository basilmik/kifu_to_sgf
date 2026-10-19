from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from _kifu_core import add  # pybind11-модуль, собранный CMake

app = FastAPI()

# На проде фронт и бэк на одном домене, но для локальной разработки CORS нужен
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SumRequest(BaseModel):
    a: int
    b: int


class SumResponse(BaseModel):
    result: int



@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/sum", response_model=SumResponse)
def sum_endpoint(req: SumRequest):
    return SumResponse(result=add(req.a, req.b))