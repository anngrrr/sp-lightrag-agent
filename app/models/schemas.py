from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str
    message: str


class ChatInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    content: str
    score: float | None = None


class Citation(BaseModel):
    source: str
    chunk_id: str
    quote: str


class ChatOutput(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    error: ErrorInfo | None = None


class GraphState(BaseModel):
    user_input: ChatInput
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    error: ErrorInfo | None = None
