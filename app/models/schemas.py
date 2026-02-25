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


class RetrievedEntity(BaseModel):
    entity_name: str
    entity_type: str
    description: str
    source: str


class RetrievedRelationship(BaseModel):
    src_id: str
    tgt_id: str
    description: str
    keywords: str
    weight: float | None = None
    source: str


class RetrievedReference(BaseModel):
    reference_id: str
    source: str


class ChatOutput(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    entities: list[RetrievedEntity] = Field(default_factory=list)
    relationships: list[RetrievedRelationship] = Field(default_factory=list)
    references: list[RetrievedReference] = Field(default_factory=list)
    error: ErrorInfo | None = None


class GraphState(BaseModel):
    user_input: ChatInput
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    entities: list[RetrievedEntity] = Field(default_factory=list)
    relationships: list[RetrievedRelationship] = Field(default_factory=list)
    references: list[RetrievedReference] = Field(default_factory=list)
    answer: str | None = None
    error: ErrorInfo | None = None
