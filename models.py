from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    language: Optional[str] = None   # kept for compatibility, but ignored
    clear_history: Optional[bool] = False  # New: clear conversation history

class SourceNode(BaseModel):
    title: str
    path: str
    node_id: str

class ChatResponse(BaseModel):
    answer: str
    detected_language: str
    sources: List[SourceNode] = []
    confidence: float
    session_id: Optional[str] = None
    conversation_turn: Optional[int] = None  # Track conversation count

class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: List[ConversationMessage]
    total_turns: int

class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_nodes: int
    gemini_model: str

class IndexStatsResponse(BaseModel):
    total_sections: int
    total_nodes: int
    top_level_sections: List[str]
    last_updated: Optional[str]

class SuggestedQuestionsResponse(BaseModel):
    questions: List[str]