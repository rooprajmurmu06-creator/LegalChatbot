"""
legal_kg_engine.py
Wraps the LegalKGSearchEngine for use in the FastAPI backend.
English only – no language detection.
"""

import sys
from io import StringIO
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
import uuid

from legal_search_engine import LegalKGSearchEngine

@dataclass
class KGResult:
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    detected_language: str = "en"   # always English

class ConversationMemory:
    """Manages conversation history per session"""
    def __init__(self, max_history: int = 10):
        self.sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.max_history = max_history
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history"""
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": None  # Could add timestamp if needed
        })
        
        # Keep only last N messages (user + assistant pairs)
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]
    
    def get_history(self, session_id: str, last_n_turns: int = 5) -> List[Dict[str, str]]:
        """Get conversation history for context"""
        if session_id not in self.sessions:
            return []
        # Return last N turns (each turn = user + assistant)
        return self.sessions[session_id][-last_n_turns * 2:]
    
    def get_history_text(self, session_id: str, last_n_turns: int = 5) -> str:
        """Get formatted conversation history as text for context"""
        history = self.get_history(session_id, last_n_turns)
        if not history:
            return ""
        
        formatted = "\n### Previous Conversation:\n"
        for msg in history:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            else:
                formatted += f"Assistant: {msg['content']}\n"
        return formatted
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self.sessions

class LegalKGWrapper:
    def __init__(self):
        # Suppress verbose prints from the engine during startup
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            self.engine = LegalKGSearchEngine()
            self.memory = ConversationMemory(max_history=15)  # Store up to 15 turns
        finally:
            sys.stdout = self._original_stdout

    async def answer(self, query: str, session_id: Optional[str] = None, 
                     include_history: bool = True) -> KGResult:
        # Generate a session ID if none provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Add user message to memory
        self.memory.add_message(session_id, "user", query)
        
        # Search knowledge graph
        facts = self.engine.search(query, max_kgs=3)
        
        if not facts:
            # Still try to answer using conversation context if available
            if include_history and self.memory.session_exists(session_id):
                # Use just history to answer follow-up questions
                context = self.memory.get_history_text(session_id, last_n_turns=3)
                if context:
                    try:
                        from gemini_client import get_gemini_client
                        client = get_gemini_client()
                        # Create a prompt that uses conversation context
                        prompt = f"""You are a legal assistant for Indian law. The user is asking a follow-up question based on previous conversation.

{context}

Current question: {query}

Please answer based on the conversation context. If the question refers to something from previous messages, use that context. If you don't know, say so politely."""
                        answer_text = await client.generate(prompt)
                        confidence = 0.7
                        self.memory.add_message(session_id, "assistant", answer_text)
                        return KGResult(
                            answer=answer_text,
                            sources=[],
                            confidence=confidence,
                            detected_language="en"
                        )
                    except:
                        pass
            
            return KGResult(
                answer="I do not know based on the available legal knowledge graph.",
                confidence=0.0,
                detected_language="en"
            )

        try:
            from gemini_client import get_gemini_client
            client = get_gemini_client()
            
            # Build context with facts and conversation history
            context = "\n".join(facts[:20])
            
            if include_history:
                # Add conversation history to context
                history_context = self.memory.get_history_text(session_id, last_n_turns=5)
                full_context = history_context + "\n### Knowledge Graph Facts:\n" + context
            else:
                full_context = context
            
            # Modify the prompt to be aware of conversation context
            prompt = f"""You are a legal assistant for Indian law. Your task is to answer questions based on the knowledge graph facts AND the conversation history.

### Rules:
1. Use the knowledge graph facts as your primary source of information.
2. Use the conversation history to understand follow-up questions and references.
3. If the user asks "what about X" or "tell me more", refer to the previous discussion.
4. Do not repeat information unless asked.
5. Be concise and helpful.
6. If you don't know something, say so clearly.

{full_context}

### Question:
{query}

### Answer:"""
            
            answer_text = await client.generate(prompt)
            confidence = 0.85
            
            # Add assistant response to memory
            self.memory.add_message(session_id, "assistant", answer_text)
            
        except Exception as e:
            print(f"Error in answer generation: {e}")
            # Fallback: return raw facts
            answer_text = "Relevant facts:\n" + "\n".join(facts[:10])
            confidence = 0.6
            self.memory.add_message(session_id, "assistant", answer_text)

        return KGResult(
            answer=answer_text,
            sources=facts[:5],
            confidence=confidence,
            detected_language="en"
        )
    
    def clear_history(self, session_id: str):
        """Clear conversation history for a session"""
        self.memory.clear_session(session_id)
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        return self.memory.get_history(session_id)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return self.memory.session_exists(session_id)