"""
gemini_client.py  — Local LLM client using Ollama
with the exact prompt from the standalone CLI main.py (English only)
"""

import os
import json
import logging
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

class GeminiClient:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model = LLM_MODEL
        self._verify_connection()

    def _verify_connection(self):
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as r:
                data = json.loads(r.read())
                available = [m["name"] for m in data.get("models", [])]
                if not any(self.model in m for m in available):
                    logger.warning(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
                else:
                    logger.info(f"Model '{self.model}' available")
        except Exception as e:
            logger.warning(f"Cannot reach Ollama at {self.base_url}: {e}")

    async def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": 1024,
                "num_ctx": 4096,
            },
        }

        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "").strip()
            if not text:
                raise RuntimeError("Ollama returned empty response")
            return text

    async def formulate_answer(self, query: str, context: str) -> str:
        """Use the exact CLI prompt (English only, allows own knowledge)"""
        prompt = f"""You are a legal assistant for Indian law. Your task is to answer questions based **strictly** on the knowledge graph facts provided below.

### Rules:
1. You can use your own knowledge.

3. If multiple facts are relevant, synthesize them into a concise answer.
4. For relationship questions, clearly state the subject, relationship, and object.
5. For yes/no questions, answer with "Yes" or "No" followed by a brief justification from the facts.
6. Read questions carefully and ensure your answer addresses all parts of the question.
7. Format your answer in plain English, no bullet points unless multiple items.
8. Use all the information available in the facts to provide the most complete answer possible.
9. When asked about full forms and its not present in the data then just say you dont know.

### Knowledge Graph Facts (triples in format: subject --[relation]--> object):
do not print triples in the output. example for a bad output: "Supreme Court of India --[has jurisdiction over]--> All of India"
use the following facts to answer the question. Do not hallucinate any facts that are not present here.
{context}

### Question:
{query}

### Answer:"""
        return await self.generate(prompt)


_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client