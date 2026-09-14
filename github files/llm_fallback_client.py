#!/usr/bin/env python3
"""
Unified LLM client with fallback:
1) Gemini (models/gemini-2.5-flash)
2) Groq   (LLaMA 3 8B instant)
3) Ollama local (llama3:8b)

Dependencies:
- google-generativeai
- groq
- requests
"""

import os
from typing import Optional, Tuple

import requests
from google import generativeai as genai
from groq import Groq


class LLMFallbackClient:
    def __init__(
        self,
        gemini_model: str = "models/gemini-2.5-flash",
        groq_model: str = "llama-3.1-8b-instant",
        ollama_model: str = "llama3:8b",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        # Gemini
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

        # Groq
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.groq_model = groq_model
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

        # Ollama
        self.ollama_model = ollama_model
        self.ollama_chat_url = f"{ollama_url.rstrip('/')}/api/chat"

    def _build_messages(self, system_prompt: Optional[str], user_prompt: str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Tuple[str, str]:
        """
        Returns:
            (text, backend_name) where backend_name in {"gemini", "groq", "ollama"}.
        Raises:
            RuntimeError if all backends fail.
        """
        messages = self._build_messages(system_prompt, user_prompt)

        # 1) Gemini
        if self.gemini_key:
            try:
                model = genai.GenerativeModel(self.gemini_model)
                resp = model.generate_content(
                    messages,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                return resp.text, "gemini"
            except Exception as e:
                print(f"[LLM] Gemini failed: {e}")

        # 2) Groq
        if self.groq_client:
            try:
                resp = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content
                return text, "groq"
            except Exception as e:
                print(f"[LLM] Groq failed: {e}")

        # 3) Ollama
        try:
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "gpu": True,
                },
            }
            r = requests.post(self.ollama_chat_url, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            text = data.get("message", {}).get("content", "")
            return text, "ollama"
        except Exception as e:
            print(f"[LLM] Ollama failed: {e}")

        raise RuntimeError("All LLM backends (Gemini, Groq, Ollama) failed.")

