#!/usr/bin/env python3
"""
ChatBot using Ollama (local LLaMA 3) with Knowledge Graph and MCP Server.

This is a copy of the existing Gemini-based chatbot, modified so that
answer generation uses a local Ollama model instead of Gemini.

The original `chatbot.py` is left unchanged.
"""

import os
import logging
from typing import Dict, Any, Optional, List

import requests
from dotenv import load_dotenv

from mcp_server import MCPServer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LiverCirrhosisChatBotOllama:
    """ChatBot for liver cirrhosis research using Ollama, KG, and RAG."""

    def __init__(
        self,
        neo4j_uri: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = None,
        ollama_model: str = "llama3:8b",
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize ChatBot backed by Ollama.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            ollama_model: Ollama model name (e.g., 'llama3:8b')
            ollama_url: Base URL of the Ollama instance
        """

        self.ollama_model = ollama_model
        self.ollama_chat_url = f"{ollama_url.rstrip('/')}/api/chat"

        neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')

        # Initialize MCP Server (handle Neo4j connection errors gracefully)
        try:
            self.mcp_server = MCPServer(neo4j_uri, neo4j_user, neo4j_password)
            self.kg_available = True
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Continuing without KG features.")
            self.mcp_server = None
            self.kg_available = False

        # System prompt (copied from original chatbot)
        self.system_prompt = """You are a specialized medical research assistant focused on liver cirrhosis and related conditions.
Your role is to provide comprehensive, evidence-based answers using the knowledge graph and research papers.

CORE PRINCIPLES:
1. Base your answers STRICTLY on the evidence provided in the context
2. Synthesize information from multiple evidence sources when available
3. Provide detailed, well-structured answers with proper medical terminology
4. Cite specific evidence sources when mentioning facts or findings
5. If evidence is insufficient, clearly state what information is available and what is missing
6. Do not make up information or provide general medical advice beyond the evidence
7. When multiple sources agree, emphasize that consensus
8. When sources conflict, mention the different perspectives
9. Use clear formatting: bullet points, numbered lists, or structured paragraphs as appropriate
10. Be thorough but concise - provide complete answers without unnecessary repetition

ANSWER FORMAT:
- Start with a direct answer to the question
- Provide supporting details from the evidence
- Include relevant statistics, findings, or relationships when available
- Cite sources using format: (Source: [source_name])
- End with limitations or uncertainties if applicable

Remember: Your goal is to provide the most comprehensive and accurate answer possible based solely on the provided evidence."""

        logger.info("Liver Cirrhosis ChatBot (Ollama) initialized")

    # --- Ollama-backed generation ---

    def _generate_with_ollama(self, prompt: str, evidence_context: List[str]) -> str:
        """
        Generate a response using the local Ollama model.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.ollama_model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
                "gpu": True,
            },
        }

        try:
            resp = requests.post(self.ollama_chat_url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            if not text:
                raise ValueError("Empty response text from Ollama.")
            logger.info("Generated response of %d characters using Ollama (%s)", len(text), self.ollama_model)
            return text
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            if evidence_context:
                return (
                    f"Based on the retrieved evidence: {evidence_context[0][:500]}... "
                    "(Note: Local model generation encountered an error. Please try again.)"
                )
            return "I encountered an error while generating the answer with the local model."

    # --- Main chat logic (copied + adapted from original chatbot) ---

    def chat(
        self,
        query: str,
        use_kg: bool = True,
        use_rag: bool = True,
        max_evidence: int = 5,
    ) -> Dict[str, Any]:
        """
        Chat with the bot.

        Args:
            query: User query
            use_kg: Whether to use knowledge graph
            use_rag: Whether to use RAG retrieval
            max_evidence: Maximum number of evidence items to include

        Returns:
            Response with answer and evidence
        """
        logger.info(f"Processing query (Ollama backend): {query}")

        # Gather evidence
        evidence_context: List[str] = []
        kg_context = None
        rag_evidence = None

        # Get RAG evidence
        if use_rag and self.mcp_server:
            try:
                rag_result = self.mcp_server.rag_answer_with_evidence(query, top_k=max_evidence)
                rag_evidence = rag_result.get("evidence", [])
                evidence_context.append(f"Retrieved {len(rag_evidence)} relevant evidence items.")

                # Add top evidence to context with more detail
                for i, ev in enumerate(rag_evidence[:max_evidence], 1):
                    meta = ev.get("metadata", {}) or {}
                    text = meta.get("content", "") or meta.get("text", "") or meta.get("triple_text", "")
                    namespace = ev.get("namespace", "unknown")

                    source = (
                        meta.get("source")
                        or meta.get("original_filename")
                        or meta.get("document_id")
                    )
                    if not source:
                        if namespace in {"papers", "triples", "claims"}:
                            source = f"{namespace}_namespace"
                        else:
                            source = "unspecified_source"

                    score = ev.get("score", 0)

                    evidence_context.append(
                        f"\n--- Evidence {i} (Relevance Score: {score:.3f}, "
                        f"Source: {source}, Type: {namespace}) ---\n{text[:500]}"
                    )
            except Exception as e:
                logger.error(f"Error in RAG retrieval: {e}")
                evidence_context.append("RAG retrieval encountered an error.")

        # Get KG context
        if use_kg and self.kg_available and self.mcp_server:
            try:
                kg_results = self.mcp_server.kg_search_graph(query, limit=5)
                if kg_results.get("results"):
                    kg_context = kg_results
                    evidence_context.append(
                        f"\nKnowledge Graph found {len(kg_results['results'])} relevant entities:"
                    )
                    for entity in kg_results["results"][:3]:
                        evidence_context.append(
                            f"- {entity.get('name', '')} ({entity.get('normalized_name', '')})"
                        )

                if kg_results.get("results"):
                    first_entity = kg_results["results"][0].get("name", "")
                    if first_entity:
                        relations = self.mcp_server.kg_find_relations(first_entity, limit=3)
                        if relations.get("relations"):
                            evidence_context.append(
                                f"\nFound {len(relations['relations'])} relations:"
                            )
                            for rel in relations["relations"][:3]:
                                subj = rel.get("subject", {}).get("name", "")
                                obj = rel.get("object", {}).get("name", "")
                                rel_type = rel.get("relation", {}).get("type", "")
                                evidence_context.append(f"- {subj} --[{rel_type}]--> {obj}")
            except Exception as e:
                logger.error(f"Error in KG retrieval: {e}")
                evidence_context.append("Knowledge Graph retrieval encountered an error.")

        evidence_text = "\n".join(evidence_context) if evidence_context else "No evidence retrieved from the knowledge base."

        prompt = f"""{self.system_prompt}

User Query: {query}

Evidence Retrieved from Research Papers:
{evidence_text}

Instructions:
1. Analyze the evidence provided above carefully
2. Extract relevant information that directly answers the user's query
3. Synthesize the information from multiple evidence sources when available
4. Provide a comprehensive, well-structured answer based on the evidence
5. If multiple pieces of evidence mention the same information, combine them
6. Cite specific evidence sources when mentioning facts
7. If the evidence is insufficient, state what information is available and what is missing
8. Use medical terminology accurately based on the evidence

Please provide a detailed answer based on the evidence above."""

        answer_text = self._generate_with_ollama(prompt, evidence_context)

        from datetime import datetime

        response_data: Dict[str, Any] = {
            "query": query,
            "answer": answer_text,
            "evidence_count": len(rag_evidence) if rag_evidence else 0,
            "kg_entities_found": len(kg_context.get("results", [])) if kg_context else 0,
            "sources": self._extract_sources(rag_evidence) if rag_evidence else [],
            "timestamp": datetime.now().isoformat(),
        }

        return response_data

    def _extract_sources(self, evidence: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from evidence."""
        sources = set()
        for ev in evidence:
            meta = ev.get("metadata", {}) or {}
            namespace = ev.get("namespace", "")
            source = (
                meta.get("source")
                or meta.get("original_filename")
                or meta.get("document_id")
            )
            if not source and namespace:
                source = f"{namespace}_namespace"
            if source:
                sources.add(str(source))
        return list(sources)

    def interactive_chat(self) -> None:
        """Simple interactive chat loop using Ollama backend."""
        print("=" * 70)
        print("Liver Cirrhosis Research ChatBot (Ollama)")
        print("=" * 70)
        print("Type your question (or 'quit' to exit).")
        print("=" * 70)
        while True:
            try:
                q = input("\nYou: ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                break
            resp = self.chat(q, use_rag=True, use_kg=self.kg_available, max_evidence=5)
            print("\nBot:")
            print(resp["answer"])

    def close(self) -> None:
        if self.mcp_server:
            self.mcp_server.close()
        logger.info("Ollama ChatBot closed")


