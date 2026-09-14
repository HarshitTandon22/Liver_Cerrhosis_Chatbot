import os
import sys
import json
from typing import Optional

import httpx


class LiverCirrhosisChatBot:
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        server_url: str = "http://127.0.0.1:8000",
    ) -> None:
        # Honor provided API keys (also used by the MCP server if running in same env)
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
        # Pinecone keys are expected from env or config.yaml defaults
        self.server_url = server_url.rstrip("/")

    def ask(self, question: str) -> str:
        payload = {"question": question, "filters": {}}
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{self.server_url}/rag.answer_with_evidence", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("answer", "")

    def interactive_chat(self) -> None:
        print("Type your question (or 'exit' to quit)\n")
        while True:
            try:
                q = input("You: ")
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in {"quit", "exit"}:
                break
            try:
                ans = self.ask(q)
                print(f"\nBot: {ans}\n")
            except Exception as e:
                print(f"\n[Error] {e}\n")

    def close(self) -> None:
        pass

#!/usr/bin/env python3
"""
ChatBot using Gemini with Knowledge Graph and MCP Server
Provides precise answers based on KG and RAG evidence.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from google import generativeai as genai
from mcp_server import MCPServer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiverCirrhosisChatBot:
    """ChatBot for liver cirrhosis research using Gemini, KG, and RAG."""
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 neo4j_uri: str = "neo4j://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: Optional[str] = None):
        """
        Initialize ChatBot.
        
        Args:
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Initialize Gemini
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in environment or pass as parameter.")
        
        genai.configure(api_key=api_key)
        # Use model IDs that are actually available for this API version.
        # These were discovered via list_gemini_models.py (see models/* names).
        self.model_names = [
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash',
            'models/gemini-flash-latest',
        ]
        
        self.model = None
        self.current_model_name = None
        for model_name in self.model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                self.current_model_name = model_name
                logger.info(f"Using Gemini model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
                continue
        
        if self.model is None:
            raise ValueError("Could not initialize any Gemini model. Please check your API key and model availability.")
        
        neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        # Initialize MCP Server (handle Neo4j connection errors gracefully)
        try:
            self.mcp_server = MCPServer(neo4j_uri, neo4j_user, neo4j_password)
            self.kg_available = True
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Continuing without KG features.")
            self.mcp_server = None
            self.kg_available = False
        
        # System prompt
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
        
        logger.info("Liver Cirrhosis ChatBot initialized")
    
    def _generate_with_retry(self, prompt: str, evidence_context: list, max_retries: int = 3) -> str:
        """
        Generate response with retry logic and model fallback for quota errors.
        
        Args:
            prompt: The prompt to send to Gemini
            evidence_context: Evidence context for fallback messages
            max_retries: Maximum number of retries per model
            
        Returns:
            Generated answer text
        """
        import time
        from google.api_core import exceptions as google_exceptions
        
        # Try current model first, then fallback to other models
        models_to_try = [self.current_model_name] + [m for m in self.model_names if m != self.current_model_name]
        
        for model_name in models_to_try:
            try:
                # Create model instance
                model = genai.GenerativeModel(model_name)
                logger.info(f"Attempting generation with model: {model_name}")
                
                # Retry with exponential backoff for quota errors
                for attempt in range(max_retries):
                    try:
                        response = model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": 0.3,
                                "top_p": 0.95,
                                "top_k": 40,
                                "max_output_tokens": 2048,
                            }
                        )
                        answer_text = response.text
                        logger.info(f"Generated response of {len(answer_text)} characters using {model_name}")
                        # Update current model if we successfully used a different one
                        if model_name != self.current_model_name:
                            self.model = model
                            self.current_model_name = model_name
                            logger.info(f"Switched to model: {model_name}")
                        return answer_text
                        
                    except google_exceptions.ResourceExhausted as e:
                        # Check if it's a quota error
                        error_str = str(e)
                        if "quota" in error_str.lower() or "429" in error_str:
                            if attempt < max_retries - 1:
                                # Extract retry delay if available
                                retry_delay = 10  # Default 10 seconds
                                if "retry in" in error_str.lower():
                                    try:
                                        # Try to extract retry delay from error message
                                        import re
                                        match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                                        if match:
                                            retry_delay = max(5, int(float(match.group(1))))
                                    except:
                                        pass
                                
                                logger.warning(f"Quota exceeded for {model_name}. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                                time.sleep(retry_delay)
                                continue
                            else:
                                # Max retries reached for this model, try next model
                                logger.warning(f"Quota exceeded for {model_name} after {max_retries} attempts. Trying next model...")
                                break
                        else:
                            # Other ResourceExhausted error, re-raise
                            raise
                            
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
                # Try next model
                continue
        
        # All models failed, return fallback message
        logger.error("All Gemini models failed. Using fallback response.")
        if evidence_context:
            answer_text = f"Based on the retrieved evidence: {evidence_context[0][:500]}... (Note: Full answer generation encountered quota/rate limit errors. Please try again in a few moments.)"
        else:
            answer_text = "I encountered quota/rate limit errors with all available models. Please wait a moment and try again, or check your API quota limits at https://ai.dev/usage?tab=rate-limit"
        
        return answer_text
    
    def chat(self, query: str, use_kg: bool = True, use_rag: bool = True,
            max_evidence: int = 5) -> Dict[str, Any]:
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
        logger.info(f"Processing query: {query}")
        
        # Gather evidence
        evidence_context = []
        kg_context = None
        rag_evidence = None
        
        # Get RAG evidence
        if use_rag and self.mcp_server:
            try:
                rag_result = self.mcp_server.rag_answer_with_evidence(query, top_k=max_evidence)
                rag_evidence = rag_result.get('evidence', [])
                evidence_context.append(f"Retrieved {len(rag_evidence)} relevant evidence items.")
                
                # Add top evidence to context with more detail
                for i, ev in enumerate(rag_evidence[:max_evidence], 1):
                    meta = ev.get('metadata', {}) or {}
                    text = meta.get('content', '') or \
                           meta.get('text', '') or \
                           meta.get('triple_text', '')
                    namespace = ev.get('namespace', 'unknown')
                    
                    # Derive a more informative source so it is not shown as "unknown"
                    source = meta.get('source') or \
                             meta.get('original_filename') or \
                             meta.get('document_id')
                    if not source:
                        # Fall back to namespace label if nothing else is available
                        if namespace in {'papers', 'triples', 'claims'}:
                            source = f"{namespace}_namespace"
                        else:
                            source = 'unspecified_source'
                    
                    score = ev.get('score', 0)
                    
                    # Include more context (500 chars instead of 300)
                    evidence_context.append(
                        f"\n--- Evidence {i} (Relevance Score: {score:.3f}, Source: {source}, Type: {namespace}) ---\n{text[:500]}"
                    )
            except Exception as e:
                logger.error(f"Error in RAG retrieval: {e}")
                evidence_context.append("RAG retrieval encountered an error.")
        
        # Get KG context
        if use_kg and self.kg_available and self.mcp_server:
            try:
                # Search for relevant entities
                kg_results = self.mcp_server.kg_search_graph(query, limit=5)
                if kg_results.get('results'):
                    kg_context = kg_results
                    evidence_context.append(f"\nKnowledge Graph found {len(kg_results['results'])} relevant entities:")
                    for entity in kg_results['results'][:3]:
                        evidence_context.append(f"- {entity.get('name', '')} ({entity.get('normalized_name', '')})")
                
                # Find relations
                if kg_results.get('results'):
                    first_entity = kg_results['results'][0].get('name', '')
                    if first_entity:
                        relations = self.mcp_server.kg_find_relations(first_entity, limit=3)
                        if relations.get('relations'):
                            evidence_context.append(f"\nFound {len(relations['relations'])} relations:")
                            for rel in relations['relations'][:3]:
                                subj = rel.get('subject', {}).get('name', '')
                                obj = rel.get('object', {}).get('name', '')
                                rel_type = rel.get('relation', {}).get('type', '')
                                evidence_context.append(f"- {subj} --[{rel_type}]--> {obj}")
            except Exception as e:
                logger.error(f"Error in KG retrieval: {e}")
                evidence_context.append("Knowledge Graph retrieval encountered an error.")
        
        # Build prompt with better formatting
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

        # Generate response with Gemini (with retry and model fallback)
        answer_text = self._generate_with_retry(prompt, evidence_context)
        
        # Build response
        response_data = {
            'query': query,
            'answer': answer_text,
            'evidence_count': len(rag_evidence) if rag_evidence else 0,
            'kg_entities_found': len(kg_context.get('results', [])) if kg_context else 0,
            'sources': self._extract_sources(rag_evidence) if rag_evidence else [],
            'timestamp': None
        }
        
        from datetime import datetime
        response_data['timestamp'] = datetime.now().isoformat()
        
        return response_data
    
    def _extract_sources(self, evidence: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from evidence."""
        sources = set()
        for ev in evidence:
            meta = ev.get('metadata', {}) or {}
            namespace = ev.get('namespace', '')
            # Prefer explicit source field, otherwise fall back to filename / document id
            source = meta.get('source') or meta.get('original_filename') or meta.get('document_id')
            if not source and namespace:
                source = f"{namespace}_namespace"
            if source:
                sources.add(str(source))
        return list(sources)
    
    def interactive_chat(self):
        """Interactive chat interface."""
        import sys
        
        print("=" * 70)
        print("Liver Cirrhosis Research ChatBot")
        print("=" * 70)
        print("Ask questions about liver cirrhosis and related conditions.")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        print("=" * 70)
        
        # Check if stdin is available for interactive input
        if not sys.stdin.isatty():
            print("\nWarning: Non-interactive terminal detected.")
            print("The chatbot will work, but you may need to run this in a terminal.")
            print("=" * 70)
        
        while True:
            try:
                # Get user input
                print("\n" + "-" * 70)
                query = input("You: ").strip()
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'q', 'bye']:
                    print("\nThank you for using the Liver Cirrhosis Research ChatBot. Goodbye!")
                    break
                
                # Skip empty queries
                if not query:
                    print("Please enter a question. Type 'quit' to exit.")
                    continue
                
                # Process query
                print("\n[Processing your query...]")
                try:
                    response = self.chat(query, use_rag=True, use_kg=self.kg_available, max_evidence=5)
                    
                    # Display answer
                    print("\n" + "=" * 70)
                    print("ChatBot:")
                    print("=" * 70)
                    print(response['answer'])
                    print("=" * 70)
                    
                    # Display metadata
                    if response.get('sources'):
                        print(f"\nSources: {', '.join(response['sources'][:5])}")
                    
                    if response.get('evidence_count', 0) > 0:
                        print(f"Evidence Retrieved: {response['evidence_count']} items from research papers")
                    
                except Exception as chat_error:
                    logger.error(f"Error processing query: {chat_error}")
                    print(f"\n[Error] I encountered an issue processing your query: {chat_error}")
                    print("Please try rephrasing your question or ask something else.")
                    continue
                
            except EOFError:
                # Handle EOF (end of file/input stream)
                print("\n\nInput stream closed. Goodbye!")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                # Handle any other unexpected errors
                logger.error(f"Unexpected error in interactive chat: {e}")
                print(f"\n[Error] An unexpected error occurred: {e}")
                print("The chatbot will continue. Please try again or type 'quit' to exit.")
                continue
    
    def close(self):
        """Close connections."""
        if self.mcp_server:
            self.mcp_server.close()
        logger.info("ChatBot closed")

def main():
    """Main function."""
    import sys
    
    print("Liver Cirrhosis Research ChatBot")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables or .env file.")
        print("Please set it in .env file or as environment variable.")
        sys.exit(1)
    
    try:
        # Initialize chatbot with API key from .env
        chatbot = LiverCirrhosisChatBot(gemini_api_key=api_key)
        
        # Interactive mode
        chatbot.interactive_chat()
        
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            chatbot.close()
        except:
            pass

if __name__ == "__main__":
    main()

