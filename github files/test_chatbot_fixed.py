#!/usr/bin/env python3
"""
Test the fixed chatbot with a sample query
"""

import os
from dotenv import load_dotenv

load_dotenv()
import sys
from chatbot import LiverCirrhosisChatBot

# Set API keys

print("=" * 60)
print("Testing Fixed ChatBot")
print("=" * 60)

try:
    chatbot = LiverCirrhosisChatBot(
        gemini_api_key=os.environ['GEMINI_API_KEY'],
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    # Test query
    test_query = "What are the main symptoms of liver cirrhosis?"
    print(f"\nTest Query: {test_query}")
    print("\nProcessing...")
    
    response = chatbot.chat(test_query, use_rag=True, use_kg=False)  # Use RAG only since Neo4j may not be available
    
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(response['answer'])
    print("\n" + "=" * 60)
    print(f"Evidence Count: {response.get('evidence_count', 0)}")
    if response.get('sources'):
        print(f"Sources: {', '.join(response['sources'][:5])}")
    print("=" * 60)
    
    chatbot.close()
    
    print("\nChatbot test completed successfully!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

