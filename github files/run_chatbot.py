#!/usr/bin/env python3
"""
Run ChatBot with loaded data
"""

import os
from dotenv import load_dotenv

load_dotenv()
from chatbot import LiverCirrhosisChatBot

# Set API keys

print("=" * 60)
print("Liver Cirrhosis Research ChatBot")
print("=" * 60)
print("\nData loaded:")
print("  - Pinecone: 2,710 paper chunks, 9,713 triples, 1,958 claims")
print("  - Neo4j: Connection issues (will work without KG)")
print("\nStarting chatbot...")
print("=" * 60)

try:
    chatbot = LiverCirrhosisChatBot(
        gemini_api_key=os.environ['GEMINI_API_KEY'],
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    chatbot.interactive_chat()
    
except KeyboardInterrupt:
    print("\n\nGoodbye!")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        chatbot.close()
    except:
        pass





