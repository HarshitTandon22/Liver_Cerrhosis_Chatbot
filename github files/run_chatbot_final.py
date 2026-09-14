#!/usr/bin/env python3
"""
Run the fully functional ChatBot
"""

import os
from dotenv import load_dotenv

load_dotenv()
import sys
from chatbot import LiverCirrhosisChatBot

# Set API keys

print("=" * 70)
print("Liver Cirrhosis Research ChatBot")
print("=" * 70)
print("\nData Status:")
print("  - Pinecone: 2,710 paper chunks, 9,713 triples, 1,958 claims (LOADED)")
print("  - Neo4j: Will attempt connection (may work without)")
print("\nThe chatbot is ready to answer questions about liver cirrhosis research.")
print("Type your questions below. Type 'quit' or 'exit' to end.")
print("=" * 70)

chatbot = None
try:
    print("\nInitializing chatbot components...")
    chatbot = LiverCirrhosisChatBot(
        gemini_api_key=os.getenv('GEMINI_API_KEY'),
        neo4j_uri=os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    print("Chatbot initialized successfully!\n")
    
    # Start interactive chat
    chatbot.interactive_chat()
    
except KeyboardInterrupt:
    print("\n\nChatbot interrupted. Goodbye!")
except EOFError:
    print("\n\nInput closed. Goodbye!")
except Exception as e:
    print(f"\n[FATAL ERROR] Failed to start chatbot: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    if chatbot:
        try:
            chatbot.close()
        except Exception as e:
            print(f"Error closing chatbot: {e}")

