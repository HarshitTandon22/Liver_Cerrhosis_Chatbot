#!/usr/bin/env python3
"""
Demo ChatBot - Run sample queries to demonstrate functionality
"""

import os
from dotenv import load_dotenv

load_dotenv()
from chatbot import LiverCirrhosisChatBot

# Set API keys

# Sample queries to demonstrate
demo_queries = [
    "What is liver cirrhosis?",
    "What are the main symptoms of liver cirrhosis?",
    "What causes liver cirrhosis?",
    "What treatments are available for liver cirrhosis?"
]

print("=" * 80)
print("Liver Cirrhosis Research ChatBot - Demo")
print("=" * 80)
print("\nThis demo will run a few sample queries to show the chatbot in action.")
print("=" * 80)

try:
    print("\nInitializing chatbot...")
    chatbot = LiverCirrhosisChatBot(
        gemini_api_key=os.environ['GEMINI_API_KEY'],
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    print("Chatbot initialized successfully!\n")
    
    for i, query in enumerate(demo_queries, 1):
        print("\n" + "=" * 80)
        print(f"QUERY {i}/{len(demo_queries)}: {query}")
        print("=" * 80)
        print("Processing...")
        
        response = chatbot.chat(query, use_rag=True, use_kg=False, max_evidence=5)
        
        print("\nANSWER:")
        print("-" * 80)
        print(response['answer'])
        print("-" * 80)
        print(f"\nStatistics:")
        print(f"  - Evidence Items Retrieved: {response.get('evidence_count', 0)}")
        print(f"  - Answer Length: {len(response['answer'])} characters")
        if response.get('sources'):
            print(f"  - Sources: {', '.join(response['sources'][:3])}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print("\nTo use the interactive chatbot, run:")
    print("  python run_chatbot_final.py")
    print("=" * 80)
    
    chatbot.close()
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()





