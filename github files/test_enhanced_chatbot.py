#!/usr/bin/env python3
"""
Test Enhanced ChatBot with improved Gemini integration
"""

import os
from dotenv import load_dotenv

load_dotenv()
import sys
from chatbot import LiverCirrhosisChatBot

# Set API keys

# Diverse test queries
test_queries = [
    "What are the main symptoms of liver cirrhosis?",
    "What causes liver cirrhosis?",
    "What treatments are available for liver cirrhosis?",
    "How is liver cirrhosis diagnosed?",
    "What are the complications of liver cirrhosis?"
]

print("=" * 80)
print("Enhanced ChatBot Test with Improved Gemini Integration")
print("=" * 80)
print(f"Testing {len(test_queries)} queries with enhanced prompting...")
print("=" * 80)

try:
    chatbot = LiverCirrhosisChatBot(
        gemini_api_key=os.environ['GEMINI_API_KEY'],
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}")
        print("Processing with Gemini and RAG...")
        
        try:
            response = chatbot.chat(query, use_rag=True, use_kg=False, max_evidence=5)
            
            print(f"\nANSWER:")
            print("-" * 80)
            print(response['answer'])
            print("-" * 80)
            print(f"\nMETADATA:")
            print(f"  - Evidence Items: {response.get('evidence_count', 0)}")
            print(f"  - Answer Length: {len(response['answer'])} characters")
            if response.get('sources'):
                print(f"  - Sources: {', '.join(response['sources'][:5])}")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n")
    
    print("=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
    
    chatbot.close()
    
except Exception as e:
    print(f"\nFatal Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)





