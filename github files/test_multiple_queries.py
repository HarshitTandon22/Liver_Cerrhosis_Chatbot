#!/usr/bin/env python3
"""
Test ChatBot with multiple diverse queries
"""

import os
from dotenv import load_dotenv

load_dotenv()
import sys
from chatbot import LiverCirrhosisChatBot

# Set API keys

# Base test queries (will be repeated to reach 100 total queries)
base_queries = [
    "What are the main symptoms of liver cirrhosis?",
    "What causes liver cirrhosis?",
    "What treatments are available for liver cirrhosis?",
    "How is liver cirrhosis diagnosed?",
    "What are the complications of liver cirrhosis?",
    "What is the relationship between hepatitis and liver cirrhosis?",
    "What are the risk factors for developing liver cirrhosis?",
    "Can liver cirrhosis be reversed?",
    "What lifestyle changes help with liver cirrhosis?",
    "What is the prognosis for liver cirrhosis patients?"
]

TARGET_QUERY_COUNT = 100

# Repeat base queries to reach ~100 total (exactly 100 by truncation)
test_queries = []
while len(test_queries) < TARGET_QUERY_COUNT:
    for q in base_queries:
        if len(test_queries) >= TARGET_QUERY_COUNT:
            break
        # Tag each query with an index so logs are easy to inspect
        test_queries.append(q)

print("=" * 70)
print("ChatBot Multi-Query Test")
print("=" * 70)
print(f"Testing {len(test_queries)} queries (target={TARGET_QUERY_COUNT})...")
print("=" * 70)

try:
    chatbot = LiverCirrhosisChatBot(
        gemini_api_key=os.environ['GEMINI_API_KEY'],
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print(f"{'='*70}")
        print("Processing...")
        
        try:
            response = chatbot.chat(query, use_rag=True, use_kg=False)
            
            print(f"\nANSWER:")
            print("-" * 70)
            print(response['answer'])
            print("-" * 70)
            print(f"Evidence Count: {response.get('evidence_count', 0)}")
            
            if response.get('sources'):
                print(f"Sources: {', '.join(response['sources'][:3])}")
            
            results.append({
                'query': query,
                'success': True,
                'evidence_count': response.get('evidence_count', 0),
                'answer_length': len(response['answer']),
                'has_sources': len(response.get('sources', [])) > 0
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'query': query,
                'success': False,
                'error': str(e)
            })
        
        print("\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r.get('success', False))
    print(f"Successful queries: {successful}/{len(test_queries)}")
    print(f"Failed queries: {len(test_queries) - successful}/{len(test_queries)}")
    
    if successful > 0:
        avg_evidence = sum(r.get('evidence_count', 0) for r in results if r.get('success')) / successful
        print(f"Average evidence items per query: {avg_evidence:.1f}")
        print(f"Queries with sources: {sum(1 for r in results if r.get('has_sources', False))}/{successful}")
    
    print("=" * 70)
    
    chatbot.close()
    
except Exception as e:
    print(f"\nFatal Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)





