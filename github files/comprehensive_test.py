#!/usr/bin/env python3
"""
Comprehensive ChatBot Test - Verify All Answers Are Generated Correctly
"""

import os
from dotenv import load_dotenv

load_dotenv()
import sys
from chatbot import LiverCirrhosisChatBot

# Set API keys

# Diverse test queries covering different aspects
test_queries = [
    "What is liver cirrhosis?",
    "What are the early signs of liver cirrhosis?",
    "Which medications are used to treat liver cirrhosis?",
    "What diagnostic tests are performed for liver cirrhosis?",
    "What is the relationship between alcohol and liver cirrhosis?",
    "Can liver cirrhosis lead to liver cancer?",
    "What is hepatic encephalopathy in cirrhosis?",
    "How does portal hypertension develop in cirrhosis?",
    "What is ascites in liver cirrhosis?",
    "What are the stages of liver cirrhosis?",
    "What is the MELD score used for in cirrhosis?",
    "What role does diet play in managing cirrhosis?",
    "How does cirrhosis affect kidney function?",
    "What is variceal bleeding in cirrhosis?",
    "What are the treatment options for decompensated cirrhosis?"
]

print("=" * 80)
print("COMPREHENSIVE CHATBOT TEST")
print("=" * 80)
print(f"Testing {len(test_queries)} diverse queries")
print("Verifying all answers are generated correctly with no errors")
print("=" * 80)

errors = []
successful = []

try:
    chatbot = LiverCirrhosisChatBot(
        gemini_api_key=os.environ['GEMINI_API_KEY'],
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}")
        
        try:
            print("Processing...")
            response = chatbot.chat(query, use_rag=True, use_kg=False, max_evidence=5)
            
            # Verify response structure
            if not response:
                errors.append(f"Query {i}: Empty response")
                print("[ERROR] Empty response")
                continue
            
            if 'answer' not in response:
                errors.append(f"Query {i}: Missing 'answer' field")
                print("[ERROR] Missing 'answer' field")
                continue
            
            answer = response['answer']
            
            if not answer or len(answer.strip()) == 0:
                errors.append(f"Query {i}: Empty answer text")
                print("[ERROR] Answer text is empty")
                continue
            
            # Check answer quality
            answer_length = len(answer)
            evidence_count = response.get('evidence_count', 0)
            sources_count = len(response.get('sources', []))
            
            print(f"\n[SUCCESS]")
            print(f"   Answer Length: {answer_length} characters")
            print(f"   Evidence Items: {evidence_count}")
            print(f"   Sources: {sources_count}")
            
            # Show first 200 chars of answer
            print(f"\n   Answer Preview:")
            print(f"   {answer[:200]}...")
            
            if response.get('sources'):
                print(f"   Sources: {', '.join(response['sources'][:3])}")
            
            successful.append({
                'query': query,
                'answer_length': answer_length,
                'evidence_count': evidence_count,
                'sources_count': sources_count
            })
            
        except Exception as e:
            error_msg = f"Query {i} ({query[:50]}...): {str(e)}"
            errors.append(error_msg)
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Final Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Queries: {len(test_queries)}")
    print(f"[SUCCESS] Successful: {len(successful)}")
    print(f"[ERROR] Errors: {len(errors)}")
    
    if successful:
        avg_length = sum(s['answer_length'] for s in successful) / len(successful)
        avg_evidence = sum(s['evidence_count'] for s in successful) / len(successful)
        avg_sources = sum(s['sources_count'] for s in successful) / len(successful)
        
        print(f"\nAverage Answer Length: {avg_length:.0f} characters")
        print(f"Average Evidence Items: {avg_evidence:.1f}")
        print(f"Average Sources: {avg_sources:.1f}")
    
    if errors:
        print(f"\n[ERROR] ERRORS ENCOUNTERED:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("\n[SUCCESS] ALL TESTS PASSED - NO ERRORS!")
    
    print("=" * 80)
    
    chatbot.close()
    
    # Exit with error code if there were errors
    if errors:
        sys.exit(1)
    
except Exception as e:
    print(f"\n[FATAL ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

