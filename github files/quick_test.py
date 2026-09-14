#!/usr/bin/env python3
"""Quick test of chatbot with Gemini"""

import os
from dotenv import load_dotenv

load_dotenv()
from chatbot import LiverCirrhosisChatBot


print("=" * 70)
print("Quick ChatBot Test")
print("=" * 70)

chatbot = LiverCirrhosisChatBot()
response = chatbot.chat('What are the main causes and symptoms of liver cirrhosis?', use_rag=True, use_kg=False)

print("\n=== ANSWER ===")
print(response['answer'])
print(f"\nEvidence: {response.get('evidence_count', 0)} items")
print(f"Sources: {len(response.get('sources', []))} sources")
if response.get('sources'):
    print(f"Source names: {', '.join(response['sources'][:3])}")

chatbot.close()
print("\nTest completed successfully!")





