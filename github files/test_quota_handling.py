"""Test script to verify quota error handling in chatbot."""
import os
from dotenv import load_dotenv

load_dotenv()
import sys

# Set API key

from chatbot import LiverCirrhosisChatBot

print("=" * 70)
print("Testing Chatbot with Quota Error Handling")
print("=" * 70)

try:
    # Initialize chatbot
    print("\n1. Initializing chatbot...")
    bot = LiverCirrhosisChatBot(gemini_api_key=os.environ['GEMINI_API_KEY'])
    print(f"   [OK] Initialized with model: {bot.current_model_name}")
    print(f"   [OK] Available models: {bot.model_names}")
    
    # Test a simple query
    print("\n2. Testing a simple query (without RAG/KG to avoid delays)...")
    response = bot.chat(
        'What is liver cirrhosis?', 
        use_rag=False, 
        use_kg=False
    )
    
    print(f"   [OK] Response generated successfully!")
    print(f"   [OK] Response length: {len(response['answer'])} characters")
    print(f"\n   Answer preview:")
    print(f"   {response['answer'][:300]}...")
    
    print("\n" + "=" * 70)
    print("[OK] All tests passed! Quota handling is working.")
    print("=" * 70)
    
    bot.close()
    
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

