#!/usr/bin/env python3
"""
Test script for ChatBot
Tests the chatbot with a sample query.
"""

import os
from dotenv import load_dotenv

load_dotenv()
from chatbot import LiverCirrhosisChatBot

# Set Gemini API key

print("=" * 60)
print("Testing Liver Cirrhosis ChatBot")
print("=" * 60)

try:
    # Initialize chatbot
    print("\nInitializing chatbot...")
    try:
        chatbot = LiverCirrhosisChatBot(gemini_api_key=os.environ['GEMINI_API_KEY'])
        print("Chatbot initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not connect to Neo4j: {e}")
        print("Continuing with Gemini-only mode...")
        # Create a minimal chatbot that doesn't require Neo4j
        from google import generativeai as genai
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        model = genai.GenerativeModel('gemini-pro')
        
        # Simple test query
        test_query = "What is liver cirrhosis?"
        print(f"\nTest Query: {test_query}")
        print("-" * 60)
        
        prompt = f"""You are a medical research assistant. Answer this question about liver cirrhosis:
        
{test_query}

Provide a brief, accurate answer."""
        
        response = model.generate_content(prompt)
        print(f"\nBot Response:")
        print(response.text)
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        exit(0)
    
    # Test query
    test_query = "What is liver cirrhosis?"
    print(f"\nTest Query: {test_query}")
    print("-" * 60)
    
    # Get response
    response = chatbot.chat(test_query, use_kg=False, use_rag=False, max_evidence=3)
    
    print(f"\nBot Response:")
    print(response['answer'])
    
    if response.get('sources'):
        print(f"\nSources: {', '.join(response['sources'][:3])}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print("\nNote: To run the interactive chatbot, use: python chatbot.py")
    
    chatbot.close()
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

