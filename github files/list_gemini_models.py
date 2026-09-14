#!/usr/bin/env python3
"""List available Gemini models"""
import os
from dotenv import load_dotenv

load_dotenv()
from google import generativeai as genai

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

print("Listing available Gemini models...")
models = genai.list_models()
for model in models:
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")





