#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv()
from chatbot import LiverCirrhosisChatBot


def main():
    questions = [
        "What are the common complications of decompensated liver cirrhosis?",
        "How is spontaneous bacterial peritonitis diagnosed and managed?",
        "Does non-selective beta-blocker therapy reduce variceal bleeding risk?",
    ]
    bot = LiverCirrhosisChatBot(gemini_api_key=os.environ['GEMINI_API_KEY'])
    for i, q in enumerate(questions, 1):
        print("=" * 80)
        print(f"Q{i}: {q}")
        resp = bot.chat(q, use_rag=True, use_kg=False, max_evidence=5)
        print("-" * 80)
        print(resp.get('answer',''))
    bot.close()


if __name__ == '__main__':
    main()








