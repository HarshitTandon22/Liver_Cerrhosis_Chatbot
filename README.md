# Liver Cirrhosis Research ChatBot - Usage Guide

## Quick Start

To start the interactive chatbot, run:

```bash
python run_chatbot_final.py
```

Or on Windows:
```bash
START_CHATBOT.bat
```

## Features

✅ **Real-time Interactive Chat** - Ask questions and get immediate answers
✅ **Evidence-Based Responses** - All answers cite sources from research papers
✅ **RAG-Powered** - Retrieves relevant information from 2,710 paper chunks, 9,713 triples, and 1,958 claims
✅ **Gemini Integration** - Uses Google's Gemini API for natural language generation
✅ **Error Handling** - Gracefully handles errors and continues operating

## How to Use

1. **Start the chatbot** by running `python run_chatbot_final.py`

2. **Type your questions** about liver cirrhosis when prompted:
   - "What is liver cirrhosis?"
   - "What are the symptoms of liver cirrhosis?"
   - "What causes liver cirrhosis?"
   - "What treatments are available?"
   - Any other question related to liver cirrhosis research

3. **Wait for the answer** - The chatbot will:
   - Search through research papers
   - Retrieve relevant evidence
   - Generate a comprehensive answer using Gemini
   - Display sources and evidence count

4. **Continue chatting** - Ask more questions or type `quit`/`exit` to end

## Example Session

```
You: What are the main symptoms of liver cirrhosis?

[Processing your query...]

ChatBot:
======================================================================
Liver cirrhosis can present with various symptoms including...
[Detailed answer with citations]
======================================================================

Sources: paper1_extracted, paper2_extracted
Evidence Retrieved: 5 items from research papers

You: What causes it?

[Processing your query...]
...
```

## Commands

- Type `quit`, `exit`, `q`, or `bye` to end the conversation
- Press `Ctrl+C` to interrupt and exit
- Empty queries will be ignored (prompt to enter a question)

## Troubleshooting

### If you get EOF errors:
- Make sure you're running in an interactive terminal (not a script)
- Try running directly: `python run_chatbot_final.py`
- Check that your terminal supports interactive input

### If the chatbot doesn't respond:
- Check that Gemini API key is set correctly
- Verify Pinecone connection is working
- Check logs for error messages

### If answers are slow:
- This is normal - the chatbot searches through thousands of documents
- First query may take longer (model loading)
- Subsequent queries should be faster

## System Requirements

- Python 3.7+
- Internet connection (for Gemini API and Pinecone)
- Valid API keys:
  - Gemini API Key
  - Pinecone API Key

## Data Status

- **Pinecone**: ✅ Loaded (2,710 papers, 9,713 triples, 1,958 claims)
- **Neo4j**: ⚠️ Optional (system works without it)

Enjoy using the Liver Cirrhosis Research ChatBot!




