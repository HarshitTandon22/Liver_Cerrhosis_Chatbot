# Quick Start Guide

## System Overview

This system implements a complete knowledge graph-based chatbot for liver cirrhosis research papers, following the architecture diagram you provided.

## What Was Built

✅ **NER/RE Extraction** (`ner_re_extractor.py`)
- Extracts entities (diseases, treatments, symptoms) from papers
- Extracts relations (TREATS, CAUSES, etc.)
- Normalizes entities and tracks provenance

✅ **Liver Ontology** (`liver_ontology.py`)
- Defines entity and relation types
- SHACL validation constraints

✅ **Neo4j Knowledge Graph** (`neo4j_kg_store.py`)
- Stores entities and relations with provenance
- Provides search, relation finding, path explanation

✅ **Enhanced Pinecone** (`pinecone_enhanced.py`)
- Three namespaces: papers, triples, claims
- Embeddings using all-MiniLM-L6-v2 (384-d, cosine)

✅ **RAG Orchestrator** (`rag_orchestrator.py`)
- Hybrid retrieval from multiple namespaces
- Cross-encoder reranking
- Evidence pack generation
- Claim-checking

✅ **MCP Server** (`mcp_server.py`)
- `kg.search_graph` - Search entities
- `kg.find_relations` - Find relations
- `kg.path_explain` - Explain paths
- `rag.answer_with_evidence` - Get answers with evidence
- `audit.provenance` - Audit provenance

✅ **ChatBot** (`chatbot.py`)
- Uses Gemini for answer generation
- Integrates KG + RAG via MCP server
- Provides precise, evidence-based answers only

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment
Create `.env` file:
```env
PINECONE_API_KEY=your_key
GEMINI_API_KEY=your_key
```

### 3. Start Neo4j
```bash
# Using Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 4. Run Setup
```bash
python setup_kg_system.py
```

This will:
1. Extract entities and relations from your existing data
2. Create Neo4j knowledge graph
3. Upload to Pinecone with three namespaces

### 5. Run ChatBot
```bash
python chatbot.py
```

## File Structure

```
├── ner_re_extractor.py          # NER/RE extraction
├── liver_ontology.py           # Ontology schema
├── neo4j_kg_store.py            # Neo4j KG store
├── pinecone_enhanced.py         # Enhanced Pinecone with namespaces
├── rag_orchestrator.py          # RAG orchestrator
├── mcp_server.py                # MCP server
├── chatbot.py                   # ChatBot
├── setup_kg_system.py           # Setup script
├── requirements.txt             # Dependencies
├── KG_SYSTEM_README.md          # Full documentation
└── QUICK_START.md               # This file
```

## Key Features

1. **Precise Answers**: ChatBot only provides information from the knowledge base
2. **Provenance Tracking**: All entities and relations include source tracking
3. **Hybrid Retrieval**: Combines KG search and vector search
4. **Evidence-Based**: All answers include evidence sources
5. **Multiple Namespaces**: Papers, triples, and claims in separate Pinecone namespaces

## Usage Examples

### ChatBot
```bash
python chatbot.py
# Ask: "What is portal vein thrombosis?"
# Ask: "What treatments are available for cirrhosis?"
```

### MCP Server
```python
from mcp_server import MCPServer

server = MCPServer()
results = server.kg_search_graph("cirrhosis")
answer = server.rag_answer_with_evidence("What is cirrhosis?")
```

## Configuration Notes

- **Neo4j**: Default credentials are `neo4j/password` - update in scripts if needed
- **Pinecone**: Uses existing index or creates new one
- **Gemini**: Requires API key in environment or passed to ChatBot

## Next Steps

1. Run the setup script to create the knowledge graph
2. Test the MCP server functions
3. Try the interactive chatbot
4. Customize the ontology if needed
5. Fine-tune retrieval parameters

## Troubleshooting

- **Neo4j errors**: Check if Neo4j is running on port 7687
- **Pinecone errors**: Verify API key in `.env`
- **Gemini errors**: Check API key is set
- **Import errors**: Install all dependencies from requirements.txt

For detailed documentation, see `KG_SYSTEM_README.md`.





