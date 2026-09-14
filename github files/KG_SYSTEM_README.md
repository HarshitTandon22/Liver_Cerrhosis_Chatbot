# Liver Cirrhosis Knowledge Graph System

A comprehensive knowledge graph-based chatbot system for liver cirrhosis research papers, built according to the specified architecture.

## Architecture Overview

```
[Research Articles] → [Ingestion & Curation] → [NER/RE + Normalization]
          │                           │                │
          │                           └─► Provenance tagging (DOI/PMID, span, model, conf)
          ▼
 [KG Store: Neo4j/RDF + SHACL]  ←── [Liver Ontology/Schema]
          │
          ├─ Triple/Text Verbalizations
          ├─ Embeddings (all-MiniLM-L6-v2, 384-d, L2-norm)
          └─► [Pinecone] Indexes (cosine)
                  • namespaces: papers | triples | claims
                  • metadata filters (year, entities, rel_type, study_type)
                          │
                          ▼
                 [RAG Orchestrator]
          (Hybrid retrieval, rerank, evidence pack, claim-check vs KG)
                          │
                          ▼
                ┌───────────────────────────┐
                │         MCP Server        │
                │  kg.search_graph          │
                │  kg.find_relations        │
                │  kg.path_explain          │
                │  rag.answer_with_evidence │
                │  audit.provenance         │
                └───────────────────────────┘
                          │
                          ▼
                      [Gemini]
           (Answer drafting + constrained prompting)
                          │
                          ▼
                  Answer + Evidence + Limits
```

## Components

### 1. NER/RE Extraction (`ner_re_extractor.py`)
- Extracts entities (diseases, treatments, symptoms, etc.) from research papers
- Extracts relations (TREATS, CAUSES, ASSOCIATED_WITH, etc.)
- Normalizes entities to standard terminology
- Tags provenance (DOI/PMID, span, model, confidence)

### 2. Liver Ontology (`liver_ontology.py`)
- Defines entity types: DISEASE, TREATMENT, SYMPTOM, ORGAN, PROCEDURE, MEDICATION, STUDY, CLAIM
- Defines relation types: TREATS, CAUSES, ASSOCIATED_WITH, HAS_SYMPTOM, etc.
- SHACL constraints for validation

### 3. Neo4j Knowledge Graph Store (`neo4j_kg_store.py`)
- Stores entities and relations in Neo4j
- Provides search, relation finding, and path explanation functions
- Validates against ontology schema

### 4. Enhanced Pinecone Manager (`pinecone_enhanced.py`)
- Manages three namespaces:
  - `papers`: Research paper chunks
  - `triples`: Knowledge graph triple verbalizations
  - `claims`: Extracted claims from papers
- Uses all-MiniLM-L6-v2 model (384 dimensions, cosine similarity)
- Supports metadata filtering

### 5. RAG Orchestrator (`rag_orchestrator.py`)
- Hybrid retrieval from multiple Pinecone namespaces
- Cross-encoder reranking
- Evidence pack generation
- Claim-checking against KG

### 6. MCP Server (`mcp_server.py`)
Provides five main functions:
- `kg.search_graph`: Search entities in KG
- `kg.find_relations`: Find relations for entities
- `kg.path_explain`: Explain paths between entities
- `rag.answer_with_evidence`: Generate answers with evidence
- `audit.provenance`: Audit provenance information

### 7. ChatBot (`chatbot.py`)
- Uses Gemini for answer generation
- Integrates with KG and RAG via MCP server
- Provides precise, evidence-based answers
- Only responds based on available data

## Installation

### Prerequisites

1. **Neo4j Database**
   ```bash
   # Install Neo4j (or use Docker)
   docker run -p 7474:7474 -p 7687:7687 neo4j:latest
   # Default credentials: neo4j/password
   ```

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **spaCy Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Environment Variables**
   Create a `.env` file:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Setup Steps

1. **Extract entities and relations from papers**
   ```bash
   python ner_re_extractor.py
   ```
   This creates `kg_data/` with:
   - `all_entities.json`
   - `all_relations.json`
   - `document_extractions.json`

2. **Create Neo4j Knowledge Graph**
   ```bash
   python neo4j_kg_store.py
   ```
   Make sure Neo4j is running and update credentials in the script.

3. **Setup Pinecone with namespaces**
   ```bash
   python pinecone_enhanced.py
   ```
   This uploads data to three namespaces: papers, triples, claims

4. **Run complete setup (recommended)**
   ```bash
   python setup_kg_system.py
   ```

## Usage

### ChatBot

Run the interactive chatbot:
```bash
python chatbot.py
```

The chatbot will:
- Use KG to find relevant entities and relations
- Use RAG to retrieve relevant evidence
- Generate precise answers using Gemini
- Only provide information based on available data

Example queries:
- "What is portal vein thrombosis?"
- "What treatments are available for liver cirrhosis?"
- "How is cirrhosis diagnosed?"

### MCP Server Functions

Test MCP server functions:
```bash
python mcp_server.py
```

Or use in code:
```python
from mcp_server import MCPServer

server = MCPServer()

# Search KG
results = server.kg_search_graph("cirrhosis", limit=10)

# Find relations
relations = server.kg_find_relations("cirrhosis", limit=10)

# Explain paths
paths = server.kg_path_explain("cirrhosis", "portal vein thrombosis")

# Get answer with evidence
answer = server.rag_answer_with_evidence("What is cirrhosis?", top_k=5)

# Audit provenance
provenance = server.audit_provenance(entity_name="cirrhosis")
```

### RAG Orchestrator

Use RAG orchestrator directly:
```python
from rag_orchestrator import RAGOrchestrator
from pinecone_enhanced import EnhancedPineconeManager
from neo4j_kg_store import Neo4jKGStore

pinecone_manager = EnhancedPineconeManager()
neo4j_store = Neo4jKGStore()
rag = RAGOrchestrator(pinecone_manager, neo4j_store)

# Create evidence pack
evidence = rag.create_evidence_pack("What is portal vein thrombosis?", top_k=10)

# Check claim
claim_result = rag.claim_check("PVT is rare in cirrhosis", evidence)
```

## Configuration

### Neo4j Configuration
Update `neo4j_kg_store.py` and `mcp_server.py`:
```python
Neo4jKGStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)
```

### Pinecone Configuration
Update `.env` file:
```env
PINECONE_API_KEY=your_key
```

Or update `pinecone_enhanced.py`:
```python
EnhancedPineconeManager(
    index_name="liver-cirrhosis-kg",
    dimension=384,
    model_name="all-MiniLM-L6-v2"
)
```

### Gemini Configuration
Set environment variable:
```bash
export GEMINI_API_KEY=your_key
```

Or update `chatbot.py`:
```python
chatbot = LiverCirrhosisChatBot(gemini_api_key="your_key")
```

## Data Flow

1. **Ingestion**: PDFs → Extracted JSON → Chunked data
2. **NER/RE**: Chunked data → Entities & Relations → KG data
3. **KG Storage**: KG data → Neo4j (with provenance)
4. **Vector Indexing**: 
   - Papers → Pinecone `papers` namespace
   - Triples → Pinecone `triples` namespace
   - Claims → Pinecone `claims` namespace
5. **Query Processing**:
   - User query → RAG retrieval → Evidence pack
   - User query → KG search → Entity/relation context
   - Combined → Gemini → Answer

## Output Structure

```
kg_data/
├── all_entities.json          # All extracted entities
├── all_relations.json          # All extracted relations
├── document_extractions.json   # Per-document extractions
└── extraction_statistics.json  # Extraction stats

chunked_data/                   # (Already exists)
embeddings/                     # (Already exists)
extracted_data/                 # (Already exists)
```

## Notes

- The system is designed to be **precise** and only provide information from the knowledge base
- All entities and relations include provenance tracking
- The chatbot uses constrained prompting to prevent hallucination
- KG validation ensures data quality through SHACL constraints

## Troubleshooting

1. **Neo4j connection errors**: Check if Neo4j is running and credentials are correct
2. **Pinecone errors**: Verify API key in `.env` file
3. **Gemini errors**: Check API key is set correctly
4. **spaCy errors**: Run `python -m spacy download en_core_web_sm`
5. **Import errors**: Install all dependencies: `pip install -r requirements.txt`

## License

This system is designed for research purposes related to liver cirrhosis medical literature.





