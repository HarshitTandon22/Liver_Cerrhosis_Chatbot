#!/usr/bin/env python3
"""
Enhanced Pinecone Manager with Namespaces
Manages multiple namespaces: papers, triples, claims
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPineconeManager:
    """Enhanced Pinecone manager with multiple namespaces."""
    
    NAMESPACES = {
        'papers': 'papers',
        'triples': 'triples',
        'claims': 'claims'
    }
    
    def __init__(self, 
                 index_name: str = "liver-cirrhosis-kg",
                 dimension: int = 384,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize enhanced Pinecone manager.
        
        Args:
            index_name: Pinecone index name
            dimension: Embedding dimension
            model_name: Sentence transformer model name
        """
        load_dotenv()
        
        self.api_key = os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            logger.warning("PINECONE_API_KEY not found in .env file. Pinecone features will be disabled.")
            self.api_key = None
            self.pc = None
            self.embedding_model = None
            return
        
        self.index_name = index_name
        self.dimension = dimension
        self.model_name = model_name
        
        # Initialize Pinecone
        if self.api_key:
            self.pc = Pinecone(api_key=self.api_key)
            
            # Load embedding model
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
        else:
            self.pc = None
            self.embedding_model = None
        
        logger.info(f"Enhanced Pinecone Manager initialized")
        logger.info(f"  - Index: {index_name}")
        logger.info(f"  - Dimension: {dimension}")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Namespaces: {list(self.NAMESPACES.values())}")
    
    def create_index(self, recreate: bool = False):
        """Create or recreate Pinecone index."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                if recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.pc.delete_index(self.index_name)
                    import time
                    time.sleep(10)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    return
            
            logger.info(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            logger.info(f"Index {self.index_name} created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def upload_papers(self, chunked_data_dir: str = "chunked_data", 
                     embeddings_dir: str = "embeddings"):
        """Upload papers to Pinecone papers namespace."""
        chunked_dir = Path(chunked_data_dir)
        embeddings_dir = Path(embeddings_dir)
        
        # Load chunks
        all_chunks_file = chunked_dir / "all_chunks.json"
        if not all_chunks_file.exists():
            logger.error(f"Chunks file not found: {all_chunks_file}")
            return
        
        with open(all_chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Load embeddings
        embeddings_file = embeddings_dir / "embeddings.npy"
        metadata_file = embeddings_dir / "metadata.json"
        
        if not embeddings_file.exists() or not metadata_file.exists():
            logger.error("Embeddings files not found")
            return
        
        embeddings = np.load(embeddings_file)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        index = self.pc.Index(self.index_name)
        namespace = self.NAMESPACES['papers']
        
        logger.info(f"Uploading {len(chunks)} paper chunks to namespace '{namespace}'...")
        
        vectors = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
            vector_data = {
                'id': meta['chunk_id'],
                'values': embeddings[i].tolist(),
                'metadata': {
                    'document_id': meta['document_id'],
                    'chunk_index': meta['chunk_index'],
                    'content': chunk.get('content', '')[:500],  # Truncate for metadata
                    'original_filename': meta['original_filename'],
                    'year': self._extract_year(meta['original_filename']) or 0,  # Use 0 if None
                    'upload_timestamp': datetime.now().isoformat()
                }
            }
            vectors.append(vector_data)
            
            # Upload in batches
            if len(vectors) >= 100:
                index.upsert(vectors=vectors, namespace=namespace)
                vectors = []
                logger.info(f"Uploaded batch: {i+1}/{len(chunks)}")
        
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
        
        logger.info(f"Uploaded {len(chunks)} paper chunks to '{namespace}' namespace")
    
    def upload_triples(self, kg_data_dir: str = "kg_data"):
        """Upload knowledge graph triples to Pinecone triples namespace."""
        kg_dir = Path(kg_data_dir)
        relations_file = kg_dir / "all_relations.json"
        
        if not relations_file.exists():
            logger.error(f"Relations file not found: {relations_file}")
            return
        
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        
        index = self.pc.Index(self.index_name)
        namespace = self.NAMESPACES['triples']
        
        logger.info(f"Uploading {len(relations)} triples to namespace '{namespace}'...")
        
        vectors = []
        for i, relation in enumerate(relations):
            # Create verbalization of triple
            subject = relation.get('subject', {}).get('text', '')
            predicate = relation.get('predicate', '')
            obj = relation.get('object', {}).get('text', '')
            
            triple_text = f"{subject} {predicate} {obj}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(triple_text, normalize_embeddings=True)
            
            vector_data = {
                'id': f"triple_{i}",
                'values': embedding.tolist(),
                'metadata': {
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj,
                    'subject_normalized': relation.get('subject', {}).get('normalized', ''),
                    'object_normalized': relation.get('object', {}).get('normalized', ''),
                    'relation_type': predicate,
                    'confidence': relation.get('confidence', 0.5),
                    'source': relation.get('source', ''),
                    'triple_text': triple_text,
                    'upload_timestamp': datetime.now().isoformat()
                }
            }
            vectors.append(vector_data)
            
            # Upload in batches
            if len(vectors) >= 100:
                index.upsert(vectors=vectors, namespace=namespace)
                vectors = []
                logger.info(f"Uploaded batch: {i+1}/{len(relations)}")
        
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
        
        logger.info(f"Uploaded {len(relations)} triples to '{namespace}' namespace")
    
    def upload_claims(self, extracted_data_dir: str = "extracted_data"):
        """Upload claims to Pinecone claims namespace."""
        extracted_dir = Path(extracted_data_dir)
        extracted_files = list(extracted_dir.glob("*_extracted.json"))
        
        index = self.pc.Index(self.index_name)
        namespace = self.NAMESPACES['claims']
        
        logger.info(f"Uploading claims from {len(extracted_files)} documents to namespace '{namespace}'...")
        
        all_claims = []
        
        for file_path in extracted_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Extract claims from different sections
            for section in ['objective', 'result', 'comparison', 'limitation']:
                section_data = doc_data.get(section, [])
                if isinstance(section_data, list):
                    for claim_text in section_data:
                        if claim_text and len(claim_text.strip()) > 50:
                            all_claims.append({
                                'text': claim_text,
                                'section': section,
                                'source': file_path.stem,
                                'year': self._extract_year(file_path.stem)
                            })
        
        logger.info(f"Found {len(all_claims)} claims")
        
        vectors = []
        for i, claim in enumerate(all_claims):
            # Generate embedding
            embedding = self.embedding_model.encode(claim['text'], normalize_embeddings=True)
            
            vector_data = {
                'id': f"claim_{i}_{claim['source']}",
                'values': embedding.tolist(),
                'metadata': {
                    'text': claim['text'][:500],  # Truncate for metadata
                    'section': claim['section'],
                    'source': claim['source'],
                    'year': claim.get('year') or 0,  # Use 0 if None
                    'study_type': self._infer_study_type(claim['text']) or 'OTHER',
                    'upload_timestamp': datetime.now().isoformat()
                }
            }
            vectors.append(vector_data)
            
            # Upload in batches
            if len(vectors) >= 100:
                index.upsert(vectors=vectors, namespace=namespace)
                vectors = []
                logger.info(f"Uploaded batch: {i+1}/{len(all_claims)}")
        
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
        
        logger.info(f"Uploaded {len(all_claims)} claims to '{namespace}' namespace")
    
    def search(self, query: str, namespace: str, top_k: int = 10,
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search in a specific namespace.
        
        Args:
            query: Search query text
            namespace: Namespace to search (papers, triples, claims)
            top_k: Number of results
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results
        """
        if not self.api_key or not self.pc:
            logger.warning("Pinecone not initialized. Returning empty results.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        
        index = self.pc.Index(self.index_name)
        ns = self.NAMESPACES.get(namespace, namespace)
        
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=ns,
            include_metadata=True,
            filter=filter_dict
        )
        
        return results['matches']
    
    def _extract_year(self, filename: str) -> Optional[int]:
        """Extract year from filename."""
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return int(year_match.group(0))
        return None
    
    def _infer_study_type(self, text: str) -> str:
        """Infer study type from text."""
        text_lower = text.lower()
        if 'randomized' in text_lower or 'rct' in text_lower:
            return 'RCT'
        elif 'cohort' in text_lower:
            return 'COHORT'
        elif 'case-control' in text_lower or 'case control' in text_lower:
            return 'CASE_CONTROL'
        elif 'review' in text_lower or 'meta-analysis' in text_lower:
            return 'REVIEW'
        else:
            return 'OTHER'

def main():
    """Main function."""
    print("Enhanced Pinecone Manager")
    print("=" * 50)
    
    manager = EnhancedPineconeManager()
    
    # Create index
    manager.create_index(recreate=False)
    
    # Upload data to namespaces
    print("\nUploading papers...")
    manager.upload_papers()
    
    print("\nUploading triples...")
    manager.upload_triples()
    
    print("\nUploading claims...")
    manager.upload_claims()
    
    print("\nUpload complete!")

if __name__ == "__main__":
    main()

