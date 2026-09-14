#!/usr/bin/env python3
"""
Pinecone Search Interface
Search and query the liver cirrhosis embeddings in Pinecone.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PineconeSearcher:
    """Search interface for Pinecone vector database."""
    
    def __init__(self, 
                 pinecone_data_dir: str = "pinecone_data",
                 embeddings_dir: str = "embeddings",
                 config_file: str = ".env"):
        """
        Initialize Pinecone searcher.
        
        Args:
            pinecone_data_dir: Directory containing Pinecone data
            embeddings_dir: Directory containing original embeddings
            config_file: Path to configuration file
        """
        self.pinecone_data_dir = Path(pinecone_data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        
        # Load configuration
        load_dotenv(config_file)
        
        # Configuration
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('INDEX_NAME', 'liver-cirrhosis-embeddings')
        
        if not self.api_key:
            logger.error("PINECONE_API_KEY not found. Please set it in your .env file.")
            raise ValueError("PINECONE_API_KEY is required")
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise
        
        # Load embedding model for text queries
        try:
            self.model = SentenceTransformer('all-MiniLM-L12-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Load chunk data for displaying results
        self.chunk_data = self._load_chunk_data()
    
    def _load_chunk_data(self) -> Dict[str, Dict[str, Any]]:
        """Load chunk data for displaying search results."""
        try:
            all_chunks_file = self.embeddings_dir / "all_chunks.json"
            if all_chunks_file.exists():
                with open(all_chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                # Create lookup dictionary
                chunk_lookup = {}
                for chunk in chunks:
                    chunk_lookup[chunk['chunk_id']] = chunk
                
                logger.info(f"Loaded {len(chunk_lookup)} chunks for display")
                return chunk_lookup
            else:
                logger.warning("Chunk data file not found")
                return {}
        except Exception as e:
            logger.error(f"Failed to load chunk data: {e}")
            return {}
    
    def text_to_embedding(self, text: str) -> List[float]:
        """Convert text to embedding vector."""
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def search_by_text(self, 
                      query_text: str, 
                      top_k: int = 10, 
                      filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using text query.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Convert text to embedding
            query_vector = self.text_to_embedding(query_text)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Enhance results with chunk content
            enhanced_results = []
            for match in results['matches']:
                chunk_id = match['id']
                score = match['score']
                metadata = match['metadata']
                
                # Get chunk content if available
                chunk_content = ""
                if chunk_id in self.chunk_data:
                    chunk_content = self.chunk_data[chunk_id].get('content', '')
                
                enhanced_result = {
                    'chunk_id': chunk_id,
                    'score': score,
                    'metadata': metadata,
                    'content': chunk_content,
                    'document_id': metadata.get('document_id', 'Unknown'),
                    'chunk_index': metadata.get('chunk_index', 0)
                }
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to search by text: {e}")
            return []
    
    def search_by_document(self, 
                          document_id: str, 
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for chunks from a specific document.
        
        Args:
            document_id: Document ID to search within
            top_k: Number of results to return
            
        Returns:
            List of chunks from the specified document
        """
        try:
            # Search with document filter
            filter_dict = {'document_id': document_id}
            
            # Get a random vector for the query (we just need the filter)
            dummy_vector = [0.0] * 384  # 384 is the embedding dimension
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Enhance results
            enhanced_results = []
            for match in results['matches']:
                chunk_id = match['id']
                score = match['score']
                metadata = match['metadata']
                
                chunk_content = ""
                if chunk_id in self.chunk_data:
                    chunk_content = self.chunk_data[chunk_id].get('content', '')
                
                enhanced_result = {
                    'chunk_id': chunk_id,
                    'score': score,
                    'metadata': metadata,
                    'content': chunk_content,
                    'document_id': metadata.get('document_id', 'Unknown'),
                    'chunk_index': metadata.get('chunk_index', 0)
                }
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to search by document: {e}")
            return []
    
    def get_document_list(self) -> List[str]:
        """Get list of all documents in the index."""
        try:
            # Get index statistics
            stats = self.index.describe_index_stats()
            
            # Extract document IDs from namespaces
            documents = set()
            if 'namespaces' in stats:
                for namespace, ns_stats in stats['namespaces'].items():
                    # Document IDs are in the metadata, we need to query to get them
                    pass
            
            # Alternative: query with a dummy vector to get some results and extract document IDs
            dummy_vector = [0.0] * 384
            results = self.index.query(
                vector=dummy_vector,
                top_k=1000,  # Get as many as possible
                include_metadata=True
            )
            
            documents = set()
            for match in results['matches']:
                if 'document_id' in match['metadata']:
                    documents.add(match['metadata']['document_id'])
            
            return sorted(list(documents))
            
        except Exception as e:
            logger.error(f"Failed to get document list: {e}")
            return []
    
    def print_search_results(self, results: List[Dict[str, Any]], query: str = ""):
        """Print search results in a formatted way."""
        if not results:
            print("No results found.")
            return
        
        print(f"\n{'='*80}")
        if query:
            print(f"SEARCH RESULTS FOR: '{query}'")
        else:
            print("SEARCH RESULTS")
        print(f"{'='*80}")
        print(f"Found {len(results)} results\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Chunk ID: {result['chunk_id']}")
            print(f"   Document: {result['document_id']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Chunk Index: {result['chunk_index']}")
            
            # Show content preview
            content = result['content']
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   Content: {preview}")
            else:
                print(f"   Content: [Content not available]")
            
            print(f"   Metadata: {result['metadata']}")
            print("-" * 80)
    
    def interactive_search(self):
        """Interactive search interface."""
        print("\n" + "="*60)
        print("PINECONE SEARCH INTERFACE")
        print("="*60)
        print("Commands:")
        print("  search <query> - Search by text")
        print("  doc <document_id> - Search within document")
        print("  list - List all documents")
        print("  stats - Show index statistics")
        print("  quit - Exit")
        print("="*60)
        
        while True:
            try:
                command = input("\nEnter command: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'list':
                    documents = self.get_document_list()
                    print(f"\nFound {len(documents)} documents:")
                    for i, doc in enumerate(documents[:20], 1):  # Show first 20
                        print(f"  {i}. {doc}")
                    if len(documents) > 20:
                        print(f"  ... and {len(documents) - 20} more")
                
                elif command.lower() == 'stats':
                    stats = self.index.describe_index_stats()
                    print(f"\nIndex Statistics:")
                    print(f"  Total vectors: {stats.get('total_vector_count', 'Unknown')}")
                    print(f"  Dimension: {stats.get('dimension', 'Unknown')}")
                    print(f"  Index fullness: {stats.get('index_fullness', 'Unknown')}")
                
                elif command.startswith('search '):
                    query = command[7:].strip()
                    if query:
                        results = self.search_by_text(query, top_k=5)
                        self.print_search_results(results, query)
                    else:
                        print("Please provide a search query")
                
                elif command.startswith('doc '):
                    doc_id = command[4:].strip()
                    if doc_id:
                        results = self.search_by_document(doc_id, top_k=5)
                        self.print_search_results(results, f"Document: {doc_id}")
                    else:
                        print("Please provide a document ID")
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")

def main():
    """Main function."""
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("❌ .env file not found!")
            print("Please create a .env file with your Pinecone API key.")
            print("You can use pinecone_config.example as a template.")
            return
        
        # Create searcher
        searcher = PineconeSearcher()
        
        # Start interactive search
        searcher.interactive_search()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

