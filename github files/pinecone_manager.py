#!/usr/bin/env python3
"""
Pinecone Vector Database Manager
Manages embeddings in Pinecone vector database for liver cirrhosis research papers.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PineconeManager:
    """Manage embeddings in Pinecone vector database."""
    
    def __init__(self, 
                 embeddings_dir: str = "embeddings",
                 pinecone_data_dir: str = "pinecone_data",
                 config_file: str = ".env"):
        """
        Initialize Pinecone manager.
        
        Args:
            embeddings_dir: Directory containing embeddings
            pinecone_data_dir: Directory for Pinecone-related data
            config_file: Path to configuration file
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.pinecone_data_dir = Path(pinecone_data_dir)
        self.pinecone_data_dir.mkdir(exist_ok=True)
        
        # Load configuration
        load_dotenv(config_file)
        
        # Configuration
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
        self.index_name = os.getenv('INDEX_NAME', 'liver-cirrhosis-embeddings')
        self.dimension = int(os.getenv('DIMENSION', 384))
        self.metric = os.getenv('METRIC', 'cosine')
        
        if not self.api_key:
            logger.error("PINECONE_API_KEY not found. Please set it in your .env file.")
            raise ValueError("PINECONE_API_KEY is required")
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info(f"Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
        
        # Statistics
        self.stats = {
            'index_name': self.index_name,
            'dimension': self.dimension,
            'metric': self.metric,
            'environment': self.environment,
            'created_date': datetime.now().isoformat(),
            'total_vectors': 0,
            'upload_success': 0,
            'upload_failed': 0,
            'processing_time': 0
        }
        
        logger.info(f"Pinecone Manager initialized:")
        logger.info(f"  - Index: {self.index_name}")
        logger.info(f"  - Dimension: {self.dimension}")
        logger.info(f"  - Metric: {self.metric}")
        logger.info(f"  - Environment: {self.environment}")
    
    def create_index(self, recreate: bool = False):
        """
        Create or recreate Pinecone index.
        
        Args:
            recreate: Whether to recreate the index if it exists
        """
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                if recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.pc.delete_index(self.index_name)
                    # Wait for deletion to complete
                    time.sleep(10)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    return
            
            # Create new index
            logger.info(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region=self.environment
                )
            )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(5)
            
            logger.info(f"Index {self.index_name} created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def load_embeddings_data(self):
        """Load embeddings and metadata."""
        # Load embeddings
        embeddings_file = self.embeddings_dir / "embeddings.npy"
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        self.embeddings = np.load(embeddings_file)
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        
        # Load metadata
        metadata_file = self.embeddings_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {len(self.metadata)} chunks")
        
        # Verify dimensions match
        if self.embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {self.embeddings.shape[1]} doesn't match index dimension {self.dimension}")
    
    def prepare_vectors(self, batch_size: int = 100) -> List[List[Dict[str, Any]]]:
        """
        Prepare vectors for upload to Pinecone.
        
        Args:
            batch_size: Number of vectors per batch
            
        Returns:
            List of batches, each containing vectors for upload
        """
        vectors = []
        
        for i, (embedding, meta) in enumerate(zip(self.embeddings, self.metadata)):
            vector_data = {
                'id': meta['chunk_id'],
                'values': embedding.tolist(),
                'metadata': {
                    'document_id': meta['document_id'],
                    'chunk_index': meta['chunk_index'],
                    'character_count': meta['character_count'],
                    'start_position': meta['start_position'],
                    'end_position': meta['end_position'],
                    'overlap_with_previous': meta['overlap_with_previous'],
                    'overlap_with_next': meta['overlap_with_next'],
                    'original_filename': meta['original_filename'],
                    'upload_timestamp': datetime.now().isoformat()
                }
            }
            vectors.append(vector_data)
        
        # Split into batches
        batches = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Prepared {len(vectors)} vectors in {len(batches)} batches")
        return batches
    
    def upload_vectors(self, batches: List[List[Dict[str, Any]]]):
        """
        Upload vectors to Pinecone in batches.
        
        Args:
            batches: List of vector batches
        """
        try:
            # Get index
            index = self.pc.Index(self.index_name)
            
            total_vectors = sum(len(batch) for batch in batches)
            uploaded = 0
            
            logger.info(f"Starting upload of {total_vectors} vectors...")
            
            for i, batch in enumerate(batches):
                try:
                    # Upload batch
                    index.upsert(vectors=batch)
                    uploaded += len(batch)
                    
                    logger.info(f"Batch {i+1}/{len(batches)} uploaded: {len(batch)} vectors")
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to upload batch {i+1}: {e}")
                    self.stats['upload_failed'] += len(batch)
                    continue
            
            self.stats['upload_success'] = uploaded
            self.stats['total_vectors'] = uploaded
            
            logger.info(f"Upload complete: {uploaded}/{total_vectors} vectors uploaded")
            
        except Exception as e:
            logger.error(f"Failed to upload vectors: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics from Pinecone."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def search_similar(self, 
                      query_vector: List[float], 
                      top_k: int = 10, 
                      filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar vectors with scores
        """
        try:
            index = self.pc.Index(self.index_name)
            
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            return results['matches']
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    def save_pinecone_data(self):
        """Save Pinecone-related data and statistics."""
        # Save statistics
        stats_file = self.pinecone_data_dir / "pinecone_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        # Save configuration
        config = {
            'index_name': self.index_name,
            'dimension': self.dimension,
            'metric': self.metric,
            'environment': self.environment,
            'created_date': datetime.now().isoformat()
        }
        
        config_file = self.pinecone_data_dir / "pinecone_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Get and save index stats
        index_stats = self.get_index_stats()
        if index_stats:
            stats_detail_file = self.pinecone_data_dir / "index_statistics.json"
            with open(stats_detail_file, 'w', encoding='utf-8') as f:
                json.dump(index_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pinecone data saved to: {self.pinecone_data_dir}")
    
    def process_all_embeddings(self, recreate_index: bool = False, batch_size: int = 100):
        """
        Process all embeddings and upload to Pinecone.
        
        Args:
            recreate_index: Whether to recreate the index
            batch_size: Batch size for uploads
        """
        start_time = datetime.now()
        
        try:
            # Create index
            self.create_index(recreate=recreate_index)
            
            # Load embeddings data
            self.load_embeddings_data()
            
            # Prepare vectors
            batches = self.prepare_vectors(batch_size=batch_size)
            
            # Upload vectors
            self.upload_vectors(batches)
            
            # Save data
            self.save_pinecone_data()
            
            # Calculate processing time
            end_time = datetime.now()
            self.stats['processing_time'] = (end_time - start_time).total_seconds()
            
            logger.info(f"Pinecone processing complete!")
            logger.info(f"Processed {self.stats['total_vectors']} vectors in {self.stats['processing_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during Pinecone processing: {e}")
            raise
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("PINECONE PROCESSING SUMMARY")
        print("="*60)
        print(f"Index: {self.index_name}")
        print(f"Dimension: {self.dimension}")
        print(f"Metric: {self.metric}")
        print(f"Environment: {self.environment}")
        print(f"Total vectors uploaded: {self.stats['total_vectors']}")
        print(f"Successful uploads: {self.stats['upload_success']}")
        print(f"Failed uploads: {self.stats['upload_failed']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        # Get current index stats
        index_stats = self.get_index_stats()
        if index_stats:
            print(f"\nCurrent Index Statistics:")
            print(f"  - Total vectors: {index_stats.get('total_vector_count', 'Unknown')}")
            print(f"  - Dimension: {index_stats.get('dimension', 'Unknown')}")
        
        print(f"\nOutput directory: {self.pinecone_data_dir}")
        print(f"Files created:")
        print(f"  - pinecone_statistics.json")
        print(f"  - pinecone_config.json")
        print(f"  - index_statistics.json")

def main():
    """Main function to run Pinecone processing."""
    print("Pinecone Vector Database Manager")
    print("=" * 40)
    
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("❌ .env file not found!")
            print("Please create a .env file with your Pinecone API key.")
            print("You can use pinecone_config.example as a template.")
            return
        
        # Create manager
        manager = PineconeManager()
        
        # Process embeddings
        manager.process_all_embeddings(recreate_index=False)
        manager.print_summary()
        
    except Exception as e:
        logger.error(f"Failed to process embeddings: {e}")
        print(f"❌ Error: {e}")
        print("Please check your Pinecone API key and configuration.")

if __name__ == "__main__":
    main()

