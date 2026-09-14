#!/usr/bin/env python3
"""
Embedding Generator for Chunked Documents
Generates embeddings for all chunked documents using all-MiniLM-L12-v2 model.
"""

import json
import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for chunked documents using sentence transformers."""
    
    def __init__(self, 
                 chunked_data_dir: str = "chunked_data",
                 embeddings_dir: str = "embeddings",
                 model_name: str = "all-MiniLM-L12-v2",
                 batch_size: int = 32):
        """
        Initialize the embedding generator.
        
        Args:
            chunked_data_dir: Directory containing chunked data
            embeddings_dir: Directory to save embeddings
            model_name: Sentence transformer model name
            batch_size: Batch size for embedding generation
        """
        self.chunked_data_dir = Path(chunked_data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Create embeddings directory
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Initialize statistics
        self.stats = {
            'model_name': model_name,
            'embedding_dimension': None,
            'total_chunks': 0,
            'total_embeddings': 0,
            'processing_date': datetime.now().isoformat(),
            'batch_size': batch_size,
            'documents_processed': 0,
            'processing_time_seconds': 0,
            'documents': []
        }
        
        logger.info(f"Embedding Generator initialized:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Input directory: {self.chunked_data_dir}")
        logger.info(f"  - Output directory: {self.embeddings_dir}")
    
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.stats['embedding_dimension'] = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.stats['embedding_dimension']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_chunks_data(self):
        """Load all chunks data."""
        # Load all chunks
        all_chunks_file = self.chunked_data_dir / "all_chunks.json"
        if not all_chunks_file.exists():
            logger.error(f"All chunks file not found: {all_chunks_file}")
            raise FileNotFoundError("All chunks file not found")
        
        with open(all_chunks_file, 'r', encoding='utf-8') as f:
            self.all_chunks = json.load(f)
        
        logger.info(f"Loaded {len(self.all_chunks)} chunks")
        self.stats['total_chunks'] = len(self.all_chunks)
    
    def prepare_texts_and_metadata(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Prepare texts and metadata for embedding generation.
        
        Returns:
            Tuple of (texts, metadata_list)
        """
        texts = []
        metadata_list = []
        
        for chunk in self.all_chunks:
            # Extract text content
            text = chunk.get('content', '').strip()
            if text:
                texts.append(text)
                
                # Prepare metadata
                metadata = {
                    'chunk_id': chunk.get('chunk_id'),
                    'document_id': chunk.get('document_id'),
                    'chunk_index': chunk.get('chunk_index'),
                    'character_count': chunk.get('character_count'),
                    'start_position': chunk.get('start_position'),
                    'end_position': chunk.get('end_position'),
                    'overlap_with_previous': chunk.get('overlap_with_previous', False),
                    'overlap_with_next': chunk.get('overlap_with_next', False),
                    'original_filename': chunk.get('document_id', 'unknown')
                }
                metadata_list.append(metadata)
        
        logger.info(f"Prepared {len(texts)} texts for embedding")
        return texts, metadata_list
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Save embeddings and metadata.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata_list: List of metadata dictionaries
        """
        # Save embeddings as numpy array
        embeddings_file = self.embeddings_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved embeddings to: {embeddings_file}")
        
        # Save metadata
        metadata_file = self.embeddings_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {metadata_file}")
        
        # Save embeddings with metadata combined
        combined_data = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
            combined_entry = {
                'chunk_id': metadata['chunk_id'],
                'document_id': metadata['document_id'],
                'chunk_index': metadata['chunk_index'],
                'embedding': embedding.tolist(),  # Convert numpy array to list for JSON
                'metadata': metadata
            }
            combined_data.append(combined_entry)
        
        combined_file = self.embeddings_dir / "embeddings_with_metadata.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved combined embeddings and metadata to: {combined_file}")
        
        # Save as pickle for faster loading (optional)
        pickle_file = self.embeddings_dir / "embeddings_with_metadata.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'metadata': metadata_list,
                'model_name': self.model_name,
                'embedding_dimension': self.stats['embedding_dimension']
            }, f)
        logger.info(f"Saved pickle file to: {pickle_file}")
    
    def save_document_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Save embeddings organized by document.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata_list: List of metadata dictionaries
        """
        # Group by document
        document_embeddings = {}
        
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
            doc_id = metadata['document_id']
            
            if doc_id not in document_embeddings:
                document_embeddings[doc_id] = {
                    'embeddings': [],
                    'metadata': []
                }
            
            document_embeddings[doc_id]['embeddings'].append(embedding.tolist())
            document_embeddings[doc_id]['metadata'].append(metadata)
        
        # Save each document's embeddings
        for doc_id, data in document_embeddings.items():
            doc_embeddings_file = self.embeddings_dir / f"{doc_id}_embeddings.json"
            
            doc_data = {
                'document_id': doc_id,
                'chunk_count': len(data['embeddings']),
                'embeddings': data['embeddings'],
                'metadata': data['metadata'],
                'model_name': self.model_name,
                'embedding_dimension': self.stats['embedding_dimension'],
                'generated_date': datetime.now().isoformat()
            }
            
            with open(doc_embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved embeddings for {len(document_embeddings)} documents")
    
    def generate_statistics(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Generate statistics about the embeddings."""
        self.stats['total_embeddings'] = len(embeddings)
        self.stats['documents_processed'] = len(set(meta['document_id'] for meta in metadata_list))
        
        # Calculate embedding statistics
        embedding_stats = {
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
            'min_norm': float(np.min(np.linalg.norm(embeddings, axis=1))),
            'max_norm': float(np.max(np.linalg.norm(embeddings, axis=1))),
            'mean_magnitude': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'shape': list(embeddings.shape)
        }
        
        self.stats['embedding_statistics'] = embedding_stats
        
        # Document statistics
        doc_stats = {}
        for metadata in metadata_list:
            doc_id = metadata['document_id']
            if doc_id not in doc_stats:
                doc_stats[doc_id] = 0
            doc_stats[doc_id] += 1
        
        self.stats['documents'] = [
            {'document_id': doc_id, 'chunk_count': count}
            for doc_id, count in doc_stats.items()
        ]
        
        # Sort by chunk count
        self.stats['documents'].sort(key=lambda x: x['chunk_count'], reverse=True)
    
    def save_statistics(self):
        """Save embedding statistics."""
        stats_file = self.embeddings_dir / "embedding_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics to: {stats_file}")
    
    def process_all_chunks(self):
        """Process all chunks and generate embeddings."""
        start_time = datetime.now()
        
        try:
            # Load model
            self.load_model()
            
            # Load chunks data
            self.load_chunks_data()
            
            # Prepare texts and metadata
            texts, metadata_list = self.prepare_texts_and_metadata()
            
            if not texts:
                logger.warning("No texts to process")
                return
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.generate_embeddings_batch(texts)
            
            # Save embeddings
            self.save_embeddings(embeddings, metadata_list)
            
            # Save document-organized embeddings
            self.save_document_embeddings(embeddings, metadata_list)
            
            # Generate and save statistics
            self.generate_statistics(embeddings, metadata_list)
            self.save_statistics()
            
            # Calculate processing time
            end_time = datetime.now()
            self.stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            logger.info(f"Embedding generation complete!")
            logger.info(f"Processed {len(embeddings)} chunks in {self.stats['processing_time_seconds']:.2f} seconds")
            logger.info(f"Embedding dimension: {self.stats['embedding_dimension']}")
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise
    
    def print_summary(self):
        """Print embedding generation summary."""
        print("\n" + "="*60)
        print("EMBEDDING GENERATION SUMMARY")
        print("="*60)
        print(f"Model: {self.stats['model_name']}")
        print(f"Embedding dimension: {self.stats['embedding_dimension']}")
        print(f"Total chunks processed: {self.stats['total_chunks']}")
        print(f"Total embeddings generated: {self.stats['total_embeddings']}")
        print(f"Documents processed: {self.stats['documents_processed']}")
        print(f"Processing time: {self.stats['processing_time_seconds']:.2f} seconds")
        print(f"Batch size: {self.stats['batch_size']}")
        
        if 'embedding_statistics' in self.stats:
            stats = self.stats['embedding_statistics']
            print(f"\nEmbedding Statistics:")
            print(f"  - Mean norm: {stats['mean_norm']:.4f}")
            print(f"  - Std norm: {stats['std_norm']:.4f}")
            print(f"  - Min norm: {stats['min_norm']:.4f}")
            print(f"  - Max norm: {stats['max_norm']:.4f}")
            print(f"  - Shape: {stats['shape']}")
        
        print(f"\nOutput directory: {self.embeddings_dir}")
        print(f"Files created:")
        print(f"  - embeddings.npy")
        print(f"  - metadata.json")
        print(f"  - embeddings_with_metadata.json")
        print(f"  - embeddings_with_metadata.pkl")
        print(f"  - embedding_statistics.json")
        print(f"  - {self.stats['documents_processed']} document-specific files")

def main():
    """Main function to run embedding generation."""
    print("Embedding Generator for Chunked Documents")
    print("=" * 50)
    
    # Configuration
    model_name = "all-MiniLM-L12-v2"
    batch_size = 32
    
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Create generator and process
    generator = EmbeddingGenerator(
        model_name=model_name,
        batch_size=batch_size
    )
    
    generator.process_all_chunks()
    generator.print_summary()

if __name__ == "__main__":
    main()

