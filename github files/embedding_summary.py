#!/usr/bin/env python3
"""
Embedding Summary Script
Shows what embeddings were generated and provides a quick overview.
"""

import json
import os
from pathlib import Path

def show_embedding_summary():
    """Show a summary of what embeddings were generated."""
    embeddings_dir = Path("embeddings")
    
    if not embeddings_dir.exists():
        print("❌ Embeddings directory not found!")
        return
    
    # Load statistics
    stats_file = embeddings_dir / "embedding_statistics.json"
    if not stats_file.exists():
        print("❌ Embedding statistics file not found!")
        return
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 50)
    
    print(f"OVERVIEW:")
    print(f"   - Model used: {stats['model_name']}")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    print(f"   - Total embeddings: {stats['total_embeddings']}")
    print(f"   - Documents processed: {stats['documents_processed']}")
    print(f"   - Processing time: {stats['processing_time_seconds']:.2f} seconds")
    print(f"   - Batch size: {stats['batch_size']}")
    
    print(f"\nEMBEDDING QUALITY:")
    if 'embedding_statistics' in stats:
        emb_stats = stats['embedding_statistics']
        print(f"   - Shape: {emb_stats['shape']}")
        print(f"   - Mean norm: {emb_stats['mean_norm']:.4f}")
        print(f"   - Standard deviation norm: {emb_stats['std_norm']:.4f}")
        print(f"   - Min norm: {emb_stats['min_norm']:.4f}")
        print(f"   - Max norm: {emb_stats['max_norm']:.4f}")
    
    print(f"\nDOCUMENT DISTRIBUTION:")
    if 'documents' in stats:
        docs = stats['documents']
        print(f"   - Average chunks per document: {sum(d['chunk_count'] for d in docs) / len(docs):.1f}")
        print(f"   - Min chunks per document: {min(d['chunk_count'] for d in docs)}")
        print(f"   - Max chunks per document: {max(d['chunk_count'] for d in docs)}")
        
        print(f"\n   TOP 5 DOCUMENTS BY CHUNK COUNT:")
        for i, doc in enumerate(docs[:5], 1):
            print(f"   {i}. {doc['document_id']}: {doc['chunk_count']} chunks")
    
    print(f"\nOUTPUT FILES:")
    print(f"   - Main directory: {embeddings_dir}")
    print(f"   - embeddings.npy: Raw numpy array of all embeddings")
    print(f"   - metadata.json: Chunk metadata")
    print(f"   - embeddings_with_metadata.json: Combined embeddings and metadata")
    print(f"   - embeddings_with_metadata.pkl: Pickle format for fast loading")
    print(f"   - embedding_statistics.json: Generation statistics")
    print(f"   - {stats['documents_processed']} document-specific embedding files")
    
    print(f"\nFILE STRUCTURE:")
    print(f"   Main files:")
    print(f"   +-- embeddings.npy ({stats['total_embeddings']} x {stats['embedding_dimension']})")
    print(f"   +-- metadata.json ({stats['total_embeddings']} entries)")
    print(f"   +-- embeddings_with_metadata.json")
    print(f"   +-- embeddings_with_metadata.pkl")
    print(f"   +-- embedding_statistics.json")
    print(f"   +-- {stats['documents_processed']} document files")
    print(f"       +-- document_name_embeddings.json")
    
    # Show file sizes
    print(f"\nFILE SIZES:")
    try:
        embeddings_npy_size = (embeddings_dir / "embeddings.npy").stat().st_size / (1024*1024)
        metadata_size = (embeddings_dir / "metadata.json").stat().st_size / 1024
        combined_size = (embeddings_dir / "embeddings_with_metadata.json").stat().st_size / (1024*1024)
        
        print(f"   - embeddings.npy: {embeddings_npy_size:.1f} MB")
        print(f"   - metadata.json: {metadata_size:.1f} KB")
        print(f"   - embeddings_with_metadata.json: {combined_size:.1f} MB")
    except FileNotFoundError:
        print("   - File sizes not available")
    
    print(f"\nUSAGE EXAMPLES:")
    print(f"   Loading embeddings:")
    print(f"   ```python")
    print(f"   import numpy as np")
    print(f"   import json")
    print(f"   ")
    print(f"   # Load embeddings")
    print(f"   embeddings = np.load('embeddings/embeddings.npy')")
    print(f"   ")
    print(f"   # Load metadata")
    print(f"   with open('embeddings/metadata.json', 'r') as f:")
    print(f"       metadata = json.load(f)")
    print(f"   ```")
    
    print(f"\n   Loading combined data:")
    print(f"   ```python")
    print(f"   import pickle")
    print(f"   ")
    print(f"   # Load combined embeddings and metadata")
    print(f"   with open('embeddings/embeddings_with_metadata.pkl', 'rb') as f:")
    print(f"       data = pickle.load(f)")
    print(f"   embeddings = data['embeddings']")
    print(f"   metadata = data['metadata']")
    print(f"   ```")
    
    print(f"\nEMBEDDING GENERATION COMPLETE!")
    print(f"   All {stats['total_embeddings']} chunks have been embedded using {stats['model_name']}.")
    print(f"   Embeddings are ready for similarity search and RAG applications.")

if __name__ == "__main__":
    show_embedding_summary()

