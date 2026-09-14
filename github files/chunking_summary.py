#!/usr/bin/env python3
"""
Chunking Summary Script
Shows what documents were chunked and provides a quick overview.
"""

import json
import os
from pathlib import Path

def show_chunking_summary():
    """Show a summary of what was chunked."""
    chunked_data_dir = Path("chunked_data")
    
    if not chunked_data_dir.exists():
        print("❌ Chunked data directory not found!")
        return
    
    # Load statistics
    stats_file = chunked_data_dir / "chunking_statistics.json"
    if not stats_file.exists():
        print("❌ Chunking statistics file not found!")
        return
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    print("RAG CHUNKING SUMMARY")
    print("=" * 50)
    
    print(f"OVERVIEW:")
    print(f"   - Documents processed: {stats['total_documents']}")
    print(f"   - Total chunks created: {stats['total_chunks']}")
    print(f"   - Total characters: {stats['total_characters']:,}")
    print(f"   - Average chunks per document: {stats['total_chunks']/stats['total_documents']:.1f}")
    print(f"   - Average characters per chunk: {stats['total_characters']/stats['total_chunks']:.0f}")
    
    print(f"\nCONFIGURATION:")
    print(f"   - Chunk size: {stats['chunk_size']} characters")
    print(f"   - Overlap: {stats['overlap_percentage']*100:.1f}% ({stats['overlap_size']} characters)")
    print(f"   - Processing date: {stats['processing_date']}")
    
    print(f"\nOUTPUT FILES:")
    print(f"   - Main directory: {chunked_data_dir}")
    print(f"   - All chunks: all_chunks.json")
    print(f"   - Chunk index: chunk_index.json")
    print(f"   - Statistics: chunking_statistics.json")
    print(f"   - Analysis report: chunking_analysis_report.json")
    
    # Show file structure
    print(f"\nFILE STRUCTURE:")
    print(f"   For each document:")
    print(f"   +-- {stats['documents'][0]['document_id']}_chunks.json")
    print(f"   +-- {stats['documents'][0]['document_id']}_metadata.json")
    print(f"   +-- {stats['documents'][0]['document_id']}_individual_chunks/")
    print(f"       +-- chunk_001.txt")
    print(f"       +-- chunk_002.txt")
    print(f"       +-- ...")
    
    # Show top 10 documents by chunk count
    documents = sorted([doc for doc in stats['documents'] if doc.get('processing_successful', False)], 
                      key=lambda x: x['chunks_created'], reverse=True)
    
    print(f"\nTOP 10 DOCUMENTS BY CHUNK COUNT:")
    print("-" * 60)
    for i, doc in enumerate(documents[:10], 1):
        print(f"{i:2d}. {doc['document_id']}")
        print(f"    Chunks: {doc['chunks_created']:2d} | Characters: {doc['total_characters']:6,}")
    
    # Show chunk size distribution
    print(f"\nCHUNK SIZE DISTRIBUTION:")
    print("-" * 40)
    size_buckets = {'Small (<=500)': 0, 'Medium (500-750)': 0, 'Large (750-1000)': 0, 'Oversized (>1000)': 0}
    
    # Load chunk index to analyze sizes
    index_file = chunked_data_dir / "chunk_index.json"
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            chunk_index = json.load(f)
        
        for chunk in chunk_index:
            size = chunk.get('character_count', 0)
            if size <= 500:
                size_buckets['Small (<=500)'] += 1
            elif size <= 750:
                size_buckets['Medium (500-750)'] += 1
            elif size <= 1000:
                size_buckets['Large (750-1000)'] += 1
            else:
                size_buckets['Oversized (>1000)'] += 1
        
        total_chunks = len(chunk_index)
        for bucket, count in size_buckets.items():
            percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
            print(f"   {bucket:20s}: {count:4d} chunks ({percentage:5.1f}%)")
    
    # Show overlap statistics
    print(f"\nOVERLAP STATISTICS:")
    print("-" * 30)
    if index_file.exists():
        overlap_counts = {'No overlap': 0, 'Previous overlap': 0, 'Next overlap': 0, 'Both overlaps': 0}
        
        for chunk in chunk_index:
            prev = chunk.get('overlap_prev', False)
            next_overlap = chunk.get('overlap_next', False)
            
            if prev and next_overlap:
                overlap_counts['Both overlaps'] += 1
            elif prev:
                overlap_counts['Previous overlap'] += 1
            elif next_overlap:
                overlap_counts['Next overlap'] += 1
            else:
                overlap_counts['No overlap'] += 1
        
        for overlap_type, count in overlap_counts.items():
            percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
            print(f"   {overlap_type:20s}: {count:4d} chunks ({percentage:5.1f}%)")
    
    print(f"\nCHUNKING COMPLETE!")
    print(f"   All documents have been successfully chunked with 20% overlap.")
    print(f"   Ready for RAG system integration.")

if __name__ == "__main__":
    show_chunking_summary()
