#!/usr/bin/env python3
"""
Embedding Analyzer
Analyzes and reports on the generated embeddings.
"""

import json
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

class EmbeddingAnalyzer:
    """Analyze generated embeddings and provide insights."""
    
    def __init__(self, embeddings_dir: str = "embeddings"):
        """
        Initialize the analyzer.
        
        Args:
            embeddings_dir: Directory containing embedding files
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.load_data()
    
    def load_data(self):
        """Load embedding data."""
        # Load statistics
        stats_file = self.embeddings_dir / "embedding_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
        
        # Load metadata
        metadata_file = self.embeddings_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
        
        # Load embeddings
        embeddings_file = self.embeddings_dir / "embeddings.npy"
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
        else:
            self.embeddings = None
    
    def analyze_embedding_quality(self):
        """Analyze embedding quality metrics."""
        if self.embeddings is None:
            return {}
        
        # Calculate norms
        norms = np.linalg.norm(self.embeddings, axis=1)
        
        # Calculate pairwise similarities (sample)
        sample_size = min(1000, len(self.embeddings))
        sample_indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
        sample_embeddings = self.embeddings[sample_indices]
        
        # Calculate cosine similarities
        similarities = np.dot(sample_embeddings, sample_embeddings.T)
        
        # Remove diagonal (self-similarities)
        mask = np.ones(similarities.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        similarities = similarities[mask]
        
        return {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'embedding_shape': list(self.embeddings.shape),
            'sample_size': sample_size
        }
    
    def analyze_document_distribution(self):
        """Analyze distribution of embeddings across documents."""
        if not self.metadata:
            return {}
        
        doc_counts = defaultdict(int)
        doc_chars = defaultdict(int)
        
        for meta in self.metadata:
            doc_id = meta['document_id']
            doc_counts[doc_id] += 1
            doc_chars[doc_id] += meta.get('character_count', 0)
        
        doc_stats = []
        for doc_id, count in doc_counts.items():
            doc_stats.append({
                'document_id': doc_id,
                'chunk_count': count,
                'total_characters': doc_chars[doc_id],
                'avg_chunk_size': doc_chars[doc_id] / count if count > 0 else 0
            })
        
        doc_stats.sort(key=lambda x: x['chunk_count'], reverse=True)
        
        return {
            'total_documents': len(doc_counts),
            'total_chunks': sum(doc_counts.values()),
            'avg_chunks_per_doc': sum(doc_counts.values()) / len(doc_counts),
            'min_chunks_per_doc': min(doc_counts.values()),
            'max_chunks_per_doc': max(doc_counts.values()),
            'top_documents': doc_stats[:10]
        }
    
    def analyze_overlap_patterns(self):
        """Analyze overlap patterns in embeddings."""
        if not self.metadata:
            return {}
        
        overlap_stats = {
            'chunks_with_prev_overlap': 0,
            'chunks_with_next_overlap': 0,
            'chunks_with_both_overlaps': 0,
            'chunks_with_no_overlap': 0
        }
        
        for meta in self.metadata:
            prev_overlap = meta.get('overlap_with_previous', False)
            next_overlap = meta.get('overlap_with_next', False)
            
            if prev_overlap and next_overlap:
                overlap_stats['chunks_with_both_overlaps'] += 1
            elif prev_overlap:
                overlap_stats['chunks_with_prev_overlap'] += 1
            elif next_overlap:
                overlap_stats['chunks_with_next_overlap'] += 1
            else:
                overlap_stats['chunks_with_no_overlap'] += 1
        
        total = sum(overlap_stats.values())
        overlap_percentages = {}
        for key in overlap_stats:
            overlap_percentages[f'{key}_percentage'] = (overlap_stats[key] / total * 100) if total > 0 else 0
        
        overlap_stats.update(overlap_percentages)
        
        return overlap_stats
    
    def find_similar_chunks(self, query_chunk_id: str, top_k: int = 5):
        """Find most similar chunks to a given chunk."""
        if self.embeddings is None or not self.metadata:
            return []
        
        # Find the query chunk
        query_idx = None
        for i, meta in enumerate(self.metadata):
            if meta['chunk_id'] == query_chunk_id:
                query_idx = i
                break
        
        if query_idx is None:
            return []
        
        query_embedding = self.embeddings[query_idx].reshape(1, -1)
        
        # Calculate similarities
        similarities = np.dot(query_embedding, self.embeddings.T).flatten()
        
        # Get top k similar chunks (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip self
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': self.metadata[idx]['chunk_id'],
                'document_id': self.metadata[idx]['document_id'],
                'similarity': float(similarities[idx]),
                'preview': self.metadata[idx].get('preview', '')[:200] + "..." if len(self.metadata[idx].get('preview', '')) > 200 else self.metadata[idx].get('preview', '')
            })
        
        return results
    
    def print_comprehensive_report(self):
        """Print comprehensive analysis report."""
        print("\n" + "="*80)
        print("EMBEDDING ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        print("\nBASIC STATISTICS:")
        print("-" * 40)
        print(f"Model: {self.stats.get('model_name', 'Unknown')}")
        print(f"Embedding dimension: {self.stats.get('embedding_dimension', 'Unknown')}")
        print(f"Total embeddings: {self.stats.get('total_embeddings', 0)}")
        print(f"Documents processed: {self.stats.get('documents_processed', 0)}")
        print(f"Processing time: {self.stats.get('processing_time_seconds', 0):.2f} seconds")
        print(f"Batch size: {self.stats.get('batch_size', 'Unknown')}")
        print(f"Generation date: {self.stats.get('processing_date', 'Unknown')}")
        
        # Embedding quality analysis
        quality_analysis = self.analyze_embedding_quality()
        if quality_analysis:
            print(f"\nEMBEDDING QUALITY ANALYSIS:")
            print("-" * 40)
            print(f"Embedding shape: {quality_analysis['embedding_shape']}")
            print(f"Mean norm: {quality_analysis['mean_norm']:.4f}")
            print(f"Std norm: {quality_analysis['std_norm']:.4f}")
            print(f"Min norm: {quality_analysis['min_norm']:.4f}")
            print(f"Max norm: {quality_analysis['max_norm']:.4f}")
            print(f"Mean similarity (sample): {quality_analysis['mean_similarity']:.4f}")
            print(f"Std similarity: {quality_analysis['std_similarity']:.4f}")
            print(f"Similarity range: [{quality_analysis['min_similarity']:.4f}, {quality_analysis['max_similarity']:.4f}]")
        
        # Document distribution analysis
        doc_analysis = self.analyze_document_distribution()
        if doc_analysis:
            print(f"\nDOCUMENT DISTRIBUTION:")
            print("-" * 40)
            print(f"Total documents: {doc_analysis['total_documents']}")
            print(f"Total chunks: {doc_analysis['total_chunks']}")
            print(f"Average chunks per document: {doc_analysis['avg_chunks_per_doc']:.1f}")
            print(f"Min chunks per document: {doc_analysis['min_chunks_per_doc']}")
            print(f"Max chunks per document: {doc_analysis['max_chunks_per_doc']}")
            
            print(f"\nTOP 5 DOCUMENTS BY CHUNK COUNT:")
            for i, doc in enumerate(doc_analysis['top_documents'][:5], 1):
                print(f"{i}. {doc['document_id']}")
                print(f"   Chunks: {doc['chunk_count']} | Characters: {doc['total_characters']:,} | Avg chunk size: {doc['avg_chunk_size']:.0f}")
        
        # Overlap analysis
        overlap_analysis = self.analyze_overlap_patterns()
        if overlap_analysis:
            print(f"\nOVERLAP PATTERN ANALYSIS:")
            print("-" * 40)
            print(f"Chunks with previous overlap: {overlap_analysis['chunks_with_prev_overlap']} ({overlap_analysis['chunks_with_prev_overlap_percentage']:.1f}%)")
            print(f"Chunks with next overlap: {overlap_analysis['chunks_with_next_overlap']} ({overlap_analysis['chunks_with_next_overlap_percentage']:.1f}%)")
            print(f"Chunks with both overlaps: {overlap_analysis['chunks_with_both_overlaps']} ({overlap_analysis['chunks_with_both_overlaps_percentage']:.1f}%)")
            print(f"Chunks with no overlap: {overlap_analysis['chunks_with_no_overlap']} ({overlap_analysis['chunks_with_no_overlap_percentage']:.1f}%)")
        
        print(f"\nFILE STRUCTURE:")
        print("-" * 40)
        print(f"Main files:")
        print(f"  - embeddings.npy (numpy array)")
        print(f"  - metadata.json (chunk metadata)")
        print(f"  - embeddings_with_metadata.json (combined)")
        print(f"  - embeddings_with_metadata.pkl (pickle format)")
        print(f"  - embedding_statistics.json (statistics)")
        print(f"  - {doc_analysis.get('total_documents', 0)} document-specific files")
        
        print("\n" + "="*80)
    
    def save_analysis_report(self, output_file="embedding_analysis_report.json"):
        """Save analysis report to JSON file."""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'basic_statistics': {
                'model_name': self.stats.get('model_name'),
                'embedding_dimension': self.stats.get('embedding_dimension'),
                'total_embeddings': self.stats.get('total_embeddings'),
                'documents_processed': self.stats.get('documents_processed'),
                'processing_time': self.stats.get('processing_time_seconds'),
                'batch_size': self.stats.get('batch_size')
            },
            'embedding_quality': self.analyze_embedding_quality(),
            'document_distribution': self.analyze_document_distribution(),
            'overlap_patterns': self.analyze_overlap_patterns()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis report saved to: {output_file}")

def main():
    """Main function to run embedding analysis."""
    analyzer = EmbeddingAnalyzer()
    
    # Print comprehensive report
    analyzer.print_comprehensive_report()
    
    # Save analysis report
    analyzer.save_analysis_report()
    
    # Example: Find similar chunks (if embeddings are available)
    if analyzer.metadata:
        print(f"\nEXAMPLE: Finding similar chunks to first chunk")
        print("-" * 60)
        first_chunk_id = analyzer.metadata[0]['chunk_id']
        similar_chunks = analyzer.find_similar_chunks(first_chunk_id, top_k=3)
        
        print(f"Query chunk: {first_chunk_id}")
        print(f"Similar chunks:")
        for i, chunk in enumerate(similar_chunks, 1):
            print(f"{i}. {chunk['chunk_id']} (similarity: {chunk['similarity']:.4f})")
            print(f"   Preview: {chunk['preview']}")

if __name__ == "__main__":
    main()
