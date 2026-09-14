#!/usr/bin/env python3
"""
Chunking Analyzer
Analyzes and reports on the RAG chunking results.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

class ChunkingAnalyzer:
    """Analyze RAG chunking results and generate reports."""
    
    def __init__(self, chunked_data_dir: str = "chunked_data"):
        """
        Initialize the analyzer.
        
        Args:
            chunked_data_dir: Directory containing chunked data
        """
        self.chunked_data_dir = Path(chunked_data_dir)
        self.load_data()
    
    def load_data(self):
        """Load chunking data."""
        # Load statistics
        stats_file = self.chunked_data_dir / "chunking_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
        
        # Load chunk index
        index_file = self.chunked_data_dir / "chunk_index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                self.chunk_index = json.load(f)
        else:
            self.chunk_index = []
        
        # Load all chunks
        all_chunks_file = self.chunked_data_dir / "all_chunks.json"
        if all_chunks_file.exists():
            with open(all_chunks_file, 'r', encoding='utf-8') as f:
                self.all_chunks = json.load(f)
        else:
            self.all_chunks = []
    
    def analyze_chunk_distribution(self):
        """Analyze the distribution of chunks across documents."""
        chunk_counts = [doc['chunks_created'] for doc in self.stats.get('documents', []) if doc.get('processing_successful', False)]
        
        if not chunk_counts:
            return {}
        
        return {
            'min_chunks': min(chunk_counts),
            'max_chunks': max(chunk_counts),
            'avg_chunks': sum(chunk_counts) / len(chunk_counts),
            'median_chunks': sorted(chunk_counts)[len(chunk_counts) // 2],
            'total_documents': len(chunk_counts),
            'total_chunks': sum(chunk_counts)
        }
    
    def analyze_overlap_patterns(self):
        """Analyze overlap patterns in chunks."""
        overlap_stats = {
            'chunks_with_prev_overlap': 0,
            'chunks_with_next_overlap': 0,
            'chunks_with_both_overlaps': 0,
            'chunks_with_no_overlap': 0,
            'total_chunks': len(self.chunk_index)
        }
        
        for chunk in self.chunk_index:
            prev_overlap = chunk.get('overlap_prev', False)
            next_overlap = chunk.get('overlap_next', False)
            
            if prev_overlap and next_overlap:
                overlap_stats['chunks_with_both_overlaps'] += 1
            elif prev_overlap:
                overlap_stats['chunks_with_prev_overlap'] += 1
            elif next_overlap:
                overlap_stats['chunks_with_next_overlap'] += 1
            else:
                overlap_stats['chunks_with_no_overlap'] += 1
        
        return overlap_stats
    
    def analyze_chunk_sizes(self):
        """Analyze chunk size distribution."""
        chunk_sizes = [chunk.get('character_count', 0) for chunk in self.chunk_index]
        
        if not chunk_sizes:
            return {}
        
        return {
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'avg_size': sum(chunk_sizes) / len(chunk_sizes),
            'median_size': sorted(chunk_sizes)[len(chunk_sizes) // 2],
            'size_distribution': self.get_size_distribution(chunk_sizes)
        }
    
    def get_size_distribution(self, chunk_sizes):
        """Get size distribution buckets."""
        buckets = {
            '0-500': 0,
            '500-750': 0,
            '750-1000': 0,
            '1000-1250': 0,
            '1250+': 0
        }
        
        for size in chunk_sizes:
            if size <= 500:
                buckets['0-500'] += 1
            elif size <= 750:
                buckets['500-750'] += 1
            elif size <= 1000:
                buckets['750-1000'] += 1
            elif size <= 1250:
                buckets['1000-1250'] += 1
            else:
                buckets['1250+'] += 1
        
        return buckets
    
    def find_top_chunking_documents(self, top_n=10):
        """Find documents with most chunks."""
        documents = [doc for doc in self.stats.get('documents', []) if doc.get('processing_successful', False)]
        sorted_docs = sorted(documents, key=lambda x: x['chunks_created'], reverse=True)
        return sorted_docs[:top_n]
    
    def analyze_content_themes(self):
        """Analyze content themes in chunks."""
        # Sample some chunks to analyze content themes
        sample_chunks = self.chunk_index[:100]  # Analyze first 100 chunks
        
        themes = defaultdict(int)
        keywords = defaultdict(int)
        
        for chunk in sample_chunks:
            preview = chunk.get('preview', '').lower()
            
            # Count medical themes
            if 'liver' in preview:
                themes['Liver'] += 1
            if 'cirrhosis' in preview:
                themes['Cirrhosis'] += 1
            if 'hepatitis' in preview:
                themes['Hepatitis'] += 1
            if 'patient' in preview:
                themes['Patient'] += 1
            if 'treatment' in preview:
                themes['Treatment'] += 1
            if 'study' in preview:
                themes['Study'] += 1
            if 'method' in preview:
                themes['Method'] += 1
            if 'result' in preview:
                themes['Result'] += 1
        
        return dict(themes)
    
    def generate_document_report(self, document_id):
        """Generate a detailed report for a specific document."""
        # Find document in stats
        doc_stats = None
        for doc in self.stats.get('documents', []):
            if doc.get('document_id') == document_id:
                doc_stats = doc
                break
        
        if not doc_stats:
            return None
        
        # Find chunks for this document
        doc_chunks = [chunk for chunk in self.chunk_index if chunk.get('document_id') == document_id]
        
        return {
            'document_id': document_id,
            'chunks_created': doc_stats.get('chunks_created', 0),
            'total_characters': doc_stats.get('total_characters', 0),
            'avg_chunk_size': doc_stats.get('total_characters', 0) / doc_stats.get('chunks_created', 1),
            'chunk_list': [
                {
                    'chunk_id': chunk.get('chunk_id'),
                    'chunk_index': chunk.get('chunk_index'),
                    'character_count': chunk.get('character_count'),
                    'preview': chunk.get('preview', '')[:100] + "..." if len(chunk.get('preview', '')) > 100 else chunk.get('preview', ''),
                    'overlap_prev': chunk.get('overlap_prev', False),
                    'overlap_next': chunk.get('overlap_next', False)
                }
                for chunk in doc_chunks
            ]
        }
    
    def print_comprehensive_report(self):
        """Print a comprehensive analysis report."""
        print("\n" + "="*80)
        print("RAG CHUNKING ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        print("\nBASIC STATISTICS:")
        print("-" * 40)
        print(f"Total documents processed: {self.stats.get('total_documents', 0)}")
        print(f"Total chunks created: {self.stats.get('total_chunks', 0)}")
        print(f"Total characters processed: {self.stats.get('total_characters', 0):,}")
        print(f"Chunk size configuration: {self.stats.get('chunk_size', 0)} characters")
        print(f"Overlap configuration: {self.stats.get('overlap_percentage', 0)*100:.1f}% ({self.stats.get('overlap_size', 0)} characters)")
        print(f"Processing date: {self.stats.get('processing_date', 'Unknown')}")
        
        # Chunk distribution analysis
        dist_analysis = self.analyze_chunk_distribution()
        if dist_analysis:
            print(f"\nCHUNK DISTRIBUTION:")
            print("-" * 40)
            print(f"Average chunks per document: {dist_analysis['avg_chunks']:.1f}")
            print(f"Minimum chunks per document: {dist_analysis['min_chunks']}")
            print(f"Maximum chunks per document: {dist_analysis['max_chunks']}")
            print(f"Median chunks per document: {dist_analysis['median_chunks']}")
        
        # Chunk size analysis
        size_analysis = self.analyze_chunk_sizes()
        if size_analysis:
            print(f"\nCHUNK SIZE ANALYSIS:")
            print("-" * 40)
            print(f"Average chunk size: {size_analysis['avg_size']:.0f} characters")
            print(f"Minimum chunk size: {size_analysis['min_size']} characters")
            print(f"Maximum chunk size: {size_analysis['max_size']} characters")
            print(f"Median chunk size: {size_analysis['median_size']} characters")
            
            print(f"\nSize distribution:")
            for bucket, count in size_analysis['size_distribution'].items():
                percentage = (count / len(self.chunk_index) * 100) if self.chunk_index else 0
                print(f"  {bucket} characters: {count} chunks ({percentage:.1f}%)")
        
        # Overlap analysis
        overlap_analysis = self.analyze_overlap_patterns()
        print(f"\nOVERLAP ANALYSIS:")
        print("-" * 40)
        total_chunks = overlap_analysis['total_chunks']
        if total_chunks > 0:
            print(f"Chunks with previous overlap: {overlap_analysis['chunks_with_prev_overlap']} ({overlap_analysis['chunks_with_prev_overlap']/total_chunks*100:.1f}%)")
            print(f"Chunks with next overlap: {overlap_analysis['chunks_with_next_overlap']} ({overlap_analysis['chunks_with_next_overlap']/total_chunks*100:.1f}%)")
            print(f"Chunks with both overlaps: {overlap_analysis['chunks_with_both_overlaps']} ({overlap_analysis['chunks_with_both_overlaps']/total_chunks*100:.1f}%)")
            print(f"Chunks with no overlap: {overlap_analysis['chunks_with_no_overlap']} ({overlap_analysis['chunks_with_no_overlap']/total_chunks*100:.1f}%)")
        
        # Top documents by chunk count
        top_docs = self.find_top_chunking_documents(5)
        print(f"\nTOP 5 DOCUMENTS BY CHUNK COUNT:")
        print("-" * 40)
        for i, doc in enumerate(top_docs, 1):
            print(f"{i}. {doc['document_id']}: {doc['chunks_created']} chunks ({doc['total_characters']:,} characters)")
        
        # Content themes
        themes = self.analyze_content_themes()
        if themes:
            print(f"\nCONTENT THEMES (from sample):")
            print("-" * 40)
            sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            for theme, count in sorted_themes[:10]:
                print(f"{theme}: {count} chunks")
        
        print("\n" + "="*80)
    
    def save_analysis_report(self, output_file="chunking_analysis_report.json"):
        """Save analysis report to JSON file."""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'basic_statistics': {
                'total_documents': self.stats.get('total_documents', 0),
                'total_chunks': self.stats.get('total_chunks', 0),
                'total_characters': self.stats.get('total_characters', 0),
                'chunk_size_config': self.stats.get('chunk_size', 0),
                'overlap_config': self.stats.get('overlap_percentage', 0),
                'overlap_size_config': self.stats.get('overlap_size', 0)
            },
            'chunk_distribution': self.analyze_chunk_distribution(),
            'chunk_size_analysis': self.analyze_chunk_sizes(),
            'overlap_analysis': self.analyze_overlap_patterns(),
            'top_documents': self.find_top_chunking_documents(10),
            'content_themes': self.analyze_content_themes()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis report saved to: {output_file}")

def main():
    """Main function to run the chunking analysis."""
    analyzer = ChunkingAnalyzer()
    
    # Print comprehensive report
    analyzer.print_comprehensive_report()
    
    # Save analysis report
    analyzer.save_analysis_report()
    
    # Example: Generate report for a specific document
    if analyzer.stats.get('documents'):
        sample_doc_id = analyzer.stats['documents'][0]['document_id']
        print(f"\nEXAMPLE DOCUMENT REPORT: {sample_doc_id}")
        print("-" * 60)
        doc_report = analyzer.generate_document_report(sample_doc_id)
        if doc_report:
            print(f"Chunks created: {doc_report['chunks_created']}")
            print(f"Total characters: {doc_report['total_characters']:,}")
            print(f"Average chunk size: {doc_report['avg_chunk_size']:.0f} characters")
            print(f"\nFirst 3 chunks:")
            for i, chunk in enumerate(doc_report['chunk_list'][:3], 1):
                print(f"{i}. {chunk['chunk_id']} ({chunk['character_count']} chars)")
                print(f"   Preview: {chunk['preview']}")
                print(f"   Overlaps: prev={chunk['overlap_prev']}, next={chunk['overlap_next']}")

if __name__ == "__main__":
    main()

