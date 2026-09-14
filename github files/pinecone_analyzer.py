#!/usr/bin/env python3
"""
Pinecone Analyzer
Analyzes and reports on the Pinecone vector database.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from pinecone import Pinecone

class PineconeAnalyzer:
    """Analyze Pinecone vector database."""
    
    def __init__(self, pinecone_data_dir: str = "pinecone_data"):
        """
        Initialize Pinecone analyzer.
        
        Args:
            pinecone_data_dir: Directory containing Pinecone data
        """
        self.pinecone_data_dir = Path(pinecone_data_dir)
        
        # Load configuration
        load_dotenv('.env')
        
        # Configuration
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('INDEX_NAME', 'liver-cirrhosis-embeddings')
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found. Please set it in your .env file.")
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            raise Exception(f"Failed to connect to Pinecone: {e}")
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Error getting index statistics: {e}")
            return {}
    
    def analyze_vector_distribution(self) -> Dict[str, Any]:
        """Analyze vector distribution and metadata."""
        try:
            # Get index stats
            stats = self.get_index_statistics()
            
            # Sample some vectors to analyze metadata
            dummy_vector = [0.0] * 384
            results = self.index.query(
                vector=dummy_vector,
                top_k=1000,  # Get as many as possible
                include_metadata=True
            )
            
            # Analyze metadata
            document_counts = {}
            chunk_counts = {}
            overlap_stats = {
                'with_prev': 0,
                'with_next': 0,
                'with_both': 0,
                'with_none': 0
            }
            
            for match in results['matches']:
                metadata = match['metadata']
                
                # Document distribution
                doc_id = metadata.get('document_id', 'unknown')
                document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
                
                # Chunk index distribution
                chunk_idx = metadata.get('chunk_index', 0)
                chunk_counts[chunk_idx] = chunk_counts.get(chunk_idx, 0) + 1
                
                # Overlap analysis
                prev_overlap = metadata.get('overlap_with_previous', False)
                next_overlap = metadata.get('overlap_with_next', False)
                
                if prev_overlap and next_overlap:
                    overlap_stats['with_both'] += 1
                elif prev_overlap:
                    overlap_stats['with_prev'] += 1
                elif next_overlap:
                    overlap_stats['with_next'] += 1
                else:
                    overlap_stats['with_none'] += 1
            
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'document_count': len(document_counts),
                'chunk_distribution': chunk_counts,
                'overlap_statistics': overlap_stats,
                'top_documents': sorted(document_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            print(f"Error analyzing vector distribution: {e}")
            return {}
    
    def test_search_performance(self, num_queries: int = 10) -> Dict[str, Any]:
        """Test search performance with sample queries."""
        import time
        
        sample_queries = [
            "liver cirrhosis treatment",
            "hepatitis B virus",
            "liver transplantation",
            "fibrosis progression",
            "hepatocellular carcinoma",
            "portal hypertension",
            "ascites management",
            "hepatic encephalopathy",
            "non-alcoholic fatty liver",
            "alcoholic liver disease"
        ]
        
        results = []
        
        for i, query in enumerate(sample_queries[:num_queries]):
            try:
                # Generate dummy vector for testing
                dummy_vector = [0.1] * 384
                
                start_time = time.time()
                search_results = self.index.query(
                    vector=dummy_vector,
                    top_k=10,
                    include_metadata=True
                )
                end_time = time.time()
                
                results.append({
                    'query': query,
                    'response_time': end_time - start_time,
                    'results_count': len(search_results['matches']),
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'query': query,
                    'response_time': 0,
                    'results_count': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate statistics
        successful_queries = [r for r in results if r['success']]
        avg_response_time = sum(r['response_time'] for r in successful_queries) / len(successful_queries) if successful_queries else 0
        
        return {
            'total_queries': len(results),
            'successful_queries': len(successful_queries),
            'failed_queries': len(results) - len(successful_queries),
            'average_response_time': avg_response_time,
            'query_results': results
        }
    
    def print_comprehensive_report(self):
        """Print comprehensive analysis report."""
        print("\n" + "="*80)
        print("PINECONE VECTOR DATABASE ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        print("\nBASIC STATISTICS:")
        print("-" * 40)
        stats = self.get_index_statistics()
        print(f"Index name: {self.index_name}")
        print(f"Total vectors: {stats.get('total_vector_count', 'Unknown')}")
        print(f"Dimension: {stats.get('dimension', 'Unknown')}")
        print(f"Index fullness: {stats.get('index_fullness', 'Unknown')}")
        print(f"Namespaces: {len(stats.get('namespaces', {}))}")
        
        # Vector distribution analysis
        print(f"\nVECTOR DISTRIBUTION ANALYSIS:")
        print("-" * 40)
        distribution = self.analyze_vector_distribution()
        
        if distribution:
            print(f"Document count: {distribution.get('document_count', 'Unknown')}")
            print(f"Overlap statistics:")
            overlap_stats = distribution.get('overlap_statistics', {})
            total_overlap = sum(overlap_stats.values())
            for overlap_type, count in overlap_stats.items():
                percentage = (count / total_overlap * 100) if total_overlap > 0 else 0
                print(f"  - {overlap_type}: {count} ({percentage:.1f}%)")
            
            print(f"\nTop 5 documents by chunk count:")
            for i, (doc_id, count) in enumerate(distribution.get('top_documents', [])[:5], 1):
                print(f"  {i}. {doc_id}: {count} chunks")
        
        # Performance testing
        print(f"\nSEARCH PERFORMANCE TEST:")
        print("-" * 40)
        performance = self.test_search_performance(num_queries=5)
        
        if performance:
            print(f"Total queries tested: {performance['total_queries']}")
            print(f"Successful queries: {performance['successful_queries']}")
            print(f"Failed queries: {performance['failed_queries']}")
            print(f"Average response time: {performance['average_response_time']:.4f} seconds")
        
        # Configuration info
        print(f"\nCONFIGURATION:")
        print("-" * 40)
        print(f"API key configured: {'Yes' if self.api_key else 'No'}")
        print(f"Environment: {os.getenv('PINECONE_ENVIRONMENT', 'Default')}")
        print(f"Metric: {os.getenv('METRIC', 'cosine')}")
        
        print("\n" + "="*80)
    
    def save_analysis_report(self, output_file: str = "pinecone_analysis_report.json"):
        """Save analysis report to JSON file."""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'index_name': self.index_name,
            'basic_statistics': self.get_index_statistics(),
            'vector_distribution': self.analyze_vector_distribution(),
            'performance_test': self.test_search_performance(num_queries=3)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis report saved to: {output_file}")

def main():
    """Main function."""
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("❌ .env file not found!")
            print("Please run setup_pinecone.py first to configure Pinecone.")
            return
        
        # Create analyzer
        analyzer = PineconeAnalyzer()
        
        # Print comprehensive report
        analyzer.print_comprehensive_report()
        
        # Save analysis report
        analyzer.save_analysis_report()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your Pinecone configuration and connection.")

if __name__ == "__main__":
    main()

