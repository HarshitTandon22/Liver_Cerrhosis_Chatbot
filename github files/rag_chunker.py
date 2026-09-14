#!/usr/bin/env python3
"""
RAG Document Chunker
Chunks extracted JSON documents for Retrieval-Augmented Generation (RAG) systems.
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGChunker:
    """Chunk documents for RAG systems with configurable chunk size and overlap."""
    
    def __init__(self, 
                 extracted_data_dir: str = "extracted_data",
                 chunked_data_dir: str = "chunked_data",
                 chunk_size: int = 1000,
                 overlap_percentage: float = 0.2):
        """
        Initialize the RAG chunker.
        
        Args:
            extracted_data_dir: Directory containing extracted JSON files
            chunked_data_dir: Directory to save chunked documents
            chunk_size: Size of each chunk in characters
            overlap_percentage: Percentage of overlap between chunks (0.0 to 1.0)
        """
        self.extracted_data_dir = Path(extracted_data_dir)
        self.chunked_data_dir = Path(chunked_data_dir)
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_percentage)
        
        # Create output directory
        self.chunked_data_dir.mkdir(exist_ok=True)
        
        # Chunking statistics
        self.chunking_stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_characters': 0,
            'chunk_size': chunk_size,
            'overlap_percentage': overlap_percentage,
            'overlap_size': self.overlap_size,
            'processing_date': datetime.now().isoformat(),
            'documents': []
        }
        
        logger.info(f"RAG Chunker initialized:")
        logger.info(f"  - Chunk size: {chunk_size} characters")
        logger.info(f"  - Overlap: {overlap_percentage*100:.1f}% ({self.overlap_size} characters)")
        logger.info(f"  - Input directory: {self.extracted_data_dir}")
        logger.info(f"  - Output directory: {self.chunked_data_dir}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for chunking.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with chunking
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()
    
    def extract_document_content(self, data: Dict[str, Any]) -> str:
        """
        Extract and combine all relevant content from a document.
        
        Args:
            data: Extracted document data
            
        Returns:
            Combined text content
        """
        content_parts = []
        
        # Add filename as context
        content_parts.append(f"Document: {data.get('filename', 'Unknown')}")
        
        # Add structured sections
        sections = [
            ('Objective', 'objective'),
            ('Method', 'method'),
            ('Novelty', 'novelty'),
            ('Results', 'result'),
            ('Comparison', 'comparison'),
            ('Limitations', 'limitation'),
            ('Physicians', 'physicians'),
            ('Responsible AI', 'responsible_ai'),
            ('Keywords', 'keywords')
        ]
        
        for section_name, section_key in sections:
            section_content = data.get(section_key, [])
            if section_content:
                if isinstance(section_content, list):
                    section_text = ' '.join(str(item) for item in section_content if item)
                else:
                    section_text = str(section_content)
                
                if section_text.strip():
                    content_parts.append(f"{section_name}: {section_text}")
        
        # Add full text preview if available
        full_text = data.get('full_text_preview', '')
        if full_text and len(full_text) > 100:
            content_parts.append(f"Full Text: {full_text}")
        
        # Combine all content
        combined_content = '\n\n'.join(content_parts)
        return self.clean_text(combined_content)
    
    def create_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            doc_id: Document identifier
            
        Returns:
            List of chunk dictionaries
        """
        if len(text) <= self.chunk_size:
            # Text is smaller than chunk size, return as single chunk
            chunk_id = f"{doc_id}_chunk_001"
            return [{
                'chunk_id': chunk_id,
                'document_id': doc_id,
                'chunk_index': 0,
                'content': text,
                'character_count': len(text),
                'start_position': 0,
                'end_position': len(text),
                'overlap_with_previous': False,
                'overlap_with_next': False
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                sentence_end = text.rfind('.', start + self.chunk_size - 200, end)
                if sentence_end > start + self.chunk_size * 0.7:  # At least 70% of chunk size
                    end = sentence_end + 1
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk_id = f"{doc_id}_chunk_{chunk_index + 1:03d}"
                
                # Determine overlap status
                overlap_prev = chunk_index > 0
                overlap_next = end < len(text)
                
                chunk_data = {
                    'chunk_id': chunk_id,
                    'document_id': doc_id,
                    'chunk_index': chunk_index,
                    'content': chunk_content,
                    'character_count': len(chunk_content),
                    'start_position': start,
                    'end_position': end,
                    'overlap_with_previous': overlap_prev,
                    'overlap_with_next': overlap_next
                }
                
                chunks.append(chunk_data)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap_size, start + 1)
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
    
    def generate_chunk_metadata(self, chunks: List[Dict[str, Any]], doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata for a document's chunks.
        
        Args:
            chunks: List of chunks
            doc_metadata: Original document metadata
            
        Returns:
            Chunk metadata
        """
        total_chars = sum(chunk['character_count'] for chunk in chunks)
        
        return {
            'document_id': doc_metadata.get('filename', 'unknown'),
            'original_filename': doc_metadata.get('filename', 'unknown'),
            'extraction_date': doc_metadata.get('extraction_date', ''),
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'average_chunk_size': total_chars / len(chunks) if chunks else 0,
            'chunk_size_config': self.chunk_size,
            'overlap_config': self.overlap_size,
            'chunking_date': datetime.now().isoformat(),
            'source_sections': {
                'has_objective': bool(doc_metadata.get('objective')),
                'has_method': bool(doc_metadata.get('method')),
                'has_novelty': bool(doc_metadata.get('novelty')),
                'has_result': bool(doc_metadata.get('result')),
                'has_comparison': bool(doc_metadata.get('comparison')),
                'has_limitation': bool(doc_metadata.get('limitation')),
                'has_physicians': bool(doc_metadata.get('physicians')),
                'has_responsible_ai': bool(doc_metadata.get('responsible_ai')),
                'has_keywords': bool(doc_metadata.get('keywords'))
            }
        }
    
    def save_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        """
        Save chunks and metadata to files.
        
        Args:
            chunks: List of chunks to save
            metadata: Chunk metadata
        """
        doc_id = metadata['document_id']
        
        # Save chunks
        chunks_file = self.chunked_data_dir / f"{doc_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata_file = self.chunked_data_dir / f"{doc_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save individual chunk files for easy access
        chunks_dir = self.chunked_data_dir / f"{doc_id}_individual_chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        for chunk in chunks:
            chunk_file = chunks_dir / f"{chunk['chunk_id']}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk['content'])
        
        logger.info(f"Saved {len(chunks)} chunks for {doc_id}")
    
    def process_document(self, json_file: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single JSON document and create chunks.
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            Tuple of (chunks, metadata)
        """
        try:
            # Load document data
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Extract content
            content = self.extract_document_content(doc_data)
            
            if not content.strip():
                logger.warning(f"No content extracted from {json_file.name}")
                return [], {}
            
            # Create chunks
            doc_id = json_file.stem.replace('_extracted', '')
            chunks = self.create_chunks(content, doc_id)
            
            # Generate metadata
            metadata = self.generate_chunk_metadata(chunks, doc_data)
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
            return [], {}
    
    def process_all_documents(self) -> None:
        """
        Process all extracted JSON documents and create chunks.
        """
        json_files = list(self.extracted_data_dir.glob("*_extracted.json"))
        
        if not json_files:
            logger.warning(f"No extracted JSON files found in {self.extracted_data_dir}")
            return
        
        logger.info(f"Found {len(json_files)} documents to chunk")
        
        all_chunks = []
        processed_docs = 0
        
        for json_file in json_files:
            logger.info(f"Processing: {json_file.name}")
            
            chunks, metadata = self.process_document(json_file)
            
            if chunks:
                # Save chunks for this document
                self.save_chunks(chunks, metadata)
                
                # Update statistics
                self.chunking_stats['total_documents'] += 1
                self.chunking_stats['total_chunks'] += len(chunks)
                self.chunking_stats['total_characters'] += sum(c['character_count'] for c in chunks)
                
                doc_stats = {
                    'filename': json_file.name,
                    'document_id': metadata.get('document_id', 'unknown'),
                    'chunks_created': len(chunks),
                    'total_characters': sum(c['character_count'] for c in chunks),
                    'processing_successful': True
                }
                
                all_chunks.extend(chunks)
                processed_docs += 1
            else:
                doc_stats = {
                    'filename': json_file.name,
                    'document_id': 'unknown',
                    'chunks_created': 0,
                    'total_characters': 0,
                    'processing_successful': False,
                    'error': 'No chunks created'
                }
            
            self.chunking_stats['documents'].append(doc_stats)
        
        # Save comprehensive chunking report
        self.save_chunking_report(all_chunks)
        
        logger.info(f"Chunking complete: {processed_docs}/{len(json_files)} documents processed")
        logger.info(f"Total chunks created: {self.chunking_stats['total_chunks']}")
        logger.info(f"Total characters processed: {self.chunking_stats['total_characters']:,}")
    
    def save_chunking_report(self, all_chunks: List[Dict[str, Any]]) -> None:
        """
        Save comprehensive chunking report.
        
        Args:
            all_chunks: All created chunks
        """
        # Save all chunks in one file
        all_chunks_file = self.chunked_data_dir / "all_chunks.json"
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        # Save chunking statistics
        stats_file = self.chunked_data_dir / "chunking_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunking_stats, f, indent=2, ensure_ascii=False)
        
        # Create a searchable index
        index_file = self.chunked_data_dir / "chunk_index.json"
        index_data = []
        
        for chunk in all_chunks:
            index_entry = {
                'chunk_id': chunk['chunk_id'],
                'document_id': chunk['document_id'],
                'chunk_index': chunk['chunk_index'],
                'character_count': chunk['character_count'],
                'preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                'overlap_prev': chunk['overlap_with_previous'],
                'overlap_next': chunk['overlap_with_next']
            }
            index_data.append(index_entry)
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Chunking report saved to {self.chunked_data_dir}")
    
    def print_summary(self) -> None:
        """Print chunking summary."""
        stats = self.chunking_stats
        
        print("\n" + "="*60)
        print("RAG CHUNKING SUMMARY")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Chunk size: {stats['chunk_size']:,} characters")
        print(f"  - Overlap: {stats['overlap_percentage']*100:.1f}% ({stats['overlap_size']} characters)")
        print(f"  - Processing date: {stats['processing_date']}")
        print()
        print(f"Results:")
        print(f"  - Documents processed: {stats['total_documents']}")
        print(f"  - Total chunks created: {stats['total_chunks']}")
        print(f"  - Total characters: {stats['total_characters']:,}")
        if stats['total_chunks'] > 0:
            print(f"  - Average chunks per document: {stats['total_chunks']/stats['total_documents']:.1f}")
            print(f"  - Average chunk size: {stats['total_characters']/stats['total_chunks']:.0f} characters")
        print()
        print(f"Output files saved to: {self.chunked_data_dir}")

def main():
    """Main function to run the RAG chunking process."""
    # Configuration
    chunk_size = 1000  # Characters per chunk
    overlap_percentage = 0.2  # 20% overlap
    
    print("RAG Document Chunker")
    print("===================")
    print(f"Chunk size: {chunk_size} characters")
    print(f"Overlap: {overlap_percentage*100:.1f}%")
    print()
    
    # Create chunker and process documents
    chunker = RAGChunker(
        chunk_size=chunk_size,
        overlap_percentage=overlap_percentage
    )
    
    chunker.process_all_documents()
    chunker.print_summary()

if __name__ == "__main__":
    main()

