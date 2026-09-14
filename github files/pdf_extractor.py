#!/usr/bin/env python3
"""
PDF Text Extractor and JSON Generator for Medical Research Papers
Extracts specific parameters from PDF files and generates JSON output.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import PyPDF2
import pdfplumber
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text from PDF files and generate structured JSON data."""
    
    def __init__(self, input_dir: str, output_dir: str = "extracted_data"):
        """
        Initialize the PDF extractor.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save JSON files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Keywords patterns for different sections
        self.section_patterns = {
            'objective': [
                r'objective[s]?',
                r'aim[s]?',
                r'purpose[s]?',
                r'goal[s]?',
                r'rationale'
            ],
            'method': [
                r'method[s]?',
                r'methodology',
                r'study design',
                r'experimental design',
                r'procedure[s]?',
                r'protocol[s]?'
            ],
            'novelty': [
                r'novel[ty]?',
                r'innovation[s]?',
                r'novel approach',
                r'new method',
                r'breakthrough',
                r'advancement[s]?'
            ],
            'result': [
                r'result[s]?',
                r'finding[s]?',
                r'outcome[s]?',
                r'conclusion[s]?',
                r'summary'
            ],
            'comparison': [
                r'comparison[s]?',
                r'compare[d]?',
                r'versus',
                r'vs\.',
                r'contrast[s]?',
                r'benchmark[s]?'
            ],
            'limitation': [
                r'limitation[s]?',
                r'constraint[s]?',
                r'restriction[s]?',
                r'drawback[s]?',
                r'challenge[s]?',
                r'shortcoming[s]?'
            ],
            'physicians': [
                r'physician[s]?',
                r'doctor[s]?',
                r'clinician[s]?',
                r'medical professional[s]?',
                r'healthcare provider[s]?',
                r'practitioner[s]?'
            ],
            'responsible_ai': [
                r'responsible ai',
                r'ethical ai',
                r'ai ethics',
                r'artificial intelligence ethics',
                r'responsible artificial intelligence',
                r'ai governance',
                r'algorithmic bias',
                r'fairness',
                r'transparency',
                r'accountability'
            ],
            'keywords': [
                r'keyword[s]?',
                r'key word[s]?',
                r'terms',
                r'descriptors'
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text_content = ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path.name}: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except Exception as e:
                logger.error(f"PyPDF2 also failed for {pdf_path.name}: {e}")
                return ""
        
        return text_content.strip()
    
    def extract_section_content(self, text: str, section_type: str) -> List[str]:
        """
        Extract content related to a specific section type.
        
        Args:
            text: Full text content
            section_type: Type of section to extract
            
        Returns:
            List of extracted content snippets
        """
        patterns = self.section_patterns.get(section_type, [])
        extracted_content = []
        
        # Split text into sentences for better matching
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence contains relevant keywords
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    extracted_content.append(sentence)
                    break
        
        # If no specific content found, try to extract paragraphs containing keywords
        if not extracted_content:
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if len(paragraph) < 50:  # Skip very short paragraphs
                    continue
                    
                for pattern in patterns:
                    if re.search(pattern, paragraph, re.IGNORECASE):
                        extracted_content.append(paragraph)
                        break
        
        return extracted_content[:5]  # Limit to 5 most relevant snippets
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        Extract potential keywords from the text.
        
        Args:
            text: Full text content
            
        Returns:
            List of potential keywords
        """
        # Look for explicit keyword sections
        keyword_patterns = [
            r'keywords?:?\s*([^\n]+)',
            r'key words?:?\s*([^\n]+)',
            r'descriptors?:?\s*([^\n]+)'
        ]
        
        keywords = []
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by common separators
                kw_list = re.split(r'[,;]', match)
                keywords.extend([kw.strip() for kw in kw_list if kw.strip()])
        
        # If no explicit keywords found, extract medical/scientific terms
        if not keywords:
            # Common medical/scientific term patterns
            medical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            # Filter for terms that appear multiple times (likely important)
            term_counts = {}
            for term in medical_terms:
                if len(term) > 3:  # Skip very short terms
                    term_counts[term] = term_counts.get(term, 0) + 1
            
            # Get most frequent terms
            keywords = [term for term, count in sorted(term_counts.items(), 
                      key=lambda x: x[1], reverse=True)[:10]]
        
        return keywords[:15]  # Limit to 15 keywords
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Extract metadata and structured content from text.
        
        Args:
            text: Full text content
            filename: Name of the source file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'filename': filename,
            'extraction_date': datetime.now().isoformat(),
            'file_size': len(text),
            'word_count': len(text.split()),
            'objective': [],
            'method': [],
            'novelty': [],
            'result': [],
            'comparison': [],
            'limitation': [],
            'physicians': [],
            'responsible_ai': [],
            'keywords': []
        }
        
        # Extract content for each section
        for section in self.section_patterns.keys():
            if section == 'keywords':
                metadata[section] = self.extract_keywords_from_text(text)
            else:
                metadata[section] = self.extract_section_content(text, section)
        
        return metadata
    
    def process_pdf_file(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF file and extract structured data.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted data or None if failed
        """
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            # Extract text
            text_content = self.extract_text_from_pdf(pdf_path)
            
            if not text_content:
                logger.warning(f"No text extracted from {pdf_path.name}")
                return None
            
            # Extract metadata and structured content
            metadata = self.extract_metadata(text_content, pdf_path.name)
            
            # Add full text for reference (truncated if too long)
            if len(text_content) > 10000:
                metadata['full_text_preview'] = text_content[:10000] + "..."
            else:
                metadata['full_text_preview'] = text_content
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            return None
    
    def save_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """
        Save extracted data to JSON file.
        
        Args:
            data: Extracted data dictionary
            output_path: Path to save JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON: {output_path.name}")
        except Exception as e:
            logger.error(f"Error saving JSON {output_path.name}: {e}")
    
    def process_all_pdfs(self) -> None:
        """
        Process all PDF files in the input directory.
        """
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        successful_extractions = 0
        failed_extractions = 0
        
        for pdf_file in pdf_files:
            # Process PDF
            extracted_data = self.process_pdf_file(pdf_file)
            
            if extracted_data:
                # Generate output filename
                output_filename = pdf_file.stem + "_extracted.json"
                output_path = self.output_dir / output_filename
                
                # Save to JSON
                self.save_json(extracted_data, output_path)
                successful_extractions += 1
            else:
                failed_extractions += 1
        
        # Generate summary report
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_files': len(pdf_files),
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': f"{(successful_extractions/len(pdf_files)*100):.1f}%"
        }
        
        summary_path = self.output_dir / "extraction_summary.json"
        self.save_json(summary, summary_path)
        
        logger.info(f"Processing complete: {successful_extractions}/{len(pdf_files)} files processed successfully")
        logger.info(f"Summary saved to: {summary_path}")

def main():
    """Main function to run the PDF extraction process."""
    # Configuration
    input_directory = "nbib_files"  # Directory containing PDF files
    output_directory = "extracted_data"  # Directory to save JSON files
    
    # Check if input directory exists
    if not os.path.exists(input_directory):
        logger.error(f"Input directory '{input_directory}' does not exist!")
        return
    
    # Create extractor and process files
    extractor = PDFExtractor(input_directory, output_directory)
    extractor.process_all_pdfs()

if __name__ == "__main__":
    main()

