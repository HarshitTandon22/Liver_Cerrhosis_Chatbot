#!/usr/bin/env python3
"""
Data Analysis Script for Extracted PDF Data
Provides utilities to analyze and work with the extracted JSON data.
"""

import json
import os
import csv
from pathlib import Path
from collections import defaultdict, Counter

class ExtractedDataAnalyzer:
    """Analyze and provide insights from extracted PDF data."""
    
    def __init__(self, extracted_data_dir: str = "extracted_data"):
        """
        Initialize the analyzer.
        
        Args:
            extracted_data_dir: Directory containing extracted JSON files
        """
        self.extracted_data_dir = Path(extracted_data_dir)
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load all extracted JSON data."""
        json_files = list(self.extracted_data_dir.glob("*_extracted.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded data from {len(self.data)} files")
    
    def get_statistics(self):
        """Get overall statistics about the extracted data."""
        stats = {
            'total_files': len(self.data),
            'avg_file_size': sum(d['file_size'] for d in self.data) / len(self.data) if self.data else 0,
            'avg_word_count': sum(d['word_count'] for d in self.data) / len(self.data) if self.data else 0,
            'section_coverage': {}
        }
        
        # Calculate coverage for each section
        for section in ['objective', 'method', 'novelty', 'result', 'comparison', 
                       'limitation', 'physicians', 'responsible_ai', 'keywords']:
            files_with_section = sum(1 for d in self.data if d.get(section))
            stats['section_coverage'][section] = {
                'files_with_content': files_with_section,
                'coverage_percentage': (files_with_section / len(self.data) * 100) if self.data else 0
            }
        
        return stats
    
    def get_common_keywords(self, top_n: int = 20):
        """Get most common keywords across all files."""
        all_keywords = []
        for data in self.data:
            keywords = data.get('keywords', [])
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        return keyword_counts.most_common(top_n)
    
    def search_by_content(self, search_term: str, section: str = None):
        """Search for files containing specific content."""
        results = []
        
        for data in self.data:
            if section:
                # Search in specific section
                content = data.get(section, [])
                if isinstance(content, list):
                    content_text = ' '.join(content).lower()
                else:
                    content_text = str(content).lower()
                
                if search_term.lower() in content_text:
                    results.append({
                        'filename': data['filename'],
                        'section': section,
                        'matches': [item for item in content if search_term.lower() in item.lower()]
                    })
            else:
                # Search in all sections
                matches = []
                for section_name in ['objective', 'method', 'novelty', 'result', 'comparison', 
                                   'limitation', 'physicians', 'responsible_ai', 'keywords']:
                    content = data.get(section_name, [])
                    if isinstance(content, list):
                        section_matches = [item for item in content if search_term.lower() in item.lower()]
                        if section_matches:
                            matches.extend([(section_name, match) for match in section_matches])
                
                if matches:
                    results.append({
                        'filename': data['filename'],
                        'matches': matches
                    })
        
        return results
    
    def export_to_csv(self, output_file: str = "extracted_data_summary.csv"):
        """Export extracted data to CSV format."""
        rows = []
        
        for data in self.data:
            row = {
                'filename': data['filename'],
                'file_size': data['file_size'],
                'word_count': data['word_count'],
                'extraction_date': data['extraction_date'],
                'objective_count': len(data.get('objective', [])),
                'method_count': len(data.get('method', [])),
                'novelty_count': len(data.get('novelty', [])),
                'result_count': len(data.get('result', [])),
                'comparison_count': len(data.get('comparison', [])),
                'limitation_count': len(data.get('limitation', [])),
                'physicians_count': len(data.get('physicians', [])),
                'responsible_ai_count': len(data.get('responsible_ai', [])),
                'keywords_count': len(data.get('keywords', [])),
                'keywords': '; '.join(data.get('keywords', [])),
                'has_objective': len(data.get('objective', [])) > 0,
                'has_method': len(data.get('method', [])) > 0,
                'has_novelty': len(data.get('novelty', [])) > 0,
                'has_result': len(data.get('result', [])) > 0,
                'has_comparison': len(data.get('comparison', [])) > 0,
                'has_limitation': len(data.get('limitation', [])) > 0,
                'has_physicians': len(data.get('physicians', [])) > 0,
                'has_responsible_ai': len(data.get('responsible_ai', [])) > 0
            }
            rows.append(row)
        
        # Write to CSV using standard library
        if rows:
            fieldnames = rows[0].keys()
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"Data exported to {output_file}")
        return rows
    
    def get_files_by_section_content(self, section: str, min_content_count: int = 1):
        """Get files that have content in a specific section."""
        files_with_content = []
        
        for data in self.data:
            content = data.get(section, [])
            if len(content) >= min_content_count:
                files_with_content.append({
                    'filename': data['filename'],
                    'content_count': len(content),
                    'sample_content': content[:2] if content else []
                })
        
        return sorted(files_with_content, key=lambda x: x['content_count'], reverse=True)
    
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("EXTRACTED DATA SUMMARY REPORT")
        print("=" * 60)
        print(f"Total files processed: {stats['total_files']}")
        print(f"Average file size: {stats['avg_file_size']:.0f} characters")
        print(f"Average word count: {stats['avg_word_count']:.0f} words")
        print()
        
        print("SECTION COVERAGE:")
        print("-" * 40)
        for section, coverage in stats['section_coverage'].items():
            print(f"{section.capitalize():<15}: {coverage['files_with_content']:>3} files ({coverage['coverage_percentage']:>5.1f}%)")
        print()
        
        print("TOP KEYWORDS:")
        print("-" * 40)
        common_keywords = self.get_common_keywords(10)
        for keyword, count in common_keywords:
            print(f"{keyword:<25}: {count:>3} occurrences")
        print()
        
        print("FILES WITH MOST CONTENT BY SECTION:")
        print("-" * 40)
        for section in ['objective', 'method', 'result', 'keywords']:
            files = self.get_files_by_section_content(section, 1)
            if files:
                print(f"\n{section.upper()} - Top 3 files:")
                for i, file_info in enumerate(files[:3], 1):
                    print(f"  {i}. {file_info['filename']} ({file_info['content_count']} items)")

def main():
    """Main function to demonstrate the analyzer."""
    analyzer = ExtractedDataAnalyzer()
    
    # Print summary report
    analyzer.print_summary_report()
    
    # Export to CSV
    analyzer.export_to_csv()
    
    # Example search
    print("\n" + "=" * 60)
    print("SEARCH EXAMPLE: Files mentioning 'liver'")
    print("=" * 60)
    liver_results = analyzer.search_by_content('liver')
    for result in liver_results[:5]:  # Show first 5 results
        print(f"\nFile: {result['filename']}")
        for section, match in result['matches'][:2]:  # Show first 2 matches
            print(f"  {section}: {match[:100]}...")

if __name__ == "__main__":
    main()
