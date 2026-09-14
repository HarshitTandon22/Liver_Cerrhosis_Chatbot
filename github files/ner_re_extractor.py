#!/usr/bin/env python3
"""
NER/RE Extraction Module with Normalization and Provenance Tagging
Extracts entities and relations from medical research papers with provenance tracking.
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import spacy
from spacy import displacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str  # DISEASE, TREATMENT, SYMPTOM, ORGAN, PROCEDURE, etc.
    normalized: str  # Normalized form
    start: int
    end: int
    confidence: float
    source: str  # DOI/PMID or filename
    span: Tuple[int, int]  # Character span in document

@dataclass
class Relation:
    """Represents an extracted relation."""
    subject: Entity
    predicate: str  # TREATS, CAUSES, ASSOCIATED_WITH, etc.
    object: Entity
    confidence: float
    source: str
    context: str  # Surrounding text
    model: str  # Model used for extraction

@dataclass
class Provenance:
    """Provenance information for entities and relations."""
    doi: Optional[str] = None
    pmid: Optional[str] = None
    span: Tuple[int, int] = (0, 0)
    model: str = "spacy_en_core_web_sm"
    confidence: float = 0.0
    extraction_date: str = None

class MedicalEntityNormalizer:
    """Normalizes medical entities to standard terminology."""
    
    def __init__(self):
        # Medical entity normalization mappings
        self.normalization_map = {
            # Liver-related terms
            'liver cirrhosis': 'CIRRHOSIS',
            'cirrhosis': 'CIRRHOSIS',
            'hepatic cirrhosis': 'CIRRHOSIS',
            'portal vein thrombosis': 'PORTAL_VEIN_THROMBOSIS',
            'pvt': 'PORTAL_VEIN_THROMBOSIS',
            'hepatitis': 'HEPATITIS',
            'hepatocellular carcinoma': 'HEPATOCELLULAR_CARCINOMA',
            'hcc': 'HEPATOCELLULAR_CARCINOMA',
            'ascites': 'ASCITES',
            'varices': 'VARICES',
            'esophageal varices': 'ESOPHAGEAL_VARICES',
            'hepatic encephalopathy': 'HEPATIC_ENCEPHALOPATHY',
            'jaundice': 'JAUNDICE',
            
            # Treatments
            'transplantation': 'LIVER_TRANSPLANTATION',
            'liver transplantation': 'LIVER_TRANSPLANTATION',
            'anticoagulation': 'ANTICOAGULATION',
            'tips': 'TIPS',
            'transjugular intrahepatic portosystemic shunt': 'TIPS',
            'sclerotherapy': 'SCLEROTHERAPY',
            'band ligation': 'BAND_LIGATION',
            
            # Procedures
            'ultrasound': 'ULTRASOUND',
            'ct scan': 'CT_SCAN',
            'mri': 'MRI',
            'endoscopy': 'ENDOSCOPY',
            'biopsy': 'BIOPSY',
            
            # Symptoms
            'abdominal pain': 'ABDOMINAL_PAIN',
            'fatigue': 'FATIGUE',
            'nausea': 'NAUSEA',
        }
        
    def normalize(self, entity_text: str) -> str:
        """Normalize entity text to standard form."""
        text_lower = entity_text.lower().strip()
        
        # Direct mapping
        if text_lower in self.normalization_map:
            return self.normalization_map[text_lower]
        
        # Partial matching for compound terms
        for key, normalized in self.normalization_map.items():
            if key in text_lower or text_lower in key:
                return normalized
        
        # Default: uppercase with underscores
        return text_lower.upper().replace(' ', '_').replace('-', '_')

class NERExtractor:
    """Named Entity Recognition extractor using spaCy."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install it:")
            logger.error("python -m spacy download en_core_web_sm")
            raise
        
        self.normalizer = MedicalEntityNormalizer()
        
        # Medical entity patterns
        self.medical_patterns = [
            # Diseases
            r'\b(?:liver|hepatic)\s+cirrhosis\b',
            r'\bportal\s+vein\s+thrombosis\b',
            r'\b(?:hepatitis\s+)?[ABC]\b',
            r'\bhepatocellular\s+carcinoma\b',
            r'\bHCC\b',
            r'\bascites\b',
            r'\bvarices\b',
            r'\besophageal\s+varices\b',
            r'\bhepatic\s+encephalopathy\b',
            
            # Treatments
            r'\b(?:liver\s+)?transplantation\b',
            r'\banticoagulation\b',
            r'\bTIPS\b',
            r'\btransjugular\s+intrahepatic\s+portosystemic\s+shunt\b',
            r'\bsclerotherapy\b',
            r'\bband\s+ligation\b',
            
            # Procedures
            r'\b(?:ultrasound|US)\b',
            r'\bCT\s+scan\b',
            r'\bMRI\b',
            r'\bendoscopy\b',
            r'\bbiopsy\b',
        ]
    
    def extract_entities(self, text: str, source: str, confidence_threshold: float = 0.5) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        
        # Use spaCy for general NER
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Filter for medical-relevant entities
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                continue
            
            normalized = self.normalizer.normalize(ent.text)
            entity = Entity(
                text=ent.text,
                label=self._map_label(ent.label_),
                normalized=normalized,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.8,  # Default confidence for spaCy
                source=source,
                span=(ent.start_char, ent.end_char)
            )
            entities.append(entity)
        
        # Extract medical-specific patterns
        for pattern in self.medical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                normalized = self.normalizer.normalize(matched_text)
                
                entity = Entity(
                    text=matched_text,
                    label='DISEASE' if 'cirrhosis' in matched_text.lower() or 'thrombosis' in matched_text.lower() else 'MEDICAL_TERM',
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    source=source,
                    span=(match.start(), match.end())
                )
                entities.append(entity)
        
        # Remove duplicates based on text and position
        unique_entities = []
        seen_keys = set()
        for entity in entities:
            # Use tuple of text, start, end as key for deduplication
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _map_label(self, spacy_label: str) -> str:
        """Map spaCy labels to medical entity labels."""
        label_map = {
            'DISEASE': 'DISEASE',
            'ORG': 'ORGANIZATION',
            'PERSON': 'PERSON',
            'GPE': 'LOCATION',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'MONEY': 'MONEY',
            'PERCENT': 'PERCENT',
            'QUANTITY': 'QUANTITY',
        }
        return label_map.get(spacy_label, 'OTHER')

class REExtractor:
    """Relation Extraction extractor."""
    
    def __init__(self):
        self.normalizer = MedicalEntityNormalizer()
        
        # Relation patterns
        self.relation_patterns = [
            # Treatment relations
            (r'(\w+(?:\s+\w+)*)\s+(?:treats?|treating|treatment\s+of)\s+(\w+(?:\s+\w+)*)', 'TREATS'),
            (r'(\w+(?:\s+\w+)*)\s+(?:is\s+used\s+to\s+treat|for\s+treating)\s+(\w+(?:\s+\w+)*)', 'TREATS'),
            
            # Cause relations
            (r'(\w+(?:\s+\w+)*)\s+(?:causes?|causing|leading\s+to)\s+(\w+(?:\s+\w+)*)', 'CAUSES'),
            (r'(\w+(?:\s+\w+)*)\s+(?:results?\s+in|results?\s+from)\s+(\w+(?:\s+\w+)*)', 'CAUSES'),
            
            # Association relations
            (r'(\w+(?:\s+\w+)*)\s+(?:is\s+associated\s+with|associated\s+with)\s+(\w+(?:\s+\w+)*)', 'ASSOCIATED_WITH'),
            (r'(\w+(?:\s+\w+)*)\s+(?:and|,)\s+(\w+(?:\s+\w+)*)\s+(?:are\s+related|relation)', 'ASSOCIATED_WITH'),
            
            # Symptom relations
            (r'(\w+(?:\s+\w+)*)\s+(?:presents?\s+with|presents?\s+as)\s+(\w+(?:\s+\w+)*)', 'PRESENTS_WITH'),
            (r'(\w+(?:\s+\w+)*)\s+(?:symptoms?\s+include|symptoms?\s+are)\s+(\w+(?:\s+\w+)*)', 'HAS_SYMPTOM'),
            
            # Diagnostic relations
            (r'(\w+(?:\s+\w+)*)\s+(?:is\s+diagnosed\s+by|diagnosis\s+of)\s+(\w+(?:\s+\w+)*)', 'DIAGNOSED_BY'),
            (r'(\w+(?:\s+\w+)*)\s+(?:detected\s+by|detection\s+of)\s+(\w+(?:\s+\w+)*)', 'DETECTED_BY'),
        ]
    
    def extract_relations(self, text: str, entities: List[Entity], source: str) -> List[Relation]:
        """Extract relations from text given entities."""
        relations = []
        
        # Extract relations using patterns
        for pattern, predicate in self.relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject_text = match.group(1).strip()
                object_text = match.group(2).strip()
                
                # Find matching entities
                subject_entity = self._find_entity(subject_text, entities, match.start(1))
                object_entity = self._find_entity(object_text, entities, match.start(2))
                
                if subject_entity and object_entity:
                    relation = Relation(
                        subject=subject_entity,
                        predicate=predicate,
                        object=object_entity,
                        confidence=0.7,
                        source=source,
                        context=text[max(0, match.start()-100):match.end()+100],
                        model="pattern_based"
                    )
                    relations.append(relation)
        
        # Extract entity co-occurrence relations
        sorted_entities = sorted(entities, key=lambda e: e.start)
        
        for i, entity1 in enumerate(sorted_entities):
            for entity2 in sorted_entities[i+1:]:
                # If entities are close together, they might be related
                if entity2.start - entity1.end < 200:
                    relation = Relation(
                        subject=entity1,
                        predicate='CO_OCCURS_WITH',
                        object=entity2,
                        confidence=0.5,
                        source=source,
                        context=text[max(0, entity1.start-50):min(len(text), entity2.end+50)],
                        model="co_occurrence"
                    )
                    relations.append(relation)
        
        return relations
    
    def _find_entity(self, text: str, entities: List[Entity], position: int) -> Optional[Entity]:
        """Find entity matching text near position."""
        text_lower = text.lower()
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                if abs(entity.start - position) < 500:  # Within 500 chars
                    return entity
        return None

class NERREExtractor:
    """Main NER/RE extraction orchestrator."""
    
    def __init__(self, extracted_data_dir: str = "extracted_data", output_dir: str = "kg_data"):
        self.extracted_data_dir = Path(extracted_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.ner_extractor = NERExtractor()
        self.re_extractor = REExtractor()
        
        self.stats = {
            'total_documents': 0,
            'total_entities': 0,
            'total_relations': 0,
            'processing_date': datetime.now().isoformat()
        }
    
    def extract_from_document(self, doc_data: Dict[str, Any]) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from a document."""
        filename = doc_data.get('filename', 'unknown')
        source = filename.replace('.pdf', '').replace('_extracted', '')
        
        # Combine all text sections
        text_parts = []
        for section in ['objective', 'method', 'result', 'comparison', 'limitation']:
            section_data = doc_data.get(section, [])
            if isinstance(section_data, list):
                text_parts.extend(section_data)
            else:
                text_parts.append(str(section_data))
        
        full_text = ' '.join(text_parts)
        
        # Extract entities
        entities = self.ner_extractor.extract_entities(full_text, source)
        
        # Extract relations
        relations = self.re_extractor.extract_relations(full_text, entities, source)
        
        return entities, relations
    
    def process_all_documents(self):
        """Process all extracted documents."""
        extracted_files = list(self.extracted_data_dir.glob("*_extracted.json"))
        
        all_entities = []
        all_relations = []
        document_extractions = {}
        
        logger.info(f"Processing {len(extracted_files)} documents...")
        
        for file_path in extracted_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                entities, relations = self.extract_from_document(doc_data)
                
                # Add provenance
                doi = self._extract_doi(doc_data)
                pmid = self._extract_pmid(doc_data)
                
                for entity in entities:
                    # Create new entity with updated source
                    updated_entity = Entity(
                        text=entity.text,
                        label=entity.label,
                        normalized=entity.normalized,
                        start=entity.start,
                        end=entity.end,
                        confidence=entity.confidence,
                        source=file_path.stem,
                        span=entity.span
                    )
                    all_entities.append(updated_entity)
                
                for relation in relations:
                    # Create new relation with updated source
                    updated_relation = Relation(
                        subject=relation.subject,
                        predicate=relation.predicate,
                        object=relation.object,
                        confidence=relation.confidence,
                        source=file_path.stem,
                        context=relation.context,
                        model=relation.model
                    )
                    all_relations.append(updated_relation)
                
                document_extractions[file_path.stem] = {
                    'entities': [asdict(e) for e in entities],
                    'relations': [asdict(r) for r in relations],
                    'doi': doi,
                    'pmid': pmid
                }
                
                self.stats['total_entities'] += len(entities)
                self.stats['total_relations'] += len(relations)
                
                logger.info(f"Processed {file_path.name}: {len(entities)} entities, {len(relations)} relations")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        self.stats['total_documents'] = len(document_extractions)
        
        # Save results
        self._save_extractions(all_entities, all_relations, document_extractions)
        
        return all_entities, all_relations, document_extractions
    
    def _extract_doi(self, doc_data: Dict[str, Any]) -> Optional[str]:
        """Extract DOI from document."""
        # Try to extract from full_text_preview or filename
        full_text = doc_data.get('full_text_preview', '')
        doi_match = re.search(r'DOI:\s*([^\s]+)', full_text, re.IGNORECASE)
        if doi_match:
            return doi_match.group(1)
        return None
    
    def _extract_pmid(self, doc_data: Dict[str, Any]) -> Optional[str]:
        """Extract PMID from document."""
        # Try to extract from full_text_preview
        full_text = doc_data.get('full_text_preview', '')
        pmid_match = re.search(r'PMID:\s*(\d+)', full_text, re.IGNORECASE)
        if pmid_match:
            return pmid_match.group(1)
        return None
    
    def _save_extractions(self, entities: List[Entity], relations: List[Relation], 
                         document_extractions: Dict[str, Any]):
        """Save extraction results."""
        # Save all entities
        entities_file = self.output_dir / "all_entities.json"
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(e) for e in entities], f, indent=2, ensure_ascii=False)
        
        # Save all relations
        relations_file = self.output_dir / "all_relations.json"
        with open(relations_file, 'w', encoding='utf-8') as f:
            # Convert entities in relations to dicts
            relations_dict = []
            for r in relations:
                rel_dict = asdict(r)
                rel_dict['subject'] = asdict(r.subject)
                rel_dict['object'] = asdict(r.object)
                relations_dict.append(rel_dict)
            json.dump(relations_dict, f, indent=2, ensure_ascii=False)
        
        # Save per-document extractions
        doc_extractions_file = self.output_dir / "document_extractions.json"
        with open(doc_extractions_file, 'w', encoding='utf-8') as f:
            json.dump(document_extractions, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_file = self.output_dir / "extraction_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extraction results to {self.output_dir}")

def main():
    """Main function."""
    print("NER/RE Extraction Module")
    print("=" * 50)
    
    extractor = NERREExtractor()
    entities, relations, document_extractions = extractor.process_all_documents()
    
    print(f"\nExtraction Summary:")
    print(f"  - Documents processed: {extractor.stats['total_documents']}")
    print(f"  - Total entities: {extractor.stats['total_entities']}")
    print(f"  - Total relations: {extractor.stats['total_relations']}")
    print(f"\nResults saved to: {extractor.output_dir}")

if __name__ == "__main__":
    main()

