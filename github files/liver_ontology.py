#!/usr/bin/env python3
"""
Liver Disease Ontology/Schema Definition
Defines the schema for liver-related medical knowledge graph.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json

@dataclass
class EntityType:
    """Entity type definition."""
    name: str
    description: str
    properties: List[str]
    required_properties: List[str]

@dataclass
class RelationType:
    """Relation type definition."""
    name: str
    description: str
    subject_types: List[str]
    object_types: List[str]
    properties: List[str]

class LiverOntology:
    """Liver Disease Ontology/Schema."""
    
    def __init__(self):
        self.entity_types = self._define_entity_types()
        self.relation_types = self._define_relation_types()
        self.schema = self._build_schema()
    
    def _define_entity_types(self) -> Dict[str, EntityType]:
        """Define entity types for liver disease knowledge graph."""
        return {
            'DISEASE': EntityType(
                name='DISEASE',
                description='Medical diseases and conditions',
                properties=['name', 'normalized_name', 'icd10_code', 'synonyms', 'description'],
                required_properties=['name', 'normalized_name']
            ),
            'TREATMENT': EntityType(
                name='TREATMENT',
                description='Medical treatments and interventions',
                properties=['name', 'normalized_name', 'type', 'description', 'indications'],
                required_properties=['name', 'normalized_name']
            ),
            'SYMPTOM': EntityType(
                name='SYMPTOM',
                description='Clinical symptoms and signs',
                properties=['name', 'normalized_name', 'severity', 'description'],
                required_properties=['name', 'normalized_name']
            ),
            'ORGAN': EntityType(
                name='ORGAN',
                description='Anatomical organs and structures',
                properties=['name', 'normalized_name', 'anatomical_location', 'description'],
                required_properties=['name', 'normalized_name']
            ),
            'PROCEDURE': EntityType(
                name='PROCEDURE',
                description='Medical procedures and tests',
                properties=['name', 'normalized_name', 'type', 'description', 'indications'],
                required_properties=['name', 'normalized_name']
            ),
            'MEDICATION': EntityType(
                name='MEDICATION',
                description='Medications and drugs',
                properties=['name', 'normalized_name', 'drug_class', 'indications', 'contraindications'],
                required_properties=['name', 'normalized_name']
            ),
            'STUDY': EntityType(
                name='STUDY',
                description='Research studies and papers',
                properties=['title', 'doi', 'pmid', 'authors', 'year', 'journal', 'study_type'],
                required_properties=['title']
            ),
            'CLAIM': EntityType(
                name='CLAIM',
                description='Medical claims and findings',
                properties=['text', 'section', 'confidence', 'evidence_level', 'context'],
                required_properties=['text']
            )
        }
    
    def _define_relation_types(self) -> Dict[str, RelationType]:
        """Define relation types for liver disease knowledge graph."""
        return {
            'TREATS': RelationType(
                name='TREATS',
                description='Treatment treats a disease or condition',
                subject_types=['TREATMENT', 'MEDICATION', 'PROCEDURE'],
                object_types=['DISEASE', 'SYMPTOM'],
                properties=['confidence', 'evidence_level', 'effectiveness']
            ),
            'CAUSES': RelationType(
                name='CAUSES',
                description='Disease or condition causes another',
                subject_types=['DISEASE', 'SYMPTOM'],
                object_types=['DISEASE', 'SYMPTOM'],
                properties=['confidence', 'evidence_level', 'causal_strength']
            ),
            'ASSOCIATED_WITH': RelationType(
                name='ASSOCIATED_WITH',
                description='General association between entities',
                subject_types=['DISEASE', 'SYMPTOM', 'TREATMENT'],
                object_types=['DISEASE', 'SYMPTOM', 'TREATMENT'],
                properties=['confidence', 'evidence_level', 'association_type']
            ),
            'HAS_SYMPTOM': RelationType(
                name='HAS_SYMPTOM',
                description='Disease has a symptom',
                subject_types=['DISEASE'],
                object_types=['SYMPTOM'],
                properties=['confidence', 'frequency', 'severity']
            ),
            'PRESENTS_WITH': RelationType(
                name='PRESENTS_WITH',
                description='Disease presents with symptom',
                subject_types=['DISEASE'],
                object_types=['SYMPTOM'],
                properties=['confidence', 'frequency']
            ),
            'DIAGNOSED_BY': RelationType(
                name='DIAGNOSED_BY',
                description='Disease is diagnosed by procedure',
                subject_types=['DISEASE'],
                object_types=['PROCEDURE'],
                properties=['confidence', 'sensitivity', 'specificity']
            ),
            'DETECTED_BY': RelationType(
                name='DETECTED_BY',
                description='Condition detected by procedure',
                subject_types=['DISEASE', 'SYMPTOM'],
                object_types=['PROCEDURE'],
                properties=['confidence', 'accuracy']
            ),
            'AFFECTS': RelationType(
                name='AFFECTS',
                description='Disease affects an organ',
                subject_types=['DISEASE'],
                object_types=['ORGAN'],
                properties=['confidence', 'severity']
            ),
            'CO_OCCURS_WITH': RelationType(
                name='CO_OCCURS_WITH',
                description='Entities co-occur in same context',
                subject_types=['DISEASE', 'SYMPTOM', 'TREATMENT'],
                object_types=['DISEASE', 'SYMPTOM', 'TREATMENT'],
                properties=['confidence', 'co_occurrence_frequency']
            ),
            'SUPPORTS': RelationType(
                name='SUPPORTS',
                description='Study or claim supports a relation',
                subject_types=['STUDY', 'CLAIM'],
                object_types=['DISEASE', 'TREATMENT', 'SYMPTOM'],
                properties=['confidence', 'evidence_level', 'support_strength']
            ),
            'MENTIONED_IN': RelationType(
                name='MENTIONED_IN',
                description='Entity mentioned in study',
                subject_types=['DISEASE', 'TREATMENT', 'SYMPTOM', 'PROCEDURE'],
                object_types=['STUDY'],
                properties=['frequency', 'section', 'context']
            )
        }
    
    def _build_schema(self) -> Dict[str, Any]:
        """Build complete schema definition."""
        return {
            'ontology_name': 'Liver Disease Knowledge Graph Ontology',
            'version': '1.0',
            'entity_types': {name: {
                'name': et.name,
                'description': et.description,
                'properties': et.properties,
                'required_properties': et.required_properties
            } for name, et in self.entity_types.items()},
            'relation_types': {name: {
                'name': rt.name,
                'description': rt.description,
                'subject_types': rt.subject_types,
                'object_types': rt.object_types,
                'properties': rt.properties
            } for name, rt in self.relation_types.items()},
            'shacl_constraints': self._generate_shacl_constraints()
        }
    
    def _generate_shacl_constraints(self) -> Dict[str, Any]:
        """Generate SHACL constraints for validation."""
        return {
            'entity_constraints': {
                'DISEASE': {
                    'required': ['name', 'normalized_name'],
                    'property_constraints': {
                        'name': {'minLength': 1, 'maxLength': 200},
                        'normalized_name': {'pattern': '^[A-Z_]+$'}
                    }
                },
                'TREATMENT': {
                    'required': ['name', 'normalized_name'],
                    'property_constraints': {
                        'name': {'minLength': 1, 'maxLength': 200}
                    }
                }
            },
            'relation_constraints': {
                'TREATS': {
                    'required': ['confidence'],
                    'property_constraints': {
                        'confidence': {'min': 0.0, 'max': 1.0}
                    }
                }
            }
        }
    
    def validate_entity(self, entity_type: str, entity_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate entity against schema."""
        if entity_type not in self.entity_types:
            return False, [f"Unknown entity type: {entity_type}"]
        
        errors = []
        et = self.entity_types[entity_type]
        
        # Check required properties
        for prop in et.required_properties:
            if prop not in entity_data:
                errors.append(f"Missing required property: {prop}")
        
        return len(errors) == 0, errors
    
    def validate_relation(self, relation_type: str, subject_type: str, 
                         object_type: str, relation_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate relation against schema."""
        if relation_type not in self.relation_types:
            return False, [f"Unknown relation type: {relation_type}"]
        
        errors = []
        rt = self.relation_types[relation_type]
        
        # Check subject type compatibility
        if subject_type not in rt.subject_types:
            errors.append(f"Subject type {subject_type} not allowed for relation {relation_type}")
        
        # Check object type compatibility
        if object_type not in rt.object_types:
            errors.append(f"Object type {object_type} not allowed for relation {relation_type}")
        
        return len(errors) == 0, errors
    
    def save_schema(self, filepath: str):
        """Save schema to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.schema, f, indent=2, ensure_ascii=False)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema definition."""
        return self.schema

