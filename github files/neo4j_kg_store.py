#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Store
Stores entities and relations in Neo4j with provenance tracking.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from liver_ontology import LiverOntology

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jKGStore:
    """Neo4j Knowledge Graph Store."""
    
    def __init__(self, uri: str = "neo4j://localhost:7687", user: str = "neo4j", 
                 password: str = None, kg_data_dir: str = "kg_data"):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            kg_data_dir: Directory containing KG data
        """
        self.uri = uri
        self.user = user
        self.password = password or os.getenv('NEO4J_PASSWORD')
        self.kg_data_dir = Path(kg_data_dir)
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity early to fail fast if unreachable
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            logger.error(f"Neo4j connectivity verification failed for {uri}: {e}")
            raise
        self.ontology = LiverOntology()
        
        self.stats = {
            'entities_created': 0,
            'relations_created': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def create_constraints(self):
        """Create constraints and indexes."""
        with self.driver.session() as session:
            # Create unique constraints for entities
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DISEASE) REQUIRE d.normalized_name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TREATMENT) REQUIRE t.normalized_name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SYMPTOM) REQUIRE s.normalized_name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PROCEDURE) REQUIRE p.normalized_name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (st:STUDY) REQUIRE st.doi IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (d:DISEASE) ON (d.name)",
                "CREATE INDEX IF NOT EXISTS FOR (t:TREATMENT) ON (t.name)",
                "CREATE INDEX IF NOT EXISTS FOR (s:SYMPTOM) ON (s.name)",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint/index: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint might already exist: {e}")
    
    def create_entity(self, entity_type: str, entity_data: Dict[str, Any], 
                     provenance: Optional[Dict[str, Any]] = None) -> str:
        """
        Create entity in Neo4j.
        
        Args:
            entity_type: Type of entity (DISEASE, TREATMENT, etc.)
            entity_data: Entity properties
            provenance: Provenance information
            
        Returns:
            Entity ID
        """
        # Validate entity
        is_valid, errors = self.ontology.validate_entity(entity_type, entity_data)
        if not is_valid:
            logger.warning(f"Entity validation failed: {errors}")
        
        with self.driver.session() as session:
            # Add provenance to entity data
            if provenance:
                entity_data['provenance'] = json.dumps(provenance)
                entity_data['doi'] = provenance.get('doi')
                entity_data['pmid'] = provenance.get('pmid')
                entity_data['extraction_date'] = provenance.get('extraction_date')
            
            # Create entity
            query = f"""
            MERGE (e:{entity_type} {{normalized_name: $normalized_name}})
            SET e += $properties
            RETURN id(e) as entity_id
            """
            
            properties = {k: v for k, v in entity_data.items() if k != 'normalized_name'}
            result = session.run(query, normalized_name=entity_data.get('normalized_name'), 
                                properties=properties)
            
            record = result.single()
            if record:
                self.stats['entities_created'] += 1
                return str(record['entity_id'])
            return None
    
    def create_relation(self, relation_type: str, subject_id: str, object_id: str,
                       relation_data: Dict[str, Any], provenance: Optional[Dict[str, Any]] = None):
        """
        Create relation in Neo4j.
        
        Args:
            relation_type: Type of relation (TREATS, CAUSES, etc.)
            subject_id: Subject entity ID
            object_id: Object entity ID
            relation_data: Relation properties
            provenance: Provenance information
        """
        with self.driver.session() as session:
            # Add provenance
            if provenance:
                relation_data['provenance'] = json.dumps(provenance)
                relation_data['source'] = provenance.get('source')
                relation_data['model'] = provenance.get('model')
            
            # Create relation
            query = f"""
            MATCH (s) WHERE id(s) = $subject_id
            MATCH (o) WHERE id(o) = $object_id
            MERGE (s)-[r:{relation_type}]->(o)
            SET r += $properties
            RETURN id(r) as relation_id
            """
            
            result = session.run(query, subject_id=int(subject_id), object_id=int(object_id),
                               properties=relation_data)
            
            record = result.single()
            if record:
                self.stats['relations_created'] += 1
                return str(record['relation_id'])
            return None
    
    def load_entities_from_extractions(self, entities_file: str):
        """Load entities from NER/RE extraction results."""
        entities_path = self.kg_data_dir / entities_file
        
        if not entities_path.exists():
            logger.error(f"Entities file not found: {entities_path}")
            return
        
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        logger.info(f"Loading {len(entities)} entities...")
        
        entity_map = {}  # Map normalized_name to entity_id
        
        for entity_data in entities:
            entity_type = entity_data.get('label', 'DISEASE')
            normalized_name = entity_data.get('normalized', entity_data.get('text', '').upper())
            
            # Create entity data
            entity_props = {
                'name': entity_data.get('text'),
                'normalized_name': normalized_name,
                'source': entity_data.get('source'),
                'confidence': entity_data.get('confidence', 0.5)
            }
            
            # Provenance
            provenance = {
                'doi': None,
                'pmid': None,
                'span': entity_data.get('span'),
                'model': 'spacy_en_core_web_sm',
                'confidence': entity_data.get('confidence', 0.5),
                'extraction_date': datetime.now().isoformat(),
                'source': entity_data.get('source')
            }
            
            entity_id = self.create_entity(entity_type, entity_props, provenance)
            if entity_id:
                entity_map[normalized_name] = entity_id
        
        logger.info(f"Created {len(entity_map)} entities")
        return entity_map
    
    def load_relations_from_extractions(self, relations_file: str, entity_map: Dict[str, str]):
        """Load relations from NER/RE extraction results."""
        relations_path = self.kg_data_dir / relations_file
        
        if not relations_path.exists():
            logger.error(f"Relations file not found: {relations_path}")
            return
        
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        
        logger.info(f"Loading {len(relations)} relations...")
        
        for relation_data in relations:
            subject_data = relation_data.get('subject', {})
            object_data = relation_data.get('object', {})
            
            subject_normalized = subject_data.get('normalized', '')
            object_normalized = object_data.get('normalized', '')
            
            subject_id = entity_map.get(subject_normalized)
            object_id = entity_map.get(object_normalized)
            
            if not subject_id or not object_id:
                continue
            
            relation_type = relation_data.get('predicate', 'ASSOCIATED_WITH')
            relation_props = {
                'confidence': relation_data.get('confidence', 0.5),
                'context': relation_data.get('context', ''),
                'model': relation_data.get('model', 'pattern_based')
            }
            
            provenance = {
                'source': relation_data.get('source'),
                'model': relation_data.get('model', 'pattern_based'),
                'confidence': relation_data.get('confidence', 0.5),
                'extraction_date': datetime.now().isoformat()
            }
            
            self.create_relation(relation_type, subject_id, object_id, 
                               relation_props, provenance)
        
        logger.info(f"Created {self.stats['relations_created']} relations")
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities by name.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        with self.driver.session() as session:
            if entity_type:
                cypher_query = f"""
                MATCH (e:{entity_type})
                WHERE e.name CONTAINS $search_term OR e.normalized_name CONTAINS $search_term
                RETURN e, labels(e) as types
                LIMIT $limit
                """
            else:
                cypher_query = """
                MATCH (e)
                WHERE e.name CONTAINS $search_term OR e.normalized_name CONTAINS $search_term
                RETURN e, labels(e) as types
                LIMIT $limit
                """
            
            result = session.run(cypher_query, search_term=query, limit=limit)
            entities = []
            
            for record in result:
                entity = dict(record['e'])
                entity['types'] = record['types']
                entities.append(entity)
            
            return entities
    
    def find_relations(self, entity_name: str, relation_type: Optional[str] = None,
                      direction: str = 'both') -> List[Dict[str, Any]]:
        """
        Find relations for an entity.
        
        Args:
            entity_name: Entity name
            relation_type: Optional relation type filter
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of relations
        """
        with self.driver.session() as session:
            if relation_type:
                if direction == 'outgoing':
                    cypher_query = f"""
                    MATCH (e)-[r:{relation_type}]->(o)
                    WHERE e.name CONTAINS $entity_name OR e.normalized_name CONTAINS $entity_name
                    RETURN e, r, o, type(r) as relation_type
                    """
                elif direction == 'incoming':
                    cypher_query = f"""
                    MATCH (e)<-[r:{relation_type}]-(o)
                    WHERE e.name CONTAINS $entity_name OR e.normalized_name CONTAINS $entity_name
                    RETURN e, r, o, type(r) as relation_type
                    """
                else:
                    cypher_query = f"""
                    MATCH (e)-[r:{relation_type}]-(o)
                    WHERE e.name CONTAINS $entity_name OR e.normalized_name CONTAINS $entity_name
                    RETURN e, r, o, type(r) as relation_type
                    """
            else:
                cypher_query = """
                MATCH (e)-[r]-(o)
                WHERE e.name CONTAINS $entity_name OR e.normalized_name CONTAINS $entity_name
                RETURN e, r, o, type(r) as relation_type
                """
            
            result = session.run(cypher_query, entity_name=entity_name)
            relations = []
            
            for record in result:
                relations.append({
                    'subject': dict(record['e']),
                    'relation': dict(record['r']),
                    'object': dict(record['o']),
                    'relation_type': record['relation_type']
                })
            
            return relations
    
    def path_explain(self, entity1_name: str, entity2_name: str, 
                    max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Find paths between two entities.
        
        Args:
            entity1_name: First entity name
            entity2_name: Second entity name
            max_depth: Maximum path depth
            
        Returns:
            List of paths
        """
        with self.driver.session() as session:
            cypher_query = f"""
            MATCH path = shortestPath((e1)-[*1..{max_depth}]-(e2))
            WHERE e1.name CONTAINS $entity1_name OR e1.normalized_name CONTAINS $entity1_name
            AND e2.name CONTAINS $entity2_name OR e2.normalized_name CONTAINS $entity2_name
            RETURN path
            LIMIT 10
            """
            
            result = session.run(cypher_query, entity1_name=entity1_name, 
                              entity2_name=entity2_name)
            paths = []
            
            for record in result:
                path = record['path']
                path_nodes = [dict(node) for node in path.nodes]
                path_rels = [dict(rel) for rel in path.relationships]
                paths.append({
                    'nodes': path_nodes,
                    'relationships': path_rels,
                    'length': len(path_rels)
                })
            
            return paths
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get KG statistics."""
        with self.driver.session() as session:
            stats_query = """
            MATCH (e)
            RETURN labels(e)[0] as type, count(e) as count
            ORDER BY count DESC
            """
            
            result = session.run(stats_query)
            entity_counts = {record['type']: record['count'] for record in result}
            
            rel_stats_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relation_type, count(r) as count
            ORDER BY count DESC
            """
            
            result = session.run(rel_stats_query)
            relation_counts = {record['relation_type']: record['count'] for record in result}
            
            return {
                'entity_counts': entity_counts,
                'relation_counts': relation_counts,
                'total_entities': sum(entity_counts.values()),
                'total_relations': sum(relation_counts.values())
            }

def main():
    """Main function to load KG data."""
    print("Neo4j Knowledge Graph Store")
    print("=" * 50)
    
    # Initialize store
    kg_store = Neo4jKGStore(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password=None
    )
    
    try:
        # Create constraints
        kg_store.create_constraints()
        
        # Load entities
        entity_map = kg_store.load_entities_from_extractions("all_entities.json")
        
        # Load relations
        kg_store.load_relations_from_extractions("all_relations.json", entity_map)
        
        # Get statistics
        stats = kg_store.get_statistics()
        print(f"\nKnowledge Graph Statistics:")
        print(f"  - Total entities: {stats['total_entities']}")
        print(f"  - Total relations: {stats['total_relations']}")
        print(f"\nEntity counts: {stats['entity_counts']}")
        print(f"Relation counts: {stats['relation_counts']}")
        
    finally:
        kg_store.close()

if __name__ == "__main__":
    main()

