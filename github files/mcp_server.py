#!/usr/bin/env python3
"""
MCP Server for Knowledge Graph and RAG
Provides functions for KG search, relation finding, path explanation, RAG answering, and provenance.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from rag_orchestrator import RAGOrchestrator
from neo4j_kg_store import Neo4jKGStore
from pinecone_enhanced import EnhancedPineconeManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPServer:
    """MCP Server providing KG and RAG functions."""
    
    def __init__(self, 
                 neo4j_uri: str = "neo4j://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: Optional[str] = None):
        """
        Initialize MCP Server.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')

        # Initialize components
        self.pinecone_manager = EnhancedPineconeManager()
        
        # Initialize Neo4j (handle failures gracefully)
        try:
            self.neo4j_store = Neo4jKGStore(neo4j_uri, neo4j_user, neo4j_password)
            self.neo4j_available = True
            logger.info("Neo4j connection successful")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Continuing without KG features.")
            self.neo4j_store = None
            self.neo4j_available = False
        
        # Initialize RAG orchestrator (works with or without Neo4j)
        self.rag_orchestrator = RAGOrchestrator(self.pinecone_manager, self.neo4j_store)
        
        logger.info("MCP Server initialized")
    
    def kg_search_graph(self, query: str, entity_type: Optional[str] = None,
                       limit: int = 10) -> Dict[str, Any]:
        """
        Search knowledge graph for entities.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum results
            
        Returns:
            Search results with entities and metadata
        """
        logger.info(f"Searching KG for: {query}")
        
        if not self.neo4j_available or not self.neo4j_store:
            return {
                'query': query,
                'entity_type': entity_type,
                'results': [],
                'total': 0,
                'error': 'Neo4j not available'
            }
        
        entities = self.neo4j_store.search_entities(query, entity_type, limit)
        
        # Format results
        results = {
            'query': query,
            'entity_type': entity_type,
            'results': [
                {
                    'name': e.get('name', ''),
                    'normalized_name': e.get('normalized_name', ''),
                    'types': e.get('types', []),
                    'properties': {k: v for k, v in e.items() if k not in ['types']}
                }
                for e in entities
            ],
            'total': len(entities)
        }
        
        return results
    
    def kg_find_relations(self, entity_name: str, relation_type: Optional[str] = None,
                         direction: str = 'both', limit: int = 20) -> Dict[str, Any]:
        """
        Find relations for an entity.
        
        Args:
            entity_name: Entity name
            relation_type: Optional relation type filter
            direction: 'outgoing', 'incoming', or 'both'
            limit: Maximum results
            
        Returns:
            Relations with entities and metadata
        """
        logger.info(f"Finding relations for: {entity_name}")
        
        if not self.neo4j_available or not self.neo4j_store:
            return {
                'entity': entity_name,
                'relation_type': relation_type,
                'direction': direction,
                'relations': [],
                'total': 0,
                'error': 'Neo4j not available'
            }
        
        relations = self.neo4j_store.find_relations(entity_name, relation_type, direction)
        
        # Limit results
        relations = relations[:limit]
        
        # Format results
        results = {
            'entity': entity_name,
            'relation_type': relation_type,
            'direction': direction,
            'relations': [
                {
                    'subject': {
                        'name': r['subject'].get('name', ''),
                        'normalized_name': r['subject'].get('normalized_name', ''),
                        'types': r['subject'].get('types', [])
                    },
                    'relation': {
                        'type': r['relation_type'],
                        'properties': r['relation']
                    },
                    'object': {
                        'name': r['object'].get('name', ''),
                        'normalized_name': r['object'].get('normalized_name', ''),
                        'types': r['object'].get('types', [])
                    }
                }
                for r in relations
            ],
            'total': len(relations)
        }
        
        return results
    
    def kg_path_explain(self, entity1_name: str, entity2_name: str,
                       max_depth: int = 3) -> Dict[str, Any]:
        """
        Find and explain paths between two entities.
        
        Args:
            entity1_name: First entity name
            entity2_name: Second entity name
            max_depth: Maximum path depth
            
        Returns:
            Paths with explanations
        """
        logger.info(f"Finding paths between: {entity1_name} and {entity2_name}")
        
        if not self.neo4j_available or not self.neo4j_store:
            return {
                'entity1': entity1_name,
                'entity2': entity2_name,
                'paths': [],
                'total': 0,
                'error': 'Neo4j not available'
            }
        
        paths = self.neo4j_store.path_explain(entity1_name, entity2_name, max_depth)
        
        # Format paths
        formatted_paths = []
        for path in paths:
            nodes = path.get('nodes', [])
            rels = path.get('relationships', [])
            
            path_explanation = []
            for i, node in enumerate(nodes):
                path_explanation.append(node.get('name', ''))
                if i < len(rels):
                    path_explanation.append(f"--[{rels[i].get('type', 'RELATED_TO')}]-->")
            
            formatted_paths.append({
                'path': path_explanation,
                'length': path.get('length', 0),
                'nodes': [{'name': n.get('name', ''), 'types': n.get('types', [])} 
                         for n in nodes],
                'relationships': [{'type': r.get('type', ''), 'properties': r} 
                                for r in rels]
            })
        
        results = {
            'entity1': entity1_name,
            'entity2': entity2_name,
            'max_depth': max_depth,
            'paths': formatted_paths,
            'total_paths': len(formatted_paths)
        }
        
        return results
    
    def rag_answer_with_evidence(self, query: str, top_k: int = 10,
                                include_kg: bool = True) -> Dict[str, Any]:
        """
        Generate answer with evidence using RAG.
        
        Args:
            query: Query to answer
            top_k: Number of top results
            include_kg: Whether to include KG context
            
        Returns:
            Answer with evidence and sources
        """
        logger.info(f"Answering query with evidence: {query}")
        
        answer = self.rag_orchestrator.answer_with_evidence(query, top_k)
        
        # Add KG context if requested
        if include_kg:
            kg_context = self.rag_orchestrator._get_kg_context(query)
            answer['kg_context'] = kg_context
        
        return answer
    
    def audit_provenance(self, entity_name: Optional[str] = None,
                        relation_id: Optional[str] = None,
                        source: Optional[str] = None) -> Dict[str, Any]:
        """
        Audit provenance for entities, relations, or sources.
        
        Args:
            entity_name: Entity name to audit
            relation_id: Relation ID to audit
            source: Source document to audit
            
        Returns:
            Provenance information
        """
        logger.info(f"Auditing provenance: entity={entity_name}, relation={relation_id}, source={source}")
        
        provenance_data = {
            'entities': [],
            'relations': [],
            'sources': []
        }
        
        # Audit entity provenance
        if entity_name:
            entities = self.neo4j_store.search_entities(entity_name, limit=10)
            for entity in entities:
                provenance = entity.get('provenance', {})
                if isinstance(provenance, str):
                    provenance = json.loads(provenance)
                
                provenance_data['entities'].append({
                    'name': entity.get('name', ''),
                    'normalized_name': entity.get('normalized_name', ''),
                    'doi': entity.get('doi'),
                    'pmid': entity.get('pmid'),
                    'source': entity.get('source', ''),
                    'extraction_date': entity.get('extraction_date'),
                    'confidence': entity.get('confidence'),
                    'provenance': provenance
                })
        
        # Audit relation provenance
        if relation_id or entity_name:
            if entity_name:
                relations = self.neo4j_store.find_relations(entity_name, limit=20)
            else:
                relations = []
            
            for relation in relations:
                rel_props = relation.get('relation', {})
                provenance = rel_props.get('provenance', {})
                if isinstance(provenance, str):
                    provenance = json.loads(provenance)
                
                provenance_data['relations'].append({
                    'subject': relation.get('subject', {}).get('name', ''),
                    'object': relation.get('object', {}).get('name', ''),
                    'relation_type': relation.get('relation_type', ''),
                    'source': rel_props.get('source', ''),
                    'model': rel_props.get('model', ''),
                    'confidence': rel_props.get('confidence'),
                    'provenance': provenance
                })
        
        # Audit source provenance
        if source:
            # Search in Pinecone for source
            for namespace in ['papers', 'triples', 'claims']:
                results = self.pinecone_manager.search(
                    source, namespace, top_k=10,
                    filter_dict={'source': source} if source else None
                )
                
                for result in results:
                    metadata = result.get('metadata', {})
                    provenance_data['sources'].append({
                        'source': metadata.get('source', ''),
                        'namespace': namespace,
                        'year': metadata.get('year'),
                        'upload_timestamp': metadata.get('upload_timestamp'),
                        'content_preview': metadata.get('content', '')[:200] or 
                                          metadata.get('text', '')[:200]
                    })
        
        return provenance_data
    
    def close(self):
        """Close connections."""
        if self.neo4j_store and self.neo4j_available:
            try:
                self.neo4j_store.close()
            except:
                pass
        logger.info("MCP Server closed")

def main():
    """Test MCP Server functions."""
    print("MCP Server Test")
    print("=" * 50)
    
    server = MCPServer()
    
    try:
        # Test KG search
        print("\n1. Testing kg.search_graph...")
        results = server.kg_search_graph("cirrhosis", limit=5)
        print(f"Found {results['total']} entities")
        
        # Test relation finding
        print("\n2. Testing kg.find_relations...")
        relations = server.kg_find_relations("cirrhosis", limit=5)
        print(f"Found {relations['total']} relations")
        
        # Test path explanation
        print("\n3. Testing kg.path_explain...")
        paths = server.kg_path_explain("cirrhosis", "portal vein thrombosis", max_depth=2)
        print(f"Found {paths['total_paths']} paths")
        
        # Test RAG answer
        print("\n4. Testing rag.answer_with_evidence...")
        answer = server.rag_answer_with_evidence("What is portal vein thrombosis?", top_k=5)
        print(f"Answer generated with {len(answer['evidence'])} evidence items")
        
        # Test provenance audit
        print("\n5. Testing audit.provenance...")
        provenance = server.audit_provenance(entity_name="cirrhosis")
        print(f"Provenance data retrieved")
        
    finally:
        server.close()

if __name__ == "__main__":
    main()

