#!/usr/bin/env python3
"""
RAG Orchestrator
Hybrid retrieval with reranking, evidence pack generation, and claim-checking.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone_enhanced import EnhancedPineconeManager
from neo4j_kg_store import Neo4jKGStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """RAG Orchestrator with hybrid retrieval and reranking."""
    
    def __init__(self, 
                 pinecone_manager: EnhancedPineconeManager,
                 neo4j_store: Optional[Neo4jKGStore] = None,
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize RAG Orchestrator.
        
        Args:
            pinecone_manager: Enhanced Pinecone manager
            neo4j_store: Neo4j knowledge graph store (optional)
            rerank_model: Cross-encoder model for reranking
        """
        self.pinecone_manager = pinecone_manager
        self.neo4j_store = neo4j_store
        
        # Load reranker
        logger.info(f"Loading reranker model: {rerank_model}")
        try:
            self.reranker = CrossEncoder(rerank_model)
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}. Continuing without reranking.")
            self.reranker = None
        
        logger.info("RAG Orchestrator initialized")
    
    def hybrid_retrieve(self, query: str, top_k: int = 20,
                       namespaces: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Hybrid retrieval from multiple namespaces.
        
        Args:
            query: Search query
            top_k: Number of results per namespace
            namespaces: List of namespaces to search (default: all)
            
        Returns:
            Dictionary mapping namespace to results
        """
        if namespaces is None:
            namespaces = ['papers', 'triples', 'claims']
        
        results = {}
        
        for namespace in namespaces:
            logger.info(f"Searching namespace '{namespace}'...")
            try:
                ns_results = self.pinecone_manager.search(query, namespace, top_k=top_k)
                results[namespace] = ns_results
            except Exception as e:
                logger.warning(f"Error searching namespace '{namespace}': {e}")
                results[namespace] = []
        
        return results
    
    def rerank(self, query: str, results: List[Dict[str, Any]], 
              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        # Prepare pairs for reranking
        pairs = []
        for result in results:
            # Get text from metadata
            text = result.get('metadata', {}).get('content', '') or \
                   result.get('metadata', {}).get('text', '') or \
                   result.get('metadata', {}).get('triple_text', '')
            pairs.append([query, text])
        
        # Rerank
        if self.reranker:
            try:
                scores = self.reranker.predict(pairs)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using original order.")
                scores = [0.5] * len(pairs)
        else:
            scores = [0.5] * len(pairs)
        
        # Add scores and sort
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return reranked[:top_k]
    
    def create_evidence_pack(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Create evidence pack with hybrid retrieval and reranking.
        
        Args:
            query: Search query
            top_k: Number of top results
            
        Returns:
            Evidence pack dictionary
        """
        # Hybrid retrieval
        hybrid_results = self.hybrid_retrieve(query, top_k=top_k * 2)
        
        # Combine results from all namespaces
        all_results = []
        for namespace, results in hybrid_results.items():
            for result in results:
                result['namespace'] = namespace
                all_results.append(result)
        
        # Rerank combined results
        reranked = self.rerank(query, all_results, top_k=top_k)
        
        # Get KG context (only if Neo4j is available)
        kg_context = {}
        if self.neo4j_store:
            try:
                kg_context = self._get_kg_context(query)
            except Exception as e:
                logger.warning(f"Could not get KG context: {e}")
                kg_context = {'entities': [], 'relations': []}
        else:
            kg_context = {'entities': [], 'relations': []}
        
        # Create evidence pack
        evidence_pack = {
            'query': query,
            'top_results': reranked,
            'kg_context': kg_context,
            'total_results': len(all_results),
            'top_k': top_k
        }
        
        return evidence_pack
    
    def claim_check(self, claim: str, evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check claim against evidence pack and knowledge graph.
        
        Args:
            claim: Claim to check
            evidence_pack: Evidence pack from retrieval
            kg_context: Knowledge graph context
            
        Returns:
            Claim checking results
        """
        # Check against retrieved evidence
        evidence_support = []
        evidence_contradict = []
        
        for result in evidence_pack.get('top_results', []):
            text = result.get('metadata', {}).get('content', '') or \
                   result.get('metadata', {}).get('text', '')
            
            # Simple similarity check (can be enhanced with NLI model)
            if self._check_support(claim, text):
                evidence_support.append({
                    'text': text,
                    'score': result.get('rerank_score', result.get('score', 0)),
                    'source': result.get('metadata', {}).get('source', ''),
                    'namespace': result.get('namespace', '')
                })
            elif self._check_contradict(claim, text):
                evidence_contradict.append({
                    'text': text,
                    'score': result.get('rerank_score', result.get('score', 0)),
                    'source': result.get('metadata', {}).get('source', ''),
                    'namespace': result.get('namespace', '')
                })
        
        # Check against KG
        kg_verification = self._verify_against_kg(claim, evidence_pack.get('kg_context', {}))
        
        # Determine claim status
        status = 'SUPPORTED' if evidence_support else 'UNSUPPORTED'
        if evidence_contradict:
            status = 'CONTRADICTED'
        
        return {
            'claim': claim,
            'status': status,
            'evidence_support': evidence_support,
            'evidence_contradict': evidence_contradict,
            'kg_verification': kg_verification,
            'confidence': self._calculate_confidence(evidence_support, evidence_contradict)
        }
    
    def _get_kg_context(self, query: str) -> Dict[str, Any]:
        """Get knowledge graph context for query."""
        if not self.neo4j_store:
            return {'entities': [], 'relations': []}
        
        try:
            # Search entities in KG
            entities = self.neo4j_store.search_entities(query, limit=5)
            
            # Find relations
            relations = []
            for entity in entities:
                entity_name = entity.get('name', '')
                if entity_name:
                    try:
                        rels = self.neo4j_store.find_relations(entity_name, limit=5)
                        relations.extend(rels)
                    except Exception as e:
                        logger.warning(f"Error finding relations for {entity_name}: {e}")
                        continue
            
            return {
                'entities': entities,
                'relations': relations[:10]  # Limit relations
            }
        except Exception as e:
            logger.warning(f"Error getting KG context: {e}")
            return {'entities': [], 'relations': []}
    
    def _check_support(self, claim: str, evidence: str) -> bool:
        """Check if evidence supports claim (simplified)."""
        # Simple keyword overlap check
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        
        overlap = len(claim_words & evidence_words)
        return overlap >= 3  # At least 3 common words
    
    def _check_contradict(self, claim: str, evidence: str) -> bool:
        """Check if evidence contradicts claim (simplified)."""
        # Simple negation detection
        negation_words = ['not', 'no', 'never', 'none', 'without', 'lacks']
        evidence_lower = evidence.lower()
        
        for neg_word in negation_words:
            if neg_word in evidence_lower and any(word in evidence_lower for word in claim.lower().split()[:3]):
                return True
        
        return False
    
    def _verify_against_kg(self, claim: str, kg_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify claim against knowledge graph."""
        # Extract entities from claim
        entities = kg_context.get('entities', [])
        relations = kg_context.get('relations', [])
        
        return {
            'entities_found': len(entities),
            'relations_found': len(relations),
            'verification_status': 'PARTIAL' if entities else 'NO_MATCH'
        }
    
    def _calculate_confidence(self, support: List[Dict], contradict: List[Dict]) -> float:
        """Calculate confidence score."""
        total_evidence = len(support) + len(contradict)
        if total_evidence == 0:
            return 0.0
        
        support_score = sum(s.get('score', 0) for s in support)
        contradict_score = sum(c.get('score', 0) for c in contradict)
        
        if support_score > contradict_score:
            return min(1.0, support_score / total_evidence)
        else:
            return 0.0
    
    def answer_with_evidence(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate answer with evidence (to be used by MCP server).
        
        Args:
            query: Query to answer
            top_k: Number of top results
            
        Returns:
            Answer with evidence
        """
        # Create evidence pack
        evidence_pack = self.create_evidence_pack(query, top_k=top_k)
        
        # Format answer
        answer = {
            'query': query,
            'answer': self._generate_answer_text(query, evidence_pack),
            'evidence': evidence_pack['top_results'],
            'kg_context': evidence_pack['kg_context'],
            'sources': self._extract_sources(evidence_pack['top_results'])
        }
        
        return answer
    
    def _generate_answer_text(self, query: str, evidence_pack: Dict[str, Any]) -> str:
        """Generate answer text from evidence (simplified)."""
        top_results = evidence_pack.get('top_results', [])
        
        if not top_results:
            return "I couldn't find relevant information to answer your query."
        
        # Combine top evidence
        answer_parts = []
        for result in top_results[:3]:  # Top 3 results
            text = result.get('metadata', {}).get('content', '') or \
                   result.get('metadata', {}).get('text', '') or \
                   result.get('metadata', {}).get('triple_text', '')
            if text:
                answer_parts.append(text[:200])  # Truncate
        
        return " ".join(answer_parts)
    
    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from results."""
        sources = set()
        for result in results:
            source = result.get('metadata', {}).get('source', '')
            if source:
                sources.add(source)
        return list(sources)

