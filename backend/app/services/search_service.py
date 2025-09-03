"""
Search service for BM25 sparse retrieval using Elasticsearch.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import re
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.core.config import settings


class SearchService:
    """Elasticsearch-based BM25 search service."""
    
    def __init__(self):
        self.client = None
        self.index_name = "questions"
        
    async def _get_client(self) -> AsyncElasticsearch:
        """Get Elasticsearch client (lazy initialization)."""
        if self.client is None:
            self.client = AsyncElasticsearch([settings.SEARCH_ENGINE_URL])
        return self.client
    
    async def initialize_index(self) -> bool:
        """Initialize Elasticsearch index with proper mappings."""
        try:
            client = await self._get_client()
            
            # Check if index exists
            if await client.indices.exists(index=self.index_name):
                return True
            
            # Create index with mappings
            mapping = {
                "mappings": {
                    "properties": {
                        "item_id": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "lang": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        
                        # Text fields for search
                        "stem": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "passage": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "solution": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        
                        # Skill and tag fields
                        "skills": {"type": "keyword"},
                        "error_tags": {"type": "keyword"},
                        "topic": {"type": "keyword"},
                        
                        # Numeric fields
                        "difficulty_a": {"type": "float"},
                        "difficulty_b": {"type": "float"},
                        "bloom_level": {"type": "keyword"},
                        "level_cefr": {"type": "keyword"},
                        
                        # Metadata
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "turkish_analyzer": {
                                "type": "standard",
                                "stopwords": ["ve", "ile", "bir", "bu", "şu", "o"]
                            },
                            "english_analyzer": {
                                "type": "standard",
                                "stopwords": "_english_"
                            }
                        }
                    }
                }
            }
            
            await client.indices.create(index=self.index_name, body=mapping)
            print(f"Created index: {self.index_name}")
            return True
            
        except Exception as e:
            print(f"Error initializing index: {e}")
            return False
    
    def _prepare_document(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare document for indexing."""
        doc = {
            "item_id": item_data.get("id"),
            "type": item_data.get("type", "unknown"),
            "lang": item_data.get("lang", "tr"),
            "status": item_data.get("status", "active"),
            "created_at": item_data.get("created_at"),
            "updated_at": item_data.get("updated_at")
        }
        
        # Add type-specific fields
        if item_data.get("type") == "math":
            doc.update({
                "stem": item_data.get("stem", ""),
                "solution": item_data.get("solution", ""),
                "skills": item_data.get("skills", []),
                "difficulty_a": item_data.get("difficulty_a", 1.0),
                "difficulty_b": item_data.get("difficulty_b", 0.0),
                "bloom_level": item_data.get("bloom_level"),
                "topic": item_data.get("topic")
            })
        elif item_data.get("type") == "english":
            doc.update({
                "passage": item_data.get("passage", ""),
                "error_tags": item_data.get("error_tags", []),
                "level_cefr": item_data.get("level_cefr"),
                "topic": item_data.get("topic")
            })
        
        return doc
    
    async def index_item(self, item_data: Dict[str, Any]) -> bool:
        """Index single item."""
        try:
            client = await self._get_client()
            doc = self._prepare_document(item_data)
            
            await client.index(
                index=self.index_name,
                id=doc["item_id"],
                body=doc
            )
            
            return True
        except Exception as e:
            print(f"Error indexing item {item_data.get('id')}: {e}")
            return False
    
    async def index_items_batch(self, items: List[Dict[str, Any]]) -> int:
        """Index multiple items in batch."""
        try:
            client = await self._get_client()
            
            # Prepare documents for bulk indexing
            actions = []
            for item in items:
                doc = self._prepare_document(item)
                action = {
                    "_index": self.index_name,
                    "_id": doc["item_id"],
                    "_source": doc
                }
                actions.append(action)
            
            # Bulk index
            success_count, failed_items = await async_bulk(
                client, actions, chunk_size=100
            )
            
            return success_count
        except Exception as e:
            print(f"Error bulk indexing: {e}")
            return 0
    
    async def search(
        self,
        query: str,
        item_type: Optional[str] = None,
        lang: str = "tr",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search items using BM25 scoring.
        
        Args:
            query: Search query
            item_type: "math" or "english"
            lang: Language code
            filters: Additional filters
            limit: Maximum results
            offset: Result offset for pagination
            
        Returns:
            Search results with BM25 scores
        """
        try:
            client = await self._get_client()
            
            # Build query
            search_query = {
                "bool": {
                    "must": [],
                    "filter": []
                }
            }
            
            # Add text search
            if query.strip():
                if item_type == "math":
                    text_query = {
                        "multi_match": {
                            "query": query,
                            "fields": ["stem^2", "solution", "skills"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    }
                elif item_type == "english":
                    text_query = {
                        "multi_match": {
                            "query": query,
                            "fields": ["passage^2", "error_tags"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    }
                else:
                    text_query = {
                        "multi_match": {
                            "query": query,
                            "fields": ["stem", "passage", "solution", "skills", "error_tags"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    }
                search_query["bool"]["must"].append(text_query)
            else:
                # Match all if no query
                search_query["bool"]["must"].append({"match_all": {}})
            
            # Add filters
            if item_type:
                search_query["bool"]["filter"].append({"term": {"type": item_type}})
            
            search_query["bool"]["filter"].append({"term": {"lang": lang}})
            search_query["bool"]["filter"].append({"term": {"status": "active"}})
            
            # Add custom filters
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        search_query["bool"]["filter"].append({"terms": {key: value}})
                    elif isinstance(value, dict) and "range" in value:
                        search_query["bool"]["filter"].append({"range": {key: value["range"]}})
                    else:
                        search_query["bool"]["filter"].append({"term": {key: value}})
            
            # Execute search
            response = await client.search(
                index=self.index_name,
                body={
                    "query": search_query,
                    "size": limit,
                    "from": offset,
                    "sort": [{"_score": {"order": "desc"}}]
                }
            )
            
            # Format results
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "item_id": hit["_id"],
                    "score": hit["_score"],
                    "metadata": hit["_source"]
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    async def search_by_skills(
        self,
        skills: List[str],
        item_type: str,
        lang: str = "tr",
        difficulty_range: Optional[Tuple[float, float]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search items by skills with BM25 scoring.
        
        Args:
            skills: List of skill tags
            item_type: "math" or "english"
            lang: Language code
            difficulty_range: (min_difficulty, max_difficulty) for math items
            limit: Maximum results
            
        Returns:
            Search results
        """
        filters = {}
        
        # Add skill filters
        if skills:
            if item_type == "math":
                filters["skills"] = skills
            elif item_type == "english":
                filters["error_tags"] = skills
        
        # Add difficulty range for math items
        if difficulty_range and item_type == "math":
            min_diff, max_diff = difficulty_range
            filters["difficulty_b"] = {
                "range": {"gte": min_diff, "lte": max_diff}
            }
        
        # Create query from skills
        query = " ".join(skills) if skills else ""
        
        return await self.search(
            query=query,
            item_type=item_type,
            lang=lang,
            filters=filters,
            limit=limit
        )
    
    async def suggest_completions(
        self,
        partial_query: str,
        item_type: Optional[str] = None,
        lang: str = "tr",
        limit: int = 5
    ) -> List[str]:
        """Get search suggestions/completions."""
        try:
            client = await self._get_client()
            
            # Use completion suggester or simple prefix search
            field = "stem" if item_type == "math" else "passage"
            
            query = {
                "bool": {
                    "must": [
                        {
                            "prefix": {
                                f"{field}.keyword": partial_query
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"lang": lang}},
                        {"term": {"status": "active"}}
                    ]
                }
            }
            
            if item_type:
                query["bool"]["filter"].append({"term": {"type": item_type}})
            
            response = await client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": limit,
                    "_source": [field]
                }
            )
            
            suggestions = []
            for hit in response["hits"]["hits"]:
                text = hit["_source"].get(field, "")
                if text and text not in suggestions:
                    suggestions.append(text)
            
            return suggestions[:limit]
            
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []
    
    async def delete_item(self, item_id: str) -> bool:
        """Delete item from search index."""
        try:
            client = await self._get_client()
            await client.delete(index=self.index_name, id=item_id)
            return True
        except Exception as e:
            print(f"Error deleting item {item_id}: {e}")
            return False
    
    async def update_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update item in search index."""
        try:
            client = await self._get_client()
            await client.update(
                index=self.index_name,
                id=item_id,
                body={"doc": updates}
            )
            return True
        except Exception as e:
            print(f"Error updating item {item_id}: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        try:
            client = await self._get_client()
            stats = await client.indices.stats(index=self.index_name)
            
            return {
                "total_docs": stats["indices"][self.index_name]["total"]["docs"]["count"],
                "index_size": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "search_time": stats["indices"][self.index_name]["total"]["search"]["query_time_in_millis"],
                "search_count": stats["indices"][self.index_name]["total"]["search"]["query_total"]
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}


# Create service instance
search_service = SearchService()