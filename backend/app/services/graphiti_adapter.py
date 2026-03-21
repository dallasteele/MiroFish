"""
Graphiti Adapter — drop-in replacement for Zep Cloud client.

Uses graphiti-core + FalkorDB (localhost:6380) for local knowledge graphs.
Entity extraction runs through Qwen 2.5:7b at localhost:11434/v1 (FREE).

Mirrors the interface expected by zep_entity_reader, zep_graph_memory_updater,
and zep_tools so callers can switch backends via Config.USE_GRAPHITI.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.driver.falkordb_driver import FalkorDriver

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.graphiti_adapter")


# ---------------------------------------------------------------------------
# Shared dataclasses — match zep_entity_reader interfaces exactly
# ---------------------------------------------------------------------------


@dataclass
class EntityNode:
    """Entity node — mirrors zep_entity_reader.EntityNode."""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Return the first non-generic label, or None."""
        for label in self.labels:
            if label not in ("Entity", "Node"):
                return label
        return None


@dataclass
class FilteredEntities:
    """Filtered entity collection — mirrors zep_entity_reader.FilteredEntities."""

    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


@dataclass
class SearchResult:
    """Search result — mirrors zep_tools.SearchResult."""

    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        parts = [
            "Search query: " + self.query,
            "Found " + str(self.total_count) + " relevant results",
        ]
        if self.facts:
            parts.append("\n### Relevant facts:")
            for i, fact in enumerate(self.facts, 1):
                parts.append(str(i) + ". " + fact)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_falkor_driver() -> FalkorDriver:
    """Build a FalkorDB driver from config."""
    uri = Config.FALKORDB_URI  # e.g. bolt://localhost:6380
    # Parse host/port from bolt URI: bolt://host:port
    host = "localhost"
    port = 6380
    try:
        without_scheme = uri.split("://", 1)[-1]
        parts = without_scheme.split(":")
        host = parts[0]
        if len(parts) > 1:
            port = int(parts[1])
    except Exception:
        logger.warning("Could not parse FALKORDB_URI '%s', using defaults", uri)

    return FalkorDriver(host=host, port=port)


def _make_llm_client() -> OpenAIClient:
    """Build an OpenAI-compatible LLM client pointing at Ollama."""
    cfg = LLMConfig(
        api_key=Config.LLM_API_KEY or "ollama",
        model=Config.LLM_MODEL_NAME or "qwen2.5:7b",
        base_url=Config.LLM_BASE_URL or "http://localhost:11434/v1",
    )
    return OpenAIClient(config=cfg)


def _graphiti_entity_to_node(gnode: Any) -> EntityNode:
    """Convert a graphiti EntityNode to our EntityNode dataclass."""
    labels = list(getattr(gnode, "labels", []))
    if "Entity" not in labels:
        labels.insert(0, "Entity")
    return EntityNode(
        uuid=str(gnode.uuid),
        name=gnode.name or "",
        labels=labels,
        summary=gnode.summary or "",
        attributes=dict(gnode.attributes or {}),
    )


def _graphiti_edge_to_dict(gedge: Any) -> Dict[str, Any]:
    """Convert a graphiti EntityEdge to a plain dict."""
    return {
        "uuid": str(gedge.uuid),
        "name": gedge.name or "",
        "fact": gedge.fact or "",
        "source_node_uuid": str(gedge.source_node_uuid),
        "target_node_uuid": str(gedge.target_node_uuid),
        "attributes": dict(gedge.attributes or {}),
    }


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------


class GraphitiAdapter:
    """
    Drop-in replacement for the Zep Cloud client used in MiroFish.

    Wraps graphiti-core with FalkorDB as the graph store and Qwen 2.5:7b
    (via Ollama) as the entity-extraction LLM.

    All async methods must be awaited.  Synchronous callers can use the
    helper `run_sync()` on any coroutine.
    """

    def __init__(self) -> None:
        self._driver = _make_falkor_driver()
        self._llm_client = _make_llm_client()
        self._graphiti: Optional[Graphiti] = None
        self._initialized = False

    async def _get_graphiti(self) -> Graphiti:
        """Lazily initialise and cache the Graphiti instance."""
        if self._graphiti is None:
            self._graphiti = Graphiti(
                uri=Config.FALKORDB_URI or "bolt://localhost:6380",
                user="",
                password="",
                llm_client=self._llm_client,
                graph_driver=self._driver,
            )
            if not self._initialized:
                try:
                    await self._graphiti.build_indices_and_constraints()
                    self._initialized = True
                    logger.info("Graphiti indices built on FalkorDB")
                except Exception as exc:
                    logger.warning("Could not build Graphiti indices: %s", exc)
                    self._initialized = True  # don't retry on every call
        return self._graphiti

    # ------------------------------------------------------------------
    # Core write: add_episode
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        graph_id: str,
        episode_type: str = "text",
        data: str = "",
        name: Optional[str] = None,
        source_description: str = "mirofish",
        reference_time: Optional[datetime] = None,
    ) -> None:
        """
        Add a text episode to the knowledge graph.

        Args:
            graph_id: Logical graph identifier (used as group_id in Graphiti).
            episode_type: One of "text", "json", "message".
            data: The episode content.
            name: Optional episode name; auto-generated if omitted.
            source_description: Human-readable description of the data source.
            reference_time: Timestamp for the episode (defaults to now).
        """
        graphiti = await self._get_graphiti()

        ep_name = name or ("episode-" + str(uuid.uuid4())[:8])
        ref_time = reference_time or datetime.now(timezone.utc)

        type_map = {
            "text": EpisodeType.text,
            "json": EpisodeType.json,
            "message": EpisodeType.message,
        }
        ep_type = type_map.get(episode_type, EpisodeType.text)

        await graphiti.add_episode(
            name=ep_name,
            episode_body=data,
            source_description=source_description,
            reference_time=ref_time,
            source=ep_type,
            group_id=graph_id,
        )
        logger.debug("add_episode: group_id=%s name=%s", graph_id, ep_name)

    # ------------------------------------------------------------------
    # Core read: search
    # ------------------------------------------------------------------

    async def search(
        self,
        graph_id: str,
        query: str,
        num_results: int = 10,
    ) -> SearchResult:
        """
        Search the knowledge graph by natural-language query.

        Args:
            graph_id: Logical graph identifier (group_id).
            query: Natural-language search string.
            num_results: Maximum number of edge/node results to return.

        Returns:
            SearchResult with facts, edges, and nodes.
        """
        graphiti = await self._get_graphiti()

        results = await graphiti.search(
            query=query,
            group_ids=[graph_id],
            num_results=num_results,
        )

        facts: List[str] = []
        edges: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []

        for edge in (results.edges or []):
            edge_dict = _graphiti_edge_to_dict(edge)
            edges.append(edge_dict)
            if edge_dict["fact"]:
                facts.append(edge_dict["fact"])

        for node in (results.nodes or []):
            node_obj = _graphiti_entity_to_node(node)
            nodes.append(node_obj.to_dict())

        return SearchResult(
            facts=facts,
            edges=edges,
            nodes=nodes,
            query=query,
            total_count=len(edges) + len(nodes),
        )

    # ------------------------------------------------------------------
    # Entity reads: get_entities
    # ------------------------------------------------------------------

    async def get_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """
        Retrieve entities from the graph, optionally filtered by type labels.

        This is the Graphiti equivalent of
        ZepEntityReader.filter_defined_entities().

        Args:
            graph_id: Logical graph identifier (group_id).
            defined_entity_types: If given, only return entities whose labels
                include at least one of these type names.
            enrich_with_edges: Whether to attach related edge/node information.

        Returns:
            FilteredEntities collection.
        """
        graphiti = await self._get_graphiti()

        # Graphiti's search with an empty query returns a broad result set.
        # We use a broad search then collect unique nodes from the response.
        results = await graphiti.search(
            query="*",
            group_ids=[graph_id],
            num_results=500,
        )

        all_nodes_raw = list(results.nodes or [])
        all_edges_raw = list(results.edges or [])

        total_count = len(all_nodes_raw)

        # Build edge lookup for enrichment
        edges_by_node: Dict[str, List[Dict[str, Any]]] = {}
        if enrich_with_edges:
            for edge in all_edges_raw:
                edge_dict = _graphiti_edge_to_dict(edge)
                src = edge_dict["source_node_uuid"]
                tgt = edge_dict["target_node_uuid"]
                for node_uuid in (src, tgt):
                    if node_uuid not in edges_by_node:
                        edges_by_node[node_uuid] = []
                    edges_by_node[node_uuid].append(edge_dict)

        # Build node uuid map
        node_map: Dict[str, EntityNode] = {}
        for gn in all_nodes_raw:
            en = _graphiti_entity_to_node(gn)
            node_map[en.uuid] = en

        # Filter by entity type labels
        filtered: List[EntityNode] = []
        entity_types_found: Set[str] = set()

        for node in node_map.values():
            custom_labels = [lb for lb in node.labels if lb not in ("Entity", "Node")]
            if not custom_labels:
                continue

            if defined_entity_types:
                matching = [lb for lb in custom_labels if lb in defined_entity_types]
                if not matching:
                    continue
                entity_type = matching[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            if enrich_with_edges:
                related_edge_dicts = edges_by_node.get(node.uuid, [])
                related_edges: List[Dict[str, Any]] = []
                related_node_uuids: Set[str] = set()

                for ed in related_edge_dicts:
                    if ed["source_node_uuid"] == node.uuid:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": ed["name"],
                            "fact": ed["fact"],
                            "target_node_uuid": ed["target_node_uuid"],
                        })
                        related_node_uuids.add(ed["target_node_uuid"])
                    else:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": ed["name"],
                            "fact": ed["fact"],
                            "source_node_uuid": ed["source_node_uuid"],
                        })
                        related_node_uuids.add(ed["source_node_uuid"])

                node.related_edges = related_edges
                node.related_nodes = [
                    {
                        "uuid": node_map[u].uuid,
                        "name": node_map[u].name,
                        "labels": node_map[u].labels,
                        "summary": node_map[u].summary,
                    }
                    for u in related_node_uuids
                    if u in node_map
                ]

            filtered.append(node)

        logger.info(
            "get_entities: group_id=%s total=%d filtered=%d types=%s",
            graph_id,
            total_count,
            len(filtered),
            entity_types_found,
        )

        return FilteredEntities(
            entities=filtered,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered),
        )

    # ------------------------------------------------------------------
    # Convenience: run a coroutine synchronously
    # ------------------------------------------------------------------

    @staticmethod
    def run_sync(coro: Any) -> Any:
        """
        Run an async coroutine synchronously.

        Creates a new event loop if none is running, otherwise schedules it
        on the existing loop (useful inside Flask sync request handlers).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ------------------------------------------------------------------
    # Synchronous wrappers (for callers that cannot use await)
    # ------------------------------------------------------------------

    def add_episode_sync(
        self,
        graph_id: str,
        episode_type: str = "text",
        data: str = "",
        name: Optional[str] = None,
        source_description: str = "mirofish",
        reference_time: Optional[datetime] = None,
    ) -> None:
        """Synchronous wrapper around add_episode."""
        self.run_sync(
            self.add_episode(
                graph_id=graph_id,
                episode_type=episode_type,
                data=data,
                name=name,
                source_description=source_description,
                reference_time=reference_time,
            )
        )

    def search_sync(
        self,
        graph_id: str,
        query: str,
        num_results: int = 10,
    ) -> SearchResult:
        """Synchronous wrapper around search."""
        return self.run_sync(
            self.search(graph_id=graph_id, query=query, num_results=num_results)
        )

    def get_entities_sync(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """Synchronous wrapper around get_entities."""
        return self.run_sync(
            self.get_entities(
                graph_id=graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=enrich_with_edges,
            )
        )
