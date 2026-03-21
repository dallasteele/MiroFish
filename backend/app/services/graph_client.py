"""
Graph client factory.

Returns either a GraphitiAdapter (local FalkorDB) or a Zep Cloud client
depending on Config.USE_GRAPHITI.

Usage:
    from app.services.graph_client import get_graph_client

    client = await get_graph_client()
    await client.add_episode(graph_id, "text", "Agent Alice posted a tweet.")
    results = await client.search(graph_id, "what did Alice do?")
"""

from typing import Union

from ..config import Config
from ..utils.logger import get_logger
from .graphiti_adapter import GraphitiAdapter

logger = get_logger("mirofish.graph_client")

# Lazy singleton for each backend type
_graphiti_instance: "GraphitiAdapter | None" = None


async def get_graph_client() -> GraphitiAdapter:
    """
    Return the active graph client.

    When Config.USE_GRAPHITI is True (the default), returns a cached
    GraphitiAdapter backed by FalkorDB + Qwen 2.5:7b.

    When Config.USE_GRAPHITI is False, falls back to the Zep Cloud client
    wrapped in a thin compatibility shim so callers keep the same interface.

    Returns:
        GraphitiAdapter (or Zep shim) ready to accept add_episode / search /
        get_entities calls.
    """
    global _graphiti_instance

    if Config.USE_GRAPHITI:
        if _graphiti_instance is None:
            logger.info(
                "Initialising GraphitiAdapter (FalkorDB at %s, LLM %s)",
                Config.FALKORDB_URI,
                Config.LLM_MODEL_NAME,
            )
            _graphiti_instance = GraphitiAdapter()
        return _graphiti_instance

    # ---------- Zep Cloud fallback (shim) ----------
    return _build_zep_shim()


# ---------------------------------------------------------------------------
# Zep compatibility shim
# ---------------------------------------------------------------------------


def _build_zep_shim() -> "GraphitiAdapter":
    """
    Wrap the Zep Cloud client in a GraphitiAdapter-compatible shim.

    This shim lets callers use the same add_episode / search / get_entities
    interface regardless of backend.  We import Zep lazily so the module
    can load even when zep-cloud is not installed.
    """
    try:
        from zep_cloud.client import Zep  # type: ignore
        from .graphiti_adapter import SearchResult, FilteredEntities, EntityNode
        from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges  # type: ignore

        class _ZepShim:
            """Thin async shim around the synchronous Zep Cloud SDK."""

            def __init__(self) -> None:
                self._client = Zep(api_key=Config.ZEP_API_KEY)

            async def add_episode(
                self,
                graph_id: str,
                episode_type: str = "text",
                data: str = "",
                **kwargs,
            ) -> None:
                self._client.graph.add(
                    graph_id=graph_id,
                    type=episode_type,
                    data=data,
                )

            async def search(
                self,
                graph_id: str,
                query: str,
                num_results: int = 10,
            ) -> SearchResult:
                resp = self._client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=num_results,
                )
                facts = []
                edges = []
                nodes = []
                for edge in getattr(resp, "edges", []) or []:
                    fact = getattr(edge, "fact", "") or ""
                    e_uuid = getattr(edge, "uuid_", None) or getattr(edge, "uuid", "")
                    edges.append({
                        "uuid": e_uuid,
                        "name": edge.name or "",
                        "fact": fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                        "attributes": edge.attributes or {},
                    })
                    if fact:
                        facts.append(fact)
                for node in getattr(resp, "nodes", []) or []:
                    n_uuid = getattr(node, "uuid_", None) or getattr(node, "uuid", "")
                    nodes.append({
                        "uuid": n_uuid,
                        "name": node.name or "",
                        "labels": node.labels or [],
                        "summary": node.summary or "",
                        "attributes": node.attributes or {},
                    })
                return SearchResult(
                    facts=facts,
                    edges=edges,
                    nodes=nodes,
                    query=query,
                    total_count=len(edges) + len(nodes),
                )

            async def get_entities(
                self,
                graph_id: str,
                defined_entity_types=None,
                enrich_with_edges: bool = True,
            ) -> FilteredEntities:
                from .zep_entity_reader import ZepEntityReader  # type: ignore
                reader = ZepEntityReader(api_key=Config.ZEP_API_KEY)
                return reader.filter_defined_entities(
                    graph_id=graph_id,
                    defined_entity_types=defined_entity_types,
                    enrich_with_edges=enrich_with_edges,
                )

        logger.info("Using Zep Cloud backend (USE_GRAPHITI=false)")
        return _ZepShim()  # type: ignore[return-value]

    except ImportError:
        logger.error(
            "USE_GRAPHITI=false but zep-cloud is not installed. "
            "Either set USE_GRAPHITI=true or install zep-cloud."
        )
        raise RuntimeError(
            "Graph backend misconfigured: USE_GRAPHITI=false but zep-cloud not found."
        )
