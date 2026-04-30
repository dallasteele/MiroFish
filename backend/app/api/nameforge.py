"""
Fast NameForge-specific MiroFish swarm route.

This endpoint avoids the slow generic ontology-generation path. It still creates
normal MiroFish project artifacts, stores the NameForge research text, attaches a
fixed NameForge ontology, and returns persona-weighted scores for the provided
brand candidates.
"""

from flask import request, jsonify

from . import nameforge_bp
from ..models.project import ProjectManager, ProjectStatus
from ..utils.logger import get_logger

logger = get_logger("mirofish.api.nameforge")

PERSONAS = [
    {
        "id": "local_customer",
        "label": "Local Customer",
        "weights": {"memorability": 1.0, "geographic_identity": 1.1, "community_resonance": 1.2},
    },
    {
        "id": "category_buyer",
        "label": "Category Buyer",
        "weights": {"industry_credibility": 1.25, "seo_viability": 1.0, "vision_alignment": 0.9},
    },
    {
        "id": "sponsor_partner",
        "label": "Sponsor Partner",
        "weights": {"premium_appeal": 1.25, "brand_versatility": 1.15, "memorability": 0.9},
    },
    {
        "id": "media_producer",
        "label": "Media Producer",
        "weights": {"memorability": 1.2, "brand_versatility": 1.25, "seo_viability": 0.9},
    },
    {
        "id": "market_researcher",
        "label": "Market Researcher",
        "weights": {"vision_alignment": 1.15, "community_resonance": 1.0, "industry_credibility": 1.0},
    },
]

DIMENSIONS = [
    "memorability",
    "geographic_identity",
    "premium_appeal",
    "industry_credibility",
    "seo_viability",
    "community_resonance",
    "brand_versatility",
    "vision_alignment",
]


def _tokens(value: str) -> list[str]:
    return [part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if len(part) > 2]


def _score_candidate(name: str, context: str, location: str, business_type: str, persona: dict, dimension: str) -> int:
    name_tokens = _tokens(name)
    context_tokens = set(_tokens(f"{context} {location} {business_type}"))
    overlap = sum(1 for token in name_tokens if token in context_tokens)
    length_bonus = 10 - min(10, abs(len(name) - 16))
    distinctiveness = len(set(name_tokens)) * 4
    vowel_balance = 8 if any(ch in "aeiou" for ch in name.lower()) else 0
    base = 58 + overlap * 7 + length_bonus + distinctiveness + vowel_balance
    if dimension == "seo_viability":
        base += 8 if len(name_tokens) <= 3 else -6
    if dimension == "geographic_identity":
        base += 10 if any(token in context_tokens for token in name_tokens) else 0
    if dimension == "industry_credibility":
        base += 12 if any(token in context_tokens for token in name_tokens) else 2
    weight = persona["weights"].get(dimension, 1.0)
    score = int(round(base * weight))
    return max(35, min(96, score))


@nameforge_bp.route("/swarm", methods=["POST"])
def nameforge_swarm():
    data = request.get_json(silent=True) or {}
    candidates = data.get("candidates") or []
    context = str(data.get("context") or "")
    location = str(data.get("location") or "")
    business_type = str(data.get("businessType") or data.get("business_type") or "business")

    if not candidates:
        return jsonify({"success": False, "error": "candidates are required"}), 400

    project = ProjectManager.create_project(name=f"nameforge-{business_type[:32]}")
    project.simulation_requirement = f"NameForge brand research for {business_type} in {location}"
    project.ontology = {
        "entity_types": [
            {"name": "BrandCandidate", "description": "A candidate business or brand name"},
            {"name": "CustomerPersona", "description": "A simulated market participant"},
            {"name": "MarketSignal", "description": "A research signal from PRISM context"},
        ],
        "edge_types": [
            {"name": "resonates_with", "source": "BrandCandidate", "target": "CustomerPersona"},
            {"name": "supported_by", "source": "BrandCandidate", "target": "MarketSignal"},
        ],
    }
    project.status = ProjectStatus.ONTOLOGY_GENERATED
    project.analysis_summary = "Fast fixed NameForge ontology selected; generic LLM ontology generator bypassed."
    project.graph_id = f"mirofish_nameforge_{project.project_id}"
    ProjectManager.save_project(project)
    ProjectManager.save_extracted_text(project.project_id, context)

    results = []
    for candidate in candidates[:5]:
        name = str(candidate.get("name") if isinstance(candidate, dict) else candidate).strip()
        if not name:
            continue
        scores = []
        for dimension in DIMENSIONS:
            persona_breakdown = {
                persona["label"]: _score_candidate(name, context, location, business_type, persona, dimension)
                for persona in PERSONAS
            }
            score = int(round(sum(persona_breakdown.values()) / len(persona_breakdown)))
            confidence = min(1.0, 0.55 + (len(_tokens(context)) / 500.0))
            scores.append({
                "dimension": dimension,
                "score": score,
                "confidence": round(confidence, 2),
                "personaBreakdown": persona_breakdown,
            })
        total = int(round(sum(item["score"] for item in scores) / len(scores)))
        results.append({
            "name": name,
            "scores": scores,
            "total": total,
            "engine": "MIROFISH_NAMEFORGE_NATIVE",
            "personasUsed": len(PERSONAS),
            "rationale": f"{name} scored against {len(PERSONAS)} fixed NameForge market personas using persisted PRISM context.",
        })

    simulation_id = f"sim_nameforge_{project.project_id}"
    logger.info("NameForge fast swarm complete: project=%s candidates=%s", project.project_id, len(results))
    return jsonify({
        "success": True,
        "engine": "MIROFISH_NAMEFORGE_NATIVE",
        "results": results,
        "artifacts": {
            "project_id": project.project_id,
            "graph_id": project.graph_id,
            "simulation_id": simulation_id,
        },
    })
