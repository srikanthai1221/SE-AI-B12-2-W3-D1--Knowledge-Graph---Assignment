"""
STEP 3 - RELATIONSHIP MAPPING
Knowledge Graph RAG pipeline:
  Document Loader -> Entity Extraction -> [Relationship Mapping] -> Graph Storage -> Graph Query

Goal of this step: connect the entities from Step 2 with labelled arrows.
These arrows become the RELATIONSHIPS (edges) of our knowledge graph, e.g.

  (Scheme) -[:OFFERED_BY]->  (Department)
  (Scheme) -[:TARGETS]->     (Beneficiary)
  (Scheme) -[:PROVIDES]->    (Benefit)
  (Scheme) -[:COVERS_CROP]-> (Crop)
  (Scheme) -[:INVOLVES]->    (Activity)

Input : step1_documents.json + step2_entities.json
Output: step3_relationships.json   (used by Step 4)

Run:
  python3 step3_map_relationships.py
"""

import json
import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

CHUNKS_FILE = "step1_documents.json"
ENTITIES_FILE = "step2_entities.json"
OUTPUT_FILE = "step3_relationships.json"
MODEL = "gpt-4o-mini"

# Which relationship connects which entity types (our graph "schema")
SCHEMA = {
    "OFFERED_BY":  ("Scheme", "Department"),
    "TARGETS":     ("Scheme", "Beneficiary"),
    "PROVIDES":    ("Scheme", "Benefit"),
    "COVERS_CROP": ("Scheme", "Crop"),
    "INVOLVES":    ("Scheme", "Activity"),
}
RelType = Literal["OFFERED_BY", "TARGETS", "PROVIDES", "COVERS_CROP", "INVOLVES"]


# ---------------------------------------------------------------------------
# 1. Describe the output we want
# ---------------------------------------------------------------------------
class Relationship(BaseModel):
    source: str = Field(description="Exact name of the source entity, copied from the entity list")
    relation: RelType
    target: str = Field(description="Exact name of the target entity, copied from the entity list")


class RelationshipList(BaseModel):
    relationships: List[Relationship]


# ---------------------------------------------------------------------------
# 2. Prompt + LLM
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You connect entities for a knowledge graph about Tamil Nadu Government schemes.\n"
     "Allowed relationships (source type -> target type):\n"
     "- OFFERED_BY : Scheme -> Department  (the department that runs the scheme)\n"
     "- TARGETS    : Scheme -> Beneficiary (who the scheme is for)\n"
     "- PROVIDES   : Scheme -> Benefit     (what the scheme gives)\n"
     "- COVERS_CROP: Scheme -> Crop        (crop the scheme is about)\n"
     "- INVOLVES   : Scheme -> Activity    (activity the scheme supports)\n"
     "Rules:\n"
     "- Use ONLY entity names from the list below, copied exactly.\n"
     "- Do NOT create OFFERED_BY relationships; they are added automatically.\n"
     "- Focus on TARGETS, PROVIDES, COVERS_CROP and INVOLVES. Most of this information is "
     "INSIDE the scheme names, e.g. 'Distribution of Rhizobium Packets - Oilseeds' "
     "PROVIDES 'Rhizobium Packets' and COVERS_CROP 'Oilseeds'; 'Training to Farmers' "
     "TARGETS 'Farmers' and PROVIDES 'Training'.\n"
     "- For every Scheme in the list, check whether any Benefit, Crop, Beneficiary or "
     "Activity in the list belongs to it.\n"
     "- Do not invent relationships.\n\n"
     "Entity list:\n{entities}"),
    ("human", "Text:\n{text}"),
])

llm = ChatOpenAI(model=MODEL, temperature=0)
mapper = prompt | llm.with_structured_output(RelationshipList)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Check that your .env file exists in this folder.")

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        entities = json.load(f)["entities"]

    # Lookup: lowercase name -> entity (used to check the LLM's answers)
    lookup = {e["name"].lower(): e for e in entities}
    departments = [e for e in entities if e["type"] == "Department"]

    found = {}  # key = (source, relation, target) -> relationship dict
    rejected = 0

    for chunk in chunks:
        cid = chunk["metadata"]["chunk_id"]

        # Give the LLM only the entities from this chunk, plus all departments
        # (the department heading may sit in a different chunk)
        chunk_entities = [e for e in entities if cid in e["chunk_ids"]]
        for d in departments:
            if d not in chunk_entities:
                chunk_entities.append(d)
        entity_text = "\n".join(f"- [{e['type']}] {e['name']}" for e in chunk_entities)

        result = mapper.invoke({"text": chunk["text"], "entities": entity_text})
        print(f"Chunk {cid}: LLM proposed {len(result.relationships)} relationships")

        # ---- Validate: both ends must exist and have the right types ----
        for r in result.relationships:
            src = lookup.get(r.source.strip().lower())
            tgt = lookup.get(r.target.strip().lower())
            expected_src, expected_tgt = SCHEMA[r.relation]
            if not src or not tgt or src["type"] != expected_src or tgt["type"] != expected_tgt:
                rejected += 1
                continue
            key = (src["name"], r.relation, tgt["name"])
            found[key] = {
                "source": src["name"], "source_type": src["type"],
                "relation": r.relation,
                "target": tgt["name"], "target_type": tgt["type"],
            }

    # ---- OFFERED_BY by code: this page belongs to one department, so every
    # ---- scheme is offered by it (more reliable than asking the LLM)
    if len(departments) == 1:
        dept = departments[0]["name"]
        for e in entities:
            if e["type"] == "Scheme":
                found[(e["name"], "OFFERED_BY", dept)] = {
                    "source": e["name"], "source_type": "Scheme",
                    "relation": "OFFERED_BY",
                    "target": dept, "target_type": "Department",
                }

    relationships = list(found.values())
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"relationships": relationships}, f, ensure_ascii=False, indent=2)

    # ---- Check your output ----
    print(f"\nValid relationships : {len(relationships)}")
    print(f"Rejected (bad names/types): {rejected}")
    counts = {}
    for r in relationships:
        counts[r["relation"]] = counts.get(r["relation"], 0) + 1
    for rel, n in sorted(counts.items()):
        print(f"   {rel:12}: {n}")

    schemes = {e["name"] for e in entities if e["type"] == "Scheme"}
    linked = {r["source"] for r in relationships if r["relation"] == "OFFERED_BY"}
    missing = sorted(schemes - linked)
    print(f"\nSchemes without a department: {len(missing)}")
    for m in missing[:5]:
        print(f"   - {m}")

    print("\nSample relationships (non-department):")
    for r in [r for r in relationships if r["relation"] != "OFFERED_BY"][:10]:
        print(f"   ({r['source']}) -[:{r['relation']}]-> ({r['target']})")

    print(f"\nSaved to {OUTPUT_FILE}  -> ready for Step 4 (Graph Storage in Neo4j)")


if __name__ == "__main__":
    main()