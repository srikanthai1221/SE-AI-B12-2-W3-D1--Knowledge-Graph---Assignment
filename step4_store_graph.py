"""
STEP 4 - GRAPH STORAGE (Neo4j AuraDB)
Knowledge Graph RAG pipeline:
  Document Loader -> Entity Extraction -> Relationship Mapping -> [Graph Storage] -> Graph Query

Goal of this step: load the entities (nodes) and relationships (edges) from
Steps 2 and 3 into your Neo4j AuraDB cloud database.

Input : step2_entities.json + step3_relationships.json
Output: a populated graph in Neo4j AuraDB

Run:
  pip install langchain-neo4j neo4j
  python3 step4_store_graph.py
"""

import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

ENTITIES_FILE = "step2_entities.json"
RELATIONSHIPS_FILE = "step3_relationships.json"

# Only these labels / relationship types are allowed into the database
# (Cypher can't take a label as a parameter, so we check them ourselves)
ALLOWED_LABELS = {"Department", "Scheme", "Beneficiary", "Benefit", "Crop", "Activity"}
ALLOWED_RELS = {"OFFERED_BY", "TARGETS", "PROVIDES", "COVERS_CROP", "INVOLVES"}


def connect() -> Neo4jGraph:
    for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        if not os.getenv(key):
            raise SystemExit(f"{key} not found. Add your Neo4j Aura details to the .env file.")

    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        refresh_schema=False,
    )
    print("Connected to Neo4j AuraDB")
    return graph


def main():
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        entities = json.load(f)["entities"]
    with open(RELATIONSHIPS_FILE, encoding="utf-8") as f:
        relationships = json.load(f)["relationships"]

    graph = connect()

    # ---- 1. Start fresh so re-running doesn't create duplicates ----
    graph.query("MATCH (n) DETACH DELETE n")
    print("Cleared old data")

    # ---- 2. Constraints: each name is unique per label (also speeds up lookups) ----
    for label in ALLOWED_LABELS:
        graph.query(
            f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        )

    # ---- 3. Create nodes, one batch per label ----
    by_label = defaultdict(list)
    for e in entities:
        if e["type"] in ALLOWED_LABELS:
            by_label[e["type"]].append({"name": e["name"], "description": e.get("description", "")})

    for label, rows in by_label.items():
        graph.query(
            f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{name: row.name}}) "
            f"SET n.description = row.description",
            params={"rows": rows},
        )
        print(f"   Created {len(rows):3} {label} nodes")

    # ---- 4. Create relationships, one batch per (source type, relation, target type) ----
    by_rel = defaultdict(list)
    for r in relationships:
        if r["relation"] in ALLOWED_RELS and r["source_type"] in ALLOWED_LABELS and r["target_type"] in ALLOWED_LABELS:
            by_rel[(r["source_type"], r["relation"], r["target_type"])].append(
                {"source": r["source"], "target": r["target"]}
            )

    for (src, rel, tgt), rows in by_rel.items():
        graph.query(
            f"UNWIND $rows AS row "
            f"MATCH (a:{src} {{name: row.source}}) "
            f"MATCH (b:{tgt} {{name: row.target}}) "
            f"MERGE (a)-[:{rel}]->(b)",
            params={"rows": rows},
        )
        print(f"   Created {len(rows):3} {rel} relationships")

    # ---- 5. Check what is in the database now ----
    nodes = graph.query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label")
    rels = graph.query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type")

    print("\nIn Neo4j now:")
    print("  Nodes:")
    for row in nodes:
        print(f"     {row['label']:12}: {row['count']}")
    print("  Relationships:")
    for row in rels:
        print(f"     {row['type']:12}: {row['count']}")

    graph.refresh_schema()
    print("\nGraph schema (Step 5 will give this to the LLM):")
    print(graph.schema)

    print("\nDone -> ready for Step 5 (Graph Query / chatbot)")


if __name__ == "__main__":
    main()