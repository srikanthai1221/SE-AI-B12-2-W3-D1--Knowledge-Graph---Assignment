"""
STEP 2 - ENTITY EXTRACTION
Knowledge Graph RAG pipeline:
  Document Loader -> [Entity Extraction] -> Relationship Mapping -> Graph Storage -> Graph Query

Goal of this step: read the chunks from Step 1 and ask OpenAI to pick out the
important "things" (entities) - Departments, Schemes, Beneficiaries, Benefits,
Crops, Activities. These will become the NODES of our knowledge graph.

Input : step1_documents.json   (created by Step 1)
Output: step2_entities.json    (used by Step 3)

Run:
  pip install langchain-openai python-dotenv
  python3 step2_extract_entities.py
"""

import json
import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()  # reads OPENAI_API_KEY from the .env file

INPUT_FILE = "step1_documents.json"
OUTPUT_FILE = "step2_entities.json"
MODEL = "gpt-4o-mini"   # cheap and good enough for extraction

# The kinds of nodes we want in our graph
EntityType = Literal["Department", "Scheme", "Beneficiary", "Benefit", "Crop", "Activity"]


# ---------------------------------------------------------------------------
# 1. Describe the output we want (Pydantic = a "form" the LLM must fill in)
# ---------------------------------------------------------------------------
class Entity(BaseModel):
    name: str = Field(description="Entity name exactly as written in the text")
    type: EntityType = Field(description="One of the allowed entity types")
    description: str = Field(default="", description="One short sentence about it, based only on the text")


class EntityList(BaseModel):
    entities: List[Entity]


# ---------------------------------------------------------------------------
# 2. Prompt + LLM
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract entities for a knowledge graph about Tamil Nadu Government schemes.\n"
     "Allowed types:\n"
     "- Department : a government department (e.g. Agriculture - Farmers Welfare Department)\n"
     "- Scheme     : a named government scheme or programme\n"
     "- Beneficiary: who the scheme is for (e.g. farmers, small farmers, women, SC/ST)\n"
     "- Benefit    : what is given (e.g. subsidy, seeds, training, packets, loan)\n"
     "- Crop       : a crop or crop group (e.g. paddy, oilseeds, pulses)\n"
     "- Activity   : an activity the scheme supports (e.g. distribution, training, irrigation)\n"
     "IMPORTANT: every item listed under a department heading on this page is a Scheme, even if its name sounds like an activity (e.g. 'Distribution of Gypsum' is a Scheme).\n"
          "ALSO extract the Crops, Benefits, Beneficiaries and Activities mentioned INSIDE each scheme name as separate entities (e.g. from 'Distribution of Rhizobium Packets - Oilseeds' extract Benefit 'Rhizobium Packets' and Crop 'Oilseeds'; from 'Training to Farmers' extract Beneficiary 'Farmers' and Benefit 'Training').\n"
     "Rules: use only information in the text. Ignore website menus, accessibility "
     "options, audio player text and other page navigation. Do not invent entities."),
    ("human", "Text:\n{text}"),
])

llm = ChatOpenAI(model=MODEL, temperature=0)
extractor = prompt | llm.with_structured_output(EntityList)


# ---------------------------------------------------------------------------
# 3. Run extraction on every chunk and merge duplicates
# ---------------------------------------------------------------------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Check that your .env file exists in this folder.")

    with open(INPUT_FILE, encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    print(f"Loaded {len(chunks)} chunks from {INPUT_FILE}\n")

    merged = {}  # key = (lowercase name, type) -> entity dict
    for chunk in chunks:
        cid = chunk["metadata"]["chunk_id"]
        result = extractor.invoke({"text": chunk["text"]})
        print(f"Chunk {cid}: found {len(result.entities)} entities")

        for e in result.entities:
            key = (e.name.strip().lower(), e.type)
            if key not in merged:
                merged[key] = {
                    "name": e.name.strip(),
                    "type": e.type,
                    "description": e.description,
                    "chunk_ids": [cid],
                }
            elif cid not in merged[key]["chunk_ids"]:
                merged[key]["chunk_ids"].append(cid)

    entities = list(merged.values())
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"entities": entities}, f, ensure_ascii=False, indent=2)

    # ---- Check your output ----
    print(f"\nTotal unique entities: {len(entities)}")
    counts = {}
    for e in entities:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    for t, n in sorted(counts.items()):
        print(f"   {t:12}: {n}")

    print("\nSample entities:")
    for e in entities[:10]:
        print(f"   [{e['type']}] {e['name']}")

    print(f"\nSaved to {OUTPUT_FILE}  -> ready for Step 3 (Relationship Mapping)")


if __name__ == "__main__":
    main()