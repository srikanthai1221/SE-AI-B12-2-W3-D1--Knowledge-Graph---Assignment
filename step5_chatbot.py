"""
STEP 5 - GRAPH QUERY (the RAG chatbot)
Knowledge Graph RAG pipeline:
  Document Loader -> Entity Extraction -> Relationship Mapping -> Graph Storage -> [Graph Query]

How it works for every question:
  1. Your question  ->  OpenAI writes a Cypher query (using the graph schema)
  2. Cypher query   ->  runs on Neo4j AuraDB, returns matching data
  3. Data + question ->  OpenAI writes a friendly answer
This is "Graph RAG": the answer is Retrieved from the graph, then the LLM
Augments it into a Generated reply.

Run:
  python3 step5_chatbot.py
Type 'exit' to quit.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# 1. Connect to Neo4j (same details as Step 4)
# ---------------------------------------------------------------------------
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),
)  # reads the schema automatically: node labels, relationships, properties


# ---------------------------------------------------------------------------
# 2. Prompt that turns a question into Cypher
#    The examples teach the LLM our graph's patterns ("few-shot prompting")
# ---------------------------------------------------------------------------
CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""Task: write a Cypher query for a Neo4j graph of Tamil Nadu Government
Agriculture schemes.

Schema:
{schema}

Rules:
- Use only the node labels, relationship types and properties in the schema.
- Match names case-insensitively and partially, e.g.
  WHERE toLower(c.name) CONTAINS 'paddy'
- If a word may only appear in the scheme name, also search Scheme.name.
- Return names (e.g. s.name), not whole nodes.
- Read-only: use only MATCH / WHERE / RETURN / WITH / ORDER BY / LIMIT.
- Output only the Cypher query. No explanation, no backticks.

Examples:
# Which schemes cover oilseeds?
MATCH (s:Scheme)-[:COVERS_CROP]->(c:Crop)
WHERE toLower(c.name) CONTAINS 'oilseed'
RETURN s.name AS scheme

# Which department runs Distribution of Gypsum?
MATCH (s:Scheme)-[:OFFERED_BY]->(d:Department)
WHERE toLower(s.name) CONTAINS 'gypsum'
RETURN s.name AS scheme, d.name AS department

# What benefits are available and from which schemes?
MATCH (s:Scheme)-[:PROVIDES]->(b:Benefit)
RETURN b.name AS benefit, collect(s.name) AS schemes

# How many schemes are there?
MATCH (s:Scheme) RETURN count(s) AS total_schemes

# Which schemes are about seeds?
MATCH (s:Scheme)
WHERE toLower(s.name) CONTAINS 'seed'
RETURN s.name AS scheme

# Which schemes are for farmers?
MATCH (s:Scheme)
OPTIONAL MATCH (s)-[:TARGETS]->(b:Beneficiary)
WITH s, b
WHERE toLower(b.name) CONTAINS 'farmer' OR toLower(s.name) CONTAINS 'farmer'
RETURN DISTINCT s.name AS scheme

Question: {question}
Cypher query:""",
)


# ---------------------------------------------------------------------------
# 3. Prompt that turns the database results into a friendly answer
# ---------------------------------------------------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant for Tamil Nadu Government Agriculture schemes.
Answer the question using ONLY the database results below.
- If there are several items, list them as bullet points.
- If the results are empty, say you could not find that in the scheme data and
  suggest asking about a crop, benefit or scheme name.
- Do not add facts that are not in the results.

Database results:
{context}

Question: {question}
Answer:""",
)


# ---------------------------------------------------------------------------
# 4. Build the Graph RAG chain
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model=MODEL, temperature=0)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_PROMPT,
    qa_prompt=QA_PROMPT,
    validate_cypher=True,            # fixes relationship directions if the LLM gets them wrong
    top_k=50,                        # max rows passed to the answer step
    return_intermediate_steps=True,  # lets us print the Cypher it wrote
    allow_dangerous_requests=True,   # required by LangChain: the LLM writes DB queries
)


# ---------------------------------------------------------------------------
# 5. Chat loop
# ---------------------------------------------------------------------------
def main():
    print("TN Agriculture Schemes - Knowledge Graph Chatbot")
    print("Try: 'Which schemes cover oilseeds?'  |  'How many schemes are there?'")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "bye"):
            print("Bot: Goodbye!")
            break

        try:
            result = chain.invoke({"query": question})
            cypher = result["intermediate_steps"][0]["query"]
            print(f"\n   [Cypher used]\n   {cypher.strip()}\n")
            print(f"Bot: {result['result']}\n")
        except Exception as e:
            print(f"Bot: Sorry, something went wrong: {e}\n")


if __name__ == "__main__":
    main()