import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

# Initialize the LLM
model = init_chat_model(
    "gpt-4o", 
    model_provider="openai",
)
cypher_model = init_chat_model(
    "gpt-4o", 
    model_provider="openai",
    temperature=0.0
)
"""
The temperature is set to 0. When generating Cypher queries, you want the output to be deterministic and precise.
"""
# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    cypher_llm=cypher_model,
    allow_dangerous_requests=True,
    verbose=True, 
)
"""
You are trusting the generation of Cypher to the LLM. It may generate invalid Cypher queries that could corrupt data
in the graph or provide access to sensitive information.You have to opt-in to this risk by setting the allow_dangerous_requests flag to True.
In a production environment, you should ensure that access to data is limited, and sufficient security is in place to prevent malicious queries.
This could include the use of a read only user or role based access control.

Setting the GraphCypherQAChain verbose parameter to True will print the generated 
Cypher query and the full context used to generate the answer.
"""
# Invoke the chain
question = "How many movies are in the Sci-Fi genre?"
response = cypher_qa.invoke({"query": question})
print(response["result"])
