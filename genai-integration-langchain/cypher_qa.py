import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts.prompt import PromptTemplate

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
# Cypher template
# Cypher template with examples
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, for example "The 39 Steps" becomes "39 Steps, The".

Schema:
{schema}
Examples:
1. Question: Get user ratings?
   Cypher: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE u.name = "User name" RETURN r.rating AS userRating
2. Question: Get average rating for a movie?
   Cypher: MATCH (m:Movie)<-[r:RATED]-(u:User) WHERE m.title = 'Movie Title' RETURN avg(r.rating) AS userRating
3. Question: Get movies for a genre?
   Cypher: MATCH ((m:Movie)-[:IN_GENRE]->(g:Genre) WHERE g.name = 'Genre Name' RETURN m.title AS movieTitle

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_template
)
# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    cypher_llm=cypher_model,
    cypher_prompt=cypher_prompt,
    #Schema - If you wanted to just include data about movies and their directors
    include_types=["Movie", "ACTED_IN", "Person"],
    #Schema - if you wanted to exclude ratings data, you could provide User and RATED as the types to the exclude_types
    exclude_types=["User", "RATED"],
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
