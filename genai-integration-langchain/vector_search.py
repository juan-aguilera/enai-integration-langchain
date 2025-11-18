import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# Create Vector
plot_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)
# Search for similar movie plots
plot = "Toys come alive"
result = plot_vector.similarity_search(plot, k=3)
print(result)
# Parse the documents
"""
The method returns a list of LangChain Document objects, each containing the plot as the content and the node properties as metadata.
You can parse the results to extract the movie titles and plots.
"""
for doc in result:
    print(f"Title: {doc.metadata['title']}")
    print(f"Plot: {doc.page_content}\n")
