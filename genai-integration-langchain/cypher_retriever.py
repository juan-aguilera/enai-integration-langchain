import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

# Initialize the LLM
model = init_chat_model("gpt-4o", model_provider="openai")

# Create a prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    context: List[dict]
    answer: str

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
    allow_dangerous_requests=True,
    return_direct=True,
) 
"""
The return_direct parameter is set to True to return the result of the Cypher query instead of an answer.
This is useful when you want to pass the raw data to the agent for further processing or analysis.
"""
# Define functions for each step in the application

# Retrieve context 
def retrieve(state: State):
    context = cypher_qa.invoke(
        {"query": state["question"]}
    )
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = model.invoke(messages)
    return {"answer": response.content}

# Define application steps
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()

# Run the application
question = "What movies has Tom Hanks acted in?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])

"""
Improve the retriever
Your challenge is to improve the retriever using the techniques you learned in the previous lessons, which could include:

Providing a custom prompt and specific instructions.
Including example questions and Cypher queries.
Using a different LLM model for Cypher generation.
Restricting the schema to provide more focused results.

Here are some examples of more complex questions you can try:

When was the movie The Abyss released?
What is the highest grossing movie of all time?
Can you recommend a Horror movie based on user rating?
What movies scored about 4 for user rating?
What are the highest rated movies with more than 100 ratings?

There is no right or wrong solution. You should experiment with different approaches to see how they affect the accuracy and relevance of the generated Cypher queries.
"""

