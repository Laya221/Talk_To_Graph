from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

# Configure Cypher generation template
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Instructions:
1. Break down the user question into key components.
2. Identify all possible related nodes, attributes, relationships, and properties from the schema for each component.
3. Ensure that the output is comprehensive and includes all relevant elements.
4. Generate a Cypher query using the identified elements to answer the user question.

Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Use toLower in cypher query when you look for string.

The question is:
{question}"""

# Initialize components
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

def initialize_qa_chain(neo4j_uri, neo4j_username, neo4j_password, openai_api_key):
    # Initialize LLM with GPT-4O
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
    
    # Initialize Neo4j graph
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        enhanced_schema=True
    )
    
    # Initialize QA chain
    qa = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
        verbose=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT
    )
    qa.qa_chain.prompt.template="""
You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Here is an example:


Only If the provided information is empty, say that you don't know the answer.
If there is provided information , be smart to extract answer. Do not say "I don't know".
PLease read the whole provided information to generate answer.
If there is provided information and you can not extract answer, tell user the information provided you have.
use emojis.
Output Must Be Natural Language.
Information:
{context}

Question: {question}
Helpful Answer:
"""
    
    return qa
    
