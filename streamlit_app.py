import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import get_openai_callback
from qa_chain import initialize_qa_chain
from schema_utils import extract_and_format_schema,fetch_current_schema
from question_utils import rephrase_question_to_schema_terms
from field_question_utils import rephrase_question_to_field_terms
from field_utils import get_fields_info

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Neo4j Chat", page_icon="💬", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load credentials from environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# Initialize QA chain with default LLM (GPT-4O)
if 'qa' not in st.session_state:
    st.session_state.qa = initialize_qa_chain(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY)

# Add model selection to sidebar
with st.sidebar:
    model_option = st.selectbox(
        'Select Language Model',
        ('o3-mini-2025-01-31', 'gpt-4', 'chatgpt-4o-latest', 'gpt-4o', 'gemini-2.0-flash', 'gemini-1.5-pro'),
        index=0
    )
    
    # Add enhancement toggles and model selections
    enable_enhancement = st.checkbox('Enable Schema Enhancement', value=False)
    if enable_enhancement:
        enhancement_model = st.selectbox(
            'Schema Enhancement Model',
            ('o3-mini-2025-01-31', 'gpt-4', 'chatgpt-4o-latest', 'gpt-4o', 'gemini-2.0-flash', 'gemini-1.5-pro'),
            key='enhancement_model',
            index=0
        )
    
    enable_rephrase = st.checkbox('Enable Question Rephrasing', value=True)
    if enable_rephrase:
        rephrase_model = st.selectbox(
            'Question Rephrasing Model',
            ('o3-mini-2025-01-31', 'gpt-4', 'chatgpt-4o-latest', 'gpt-4o', 'gemini-2.0-flash', 'gemini-1.5-pro'),
            key='rephrase_model',
            index=0
        )

# Update LLM based on selected model
if model_option in ['o3-mini-2025-01-31','gpt-4', 'chatgpt-4o-latest', 'gpt-4o']:
    llm = ChatOpenAI(model=model_option, api_key=OPENAI_API_KEY)
else:
    llm = ChatGoogleGenerativeAI(model=model_option, google_api_key=GEMINI_API_KEY)

# Update only the LLM component of the QA chain
st.session_state.qa.cypher_generation_chain.llm = llm
# Add Neo4j connection interface to sidebar
with st.sidebar:
    st.divider()
    st.subheader("Neo4j Connection")
    neo4j_uri = st.text_input("Neo4j URI", value=NEO4J_URI or "")
    neo4j_username = st.text_input("Neo4j Username", value=NEO4J_USERNAME or "")
    neo4j_password = st.text_input("Neo4j Password", value=NEO4J_PASSWORD or "", type="password")
    
    if st.button("Connect to Neo4j"):
        try:
            # Update the schema with new credentials
            st.session_state.full_schema = fetch_current_schema(neo4j_uri, neo4j_username, neo4j_password)
            st.session_state.fields_info = get_fields_info(neo4j_uri, neo4j_username, neo4j_password)
            # Reinitialize QA chain with new credentials
            st.session_state.qa = initialize_qa_chain(neo4j_uri, neo4j_username, neo4j_password, OPENAI_API_KEY)
            st.success("Successfully connected to Neo4j!")
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {str(e)}")

# Initialize Neo4j graph and LLM components
if 'full_schema' not in st.session_state:
    st.session_state.full_schema = fetch_current_schema(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
if 'fields_info' not in st.session_state:
    st.session_state.fields_info = get_fields_info(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)


# Streamlit UI
st.title("💬 Neo4j Chat Assistant")
st.caption("Ask questions about your Neo4j database")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "metrics" in message:
            st.caption(f"Token Usage: {message['metrics']['total_tokens']} | Cost: ${message['metrics']['total_cost']:.6f}")

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            original_prompt = prompt
            try:
                if enable_rephrase:
                    api_key = GEMINI_API_KEY if rephrase_model == 'gemini-2.0-flash' else OPENAI_API_KEY
                    provider = 'gemini' if rephrase_model == 'gemini-2.0-flash' else 'openai'
                    prompt = rephrase_question_to_schema_terms(
                        schema=st.session_state.full_schema, 
                        user_question=prompt, 
                        model=rephrase_model, 
                        provider=provider, 
                        api_key=api_key
                    )
                    prompt =rephrase_question_to_field_terms(
                        fields_values=st.session_state.fields_info, 
                        user_question=prompt, 
                        model=rephrase_model, 
                        provider=provider, 
                        api_key=api_key
                    )
                    prompt=prompt.replace('Rephrased Question:','')
                    prompt=prompt.replace('rephrased question:','')
                    prompt=prompt.replace('Rephrased question:','')
                    st.write("Original question:", original_prompt)
                    st.write("Rephrased question:", prompt)
                    st.divider()
            except Exception as e:
                response = {"result": f"Error in rephrasing question: {str(e)}"}
            try:
                if enable_enhancement:
                    api_key = GEMINI_API_KEY if enhancement_model == 'gemini-2.0-flash' else OPENAI_API_KEY
                    provider = 'gemini' if enhancement_model == 'gemini-2.0-flash' else 'openai'
                    enhance = extract_and_format_schema(
                        prompt, 
                        st.session_state.full_schema, 
                        enhancement_model, 
                        provider, 
                        api_key
                    )
                    with get_openai_callback() as cb:
                        response = st.session_state.qa.invoke({"query": enhance+'\n'+prompt})
                else:
                    with get_openai_callback() as cb:
                        response = st.session_state.qa.invoke({"query": prompt})
            except Exception as e:
                    response = {"result": f"Error processing question: {str(e)}"}
                
                # Display response
            st.write(response["result"])
                
                # Add response to chat history with metrics
            st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["result"],
                    "metrics": {
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_cost": cb.total_cost
                    }
                })
                
                # Display metrics
            st.caption(f"Token Usage: {cb.total_tokens} | Cost: ${cb.total_cost:.6f}")

# Sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This chat interface allows you to query your Neo4j database using natural language.
    The assistant will:
    1. Understand your question
    2. Generate appropriate Cypher queries
    3. Execute them against the database
    4. Present the results in a readable format
    """)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
