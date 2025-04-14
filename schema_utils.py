from agent import SpecificAgent
def extract_and_format_schema(user_query: str, full_schema: str, model: str, provider: str, api_key: str) -> str:
    # Step 1: Create agent to extract relevant part of the schema
    extract_template = """
    from this schema return only part related to questions.
    Q:{user_question}

    schema:
    {schema}

    Output must be only small schema with same names and elements.

    schema:
    """

    extract_agent = SpecificAgent.create_agent(
        prompt_template=extract_template,
        api_key=api_key,
        model=model,
        provider=provider,
        output_processing="ENHANCED_QUERY",
        name="RelevantSchemaExtractor"
    )

    # Step 2: Extract the related schema
    relevant_schema = extract_agent.invoke(schema=full_schema, user_question=user_query)

    # Step 3: Create agent to rewrite the schema into structured format
    format_template = """
    rewrite the schema to be like:
    schema={{
        "entities": {{
            "Entity1": {{
                "attributes": ["attribute1", "attribute2", "attribute3"],
                "constraints": ["unique_attribute"]
            }},
            "Entity2": {{
                "attributes": ["attribute4", "attribute5"],
                "constraints": ["unique_attribute"],
                "indexes": ["indexed_attribute"]
            }},
            "Entity3": {{
                "attributes": ["attribute6", "attribute7", "attribute8"]
            }},
            "Entity4": {{
                "attributes": ["attribute9", "attribute10"],
                "indexes": ["indexed_attribute"]
            }}
        }},
        "relationships": [
            {{"type": "RELATION_TYPE1", "from": "Entity1", "to": "Entity2", "properties": ["property1"]}},
            {{"type": "RELATION_TYPE2", "from": "Entity1", "to": "Entity3", "properties": ["property2", "property3"]}},
            {{"type": "RELATION_TYPE3", "from": "Entity3", "to": "Entity4", "properties": ["property4", "property5"]}}
        ]
    }}

    schema:
    {schema}

    Output must schema with same names and elements.

    schema:
    """

    format_agent = SpecificAgent.create_agent(
        prompt_template=format_template,
        api_key=api_key,
        model=model,
        provider=provider,
        output_processing="ENHANCED_QUERY",
        name="SchemaFormatter"
    )

    # Step 4: Format the schema
    formatted_schema = format_agent.invoke(schema=relevant_schema)

    return formatted_schema
