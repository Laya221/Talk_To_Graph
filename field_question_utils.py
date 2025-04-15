from agent import SpecificAgent
def rephrase_question_to_field_terms(
    user_question: str,
    fields_values: list,
    api_key: str,
    model: str,
    provider: str
) -> str:
    """
    Rephrases a user question using field names or values based on provided field examples.

    Parameters:
        user_question (str): The original user question.
        fields_values (list): A list of dicts with 'key' and 'examples'.
        api_key (str): API key for the LLM provider.
        model (str): LLM model to use.
        provider (str): API provider name.

    Returns:
        str: Rephrased question aligned with field names or example values.
    """

    prompt_template = """
TASK:
Rephrase the user's question by replacing parts of the question with corresponding field names or example values when there's a match in meaning. Use the field keys or values to make the question more aligned with database-friendly or schema-aware language, while preserving the original intent.

Examples:

Field: "gender", Examples: ["F", "M"]
User Question: "How many female users are in the city?"
Rephrased: "How many F users are in the city?"

Field: "arround_district", Examples: ["NGOYO", "TIETIE"]
User Question: "What are the most common areas?"
Rephrased: "What are the most common arround_district values?"

Field: "brand_name", Examples: ["Tecno Telecom"]
User Question: "Which phone brands are popular?"
Rephrased: "Which brand_name values are popular?"

Use the best match from field names or example values based on semantic similarity.

Fields and Example Values:
{fields}

User Question: {user_question}
Rephrased Question:
"""

    formatted_fields = "\n".join(
        [f'- Field: "{field["key"]}", Examples: {field["examples"]}' for field in fields_values]
    )

    agent = SpecificAgent.create_agent(
        prompt_template=prompt_template,
        api_key=api_key,
        model=model,
        provider=provider,
        output_processing="ENHANCED_QUERY",
        name="FieldValueRephrasingAgent"
    )

    return agent.invoke(fields=formatted_fields, user_question=user_question)