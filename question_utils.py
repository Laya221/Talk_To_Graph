from agent import SpecificAgent

def rephrase_question_to_schema_terms(
    user_question: str,
    schema: str,
    api_key: str,
    model: str,
    provider: str 
) -> str:
    """
    Rephrases a user question using schema-specific terminology.

    Parameters:
        user_question (str): The original user question.
        schema (str): The schema containing the correct terminology.
        api_key (str): API key for the LLM provider.
        model (str): LLM model to use (default: 'gpt-4').
        provider (str): API provider name (default: 'openai').

    Returns:
        str: Rephrased question aligned with schema terms.
    """

    prompt_template = """
TASK: Rephrase the user's question to align with the terminology used in the schema, without changing the intent of the question. Replace user-provided words with corresponding schema terms where appropriate.

Example 1:
Schema term: "zone"
User Question: "Which area has the highest revenue?"
Rephrased: "Which zone has the highest revenue?"

Example 2:
Schema terms: "customer_id", "purchase_amount", "transaction_date"
User Question: "What did the user buy, how much did they spend, and when?"
Rephrased: "What did the customer_id buy, what was the purchase_amount, and what is the transaction_date?"

schema:
{schema}

Only output the rephrased question.
User Question: {user_question}
Rephrased Question:
"""


    
    agent = SpecificAgent.create_agent(
        prompt_template=prompt_template,
        api_key=api_key,
        model=model,
        provider=provider,
        output_processing="ENHANCED_QUERY",
        name="SchemaAwareRephrasingAgent"
    )

    return agent.invoke(schema=schema, user_question=user_question)