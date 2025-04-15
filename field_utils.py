from neo4j import GraphDatabase

def get_fields_info(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD):
    query = """
    CALL db.propertyKeys() YIELD propertyKey
    WITH propertyKey
    LIMIT 100
    CALL {
        WITH propertyKey
        MATCH (n)
        WHERE n[propertyKey] IS NOT NULL
        RETURN propertyKey AS key, collect(DISTINCT n[propertyKey])[..5] AS examples
    }
    RETURN key, examples
    ORDER BY key
    """

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]