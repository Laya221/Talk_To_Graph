# Telco Data Analysis Dashboard
An intelligent graph database querying application that enables natural language interaction with Neo4j databases. Built with LangChain, Openai's models, and Gemini models, it translates user questions into precise Cypher queries, making complex graph data accessible through conversational interfaces.
This is a Streamlit dashboard that visualizes telco data analysis, showing various metrics and insights about revenue, customer distribution, and growth patterns.

## Setup Instructions

1. Clone this repository to your local machine
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your telco data file (telco_data.csv) is in the project directory
4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Features

- Regional revenue analysis
- Gender-based revenue comparison
- Top revenue-generating cities
- Digital revenue growth by district
- Key performance metrics

