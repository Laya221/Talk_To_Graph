# Telco Data Analysis Dashboard

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

## Deployment

To deploy this app to Streamlit Cloud:

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Deploy the app by selecting the repository and the main file (streamlit_app.py)

## Requirements

All required packages are listed in `requirements.txt`. The main dependencies are:
- streamlit
- pandas
- numpy
- plotly