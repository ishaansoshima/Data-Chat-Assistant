# README.md for the Conversational CSV Query Application

# Conversational CSV Query Application

This project is a Streamlit-based web application that allows users to upload CSV files and interact with their data using natural language queries. The application leverages LangChain and Groq APIs to generate Python code for data analysis and provides both the code and results to the user.

## Features
- Upload CSV files and preview the data.
- Ask natural language questions about the data.
- Automatically generates and executes Python code to answer queries.
- Displays results and provides human-readable explanations.
- Supports visualizations using Matplotlib.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sql-ai-agent
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On macOS/Linux
   myenv\Scripts\activate   # On Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Groq API key:
   - Add your Groq API key to a `.streamlit/secrets.toml` file in the following format:
     ```toml
     [secrets]
     GROQ_API_KEY = "your_groq_api_key_here"
     ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run main2.py
   ```

2. Open the application in your browser at `http://localhost:8501`.

3. Upload a CSV file and start asking questions about your data!

## Example Questions
- "What are the basic statistics for this dataset?"
- "How many rows contain missing values?"
- "Show me the distribution of values in [column_name]."
- "Create a bar chart showing the counts of [category_column]."
- "Filter the data to show only rows where [condition]."

## License
This project is licensed under the MIT License. See the LICENSE file for details.