# Conversational CSV Query Application with LangChain and Groq
# This application allows users to upload CSV data, query it in natural language,
# and displays both the code used and the results

import os
import pandas as pd
import streamlit as st
import numpy as np   # For numerical operations on arrays
import matplotlib.pyplot as plt  # For creating visualizations
from io import StringIO  
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re
import textwrap  # Add this import at the top with your other imports

# Set up Groq API
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model_name="llama3-70b-8192", api_key=groq_api_key)

# Configure Streamlit interface
st.title("CSV Data Chat Assistant")
st.write("Upload your CSV file and chat with your data in natural language!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File upload section
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Store the DataFrame in session state
    st.session_state.dataframe = df
    
    # Display sample of the uploaded data
    st.subheader("Preview of your data:")
    st.dataframe(df.head())

    # Extract dataframe schema information
    column_info = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        sample = str(df[column].iloc[0]) if not df[column].empty else "N/A"
        column_info.append(f"- {column} (Type: {dtype}, Example: {sample})")
    
    schema_info = "\n".join(column_info)
    
    # Create the combined prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful data assistant that helps users explore their CSV data. 
        Given the data schema and a user question:
        
        1. Generate a valid pandas code snippet to answer the question. Make sure the code is executable.
        2. The code must produce a result to show to the user - either by assigning to a variable called 'result',
           or by creating a DataFrame that can be displayed.
        3. Then provide a clear, human-readable interpretation that explains the results.
        
        Always format your response exactly as follows:
        
        ```python
        # Your pandas code here that assigns to 'result' or creates a DataFrame
        ```
        
        **Human-Readable Answer:**
        Your clear explanation here, using non-technical language that explains the insights from the data.
        
        Data schema:
        {schema}
        
        Total rows: {row_count}
        """),
        ("human", "{question}")
    ])
    
    def execute_pandas_query(input_text):
        # Extract Python code between backticks
        pattern = r'```python\s+(.*?)\s+```'
        match = re.search(pattern, input_text, re.DOTALL)
        
        if not match:
            return "I couldn't find valid Python code in the response.", input_text
            
        code = match.group(1)
        
        # Add result variable if not present
        if "result" not in code:
            code += "\n# Store the final DataFrame as result if not done already\nif 'result' not in locals():\n    for var in list(locals().keys()):\n        if isinstance(locals()[var], pd.DataFrame) and var != 'df':\n            result = locals()[var]\n            break\n    if 'result' not in locals():\n        result = df"
        
        # Create a full code block with setup - fixed try/except structure
        full_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Access the dataframe
df = st.session_state.dataframe

# Create a copy to avoid modifying the original
df = df.copy()

# Initialize result variable
result = None

# Execute the query
try:
{textwrap.indent(code, '    ')}
except Exception as e:
    result = f"Error: {{str(e)}}"
"""
        
        # Execute the code
        try:
            local_vars = {"st": st, "pd": pd, "np": np, "plt": plt, "StringIO": StringIO}
            exec(full_code, globals(), local_vars)
            
            # Check for result variable
            result = local_vars.get("result", None)
            if result is None:
                # Look for any DataFrame that was created
                for var_name, var_value in local_vars.items():
                    if isinstance(var_value, pd.DataFrame) and var_name != "df":
                        result = var_value
                        break
                # If still no result, use the original df as fallback
                if result is None and "df" in local_vars:
                    result = local_vars["df"]
            
            # If a figure was created, capture it
            fig = plt.gcf()
            if fig.get_axes():
                result = fig
                
            return result, input_text
            
        except Exception as e:
            return f"Error executing code: {str(e)}", input_text
    
    # Build the LangChain
    chain = (
        {"question": RunnablePassthrough(), 
         "schema": lambda _: schema_info,
         "row_count": lambda _: len(df)}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    # Chat input
    if prompt := st.chat_input("Ask about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(prompt)
                results, full_response = execute_pandas_query(response)
                
                # Extract the Python code for display
                code_pattern = r'```python\s+(.*?)\s+```'
                code_match = re.search(code_pattern, full_response, re.DOTALL)
                if code_match:
                    code_block = code_match.group(0)
                    st.subheader("Pandas Code:")
                    st.code(code_match.group(1), language="python")
                
                # Display results
                st.subheader("Query Results:")
                if isinstance(results, str) and results.startswith("Error"):
                    st.error(results)
                elif isinstance(results, pd.DataFrame):
                    st.dataframe(results)
                elif isinstance(results, plt.Figure):
                    st.pyplot(results)
                    plt.close(results)  # Close the figure to free memory
                else:
                    st.write(results)
                
                # Extract and display the human-readable answer
                human_answer_pattern = r'\*\*Human-Readable Answer:\*\*(.*?)(?=$|```)'
                human_answer_match = re.search(human_answer_pattern, full_response, re.DOTALL)
                
                if human_answer_match:
                    st.subheader("Explanation:")
                    st.markdown(human_answer_match.group(1).strip())
                else:
                    # Fallback if no human answer section is found
                    st.markdown(full_response)
                
        # Add assistant response to chat history with a summary
        if human_answer_match:
            display_response = human_answer_match.group(1).strip()
        else:
            display_response = "I processed your question about the data."
            
        st.session_state.messages.append({"role": "assistant", "content": display_response})

else:
    st.info("Please upload a CSV file to begin chatting with your data!")
    
    # Add guidance for users
    st.markdown("""
    ### Example Questions You Can Ask
    Once you upload a CSV file, you can ask questions like:
    - "What are the basic statistics for this dataset?"
    - "How many rows contain missing values?"
    - "Show me the distribution of values in [column_name]"
    - "Create a bar chart showing the counts of [category_column]"
    - "Could you filter the data to show only rows where [condition]?"
    """)