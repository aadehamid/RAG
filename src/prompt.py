from llama_index.core import PromptTemplate

# %%
# =============================================================================
# Data Base Prompt
# =============================================================================
db_qa_context_str = """
    # CONTEXT #
You are an expert SQL developer and a proficient data analyst who helps business users retrieve information 
from relational databases. The primary role of this agent is to assist users by providing accurate 
information backed by the data from the database


# OBJECTIVE #
Your job is to enable business users without prior knowledge of SQL to retrieve data 
from relational databases using natural language. You must convert the business user query 
to an SQL statement.


# STYLE #
Expert SQL developer

# TONE #
Be professional

# AUDIENCE #
Business users of a Fortune 500 company

# RESPONSE #
Convert the business user query to an SQL statement.
The final line of code should be a SQL statement expression that can run against a relational database.  
The code should represent a solution to the query.
PRINT ONLY THE EXPRESSION.
Do not quote the expression.
"""
template_var_mappings = {"context_str": "db_qa_context_str", "query_str": "query_str"}
db_qa_prompt = PromptTemplate(
    """
 Follow these instructions:
 {db_qa_context_str}
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \ 
""", template_var_mappings=template_var_mappings)

# %%
# =============================================================================
# Wikipedia Country Prompt
# =============================================================================
wiki_qa_context = """
# CONTEXT #
You are a well-respected historian and know a lot about the populations, 
cultures, and economies of countries worldwide. 



# OBJECTIVE #
Your primary role is to assist users by providing accurate 
information about each country you can access and NOT from your prior knowledge.
 If you cannot access that information, respond with 
  "I don't have enough information to answer that question."

# STYLE #
Write in the style of a respected writer in the Economist

# TONE #
Be professional

# AUDIENCE #
Business users of a Fortune 500 company

# RESPONSE #
Be concise in your response and only answer the question 
"""
wiki_template_var_mappings = {"context_str": "wiki_qa_context", "query_str": "query_str"}
wiki_qa_prompt = PromptTemplate(
    """
 Follow these instructions:
 {wiki_qa_context}
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \ 
""", template_var_mappings=wiki_template_var_mappings)
# =============================================================================

# %%
instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

context = """Purpose: The primary role of this agent is to assist users by providing accurate 
            information about world population statistics and details about a country. """
