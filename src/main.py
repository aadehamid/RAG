import sqlite3

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine

load_dotenv(find_dotenv())
import os
import openai
openai.api_key = os.getenv("OPENA_AI_KEY")
from pathlib import Path
import pandas as pd

# from llama_index.core.query_engine import PandasQueryEngine
# from prompt import new_prompt, instruction_str, context
from prompt import wiki_qa_prompt, context

from prompt import (
    context, instruction_str,
    new_prompt)
from note_engine import note_engine
from load_index import (get_sentence_window_query_engine,
                        get_index,
                        text_to_query_engine)

from prompt import wiki_qa_prompt, context
from note_engine import note_engine

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine

# %%
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-3-small"
llm = OpenAI(temperature=0.1, model=model_name)
embed_model = OpenAIEmbedding(model=embedding_model_name)
Settings.llm = llm
Settings.embed_model = embed_model
table_name = "population"

# %%
import os
os.getcwd()
# %%
dbpath = "/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/" \
                  "embeddings/RAG/agent/agent_data/WorldPopulation2023.db"
print(dbpath)
# Connect to sqlite db
conn = sqlite3.connect(dbpath)
engine = create_engine("sqlite:///" + dbpath)

sql_query_engine = text_to_query_engine(model_name,
                                        embedding_model_name,
                                        all_table_names = [table_name],
                                        engine = engine)

# %%

chromapath = ("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/" \
              "embeddings/RAG/agent/agent_data/chroma_db")
collection_name = 'canada'
index = get_index(chromapath, collection_name, nodes = None)

canada_engine = get_sentence_window_query_engine(index)
canada_engine.update_prompts({"wikipedia_canada": wiki_qa_prompt})
canada_engine.update_prompts({"response_synthesizer:text_qa_template": wiki_qa_prompt})
# %%
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=sql_query_engine,
        metadata=ToolMetadata(
            name="sql_query_engine",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about canada the country",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# %%
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)