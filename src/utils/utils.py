import os
import pickle
import re
import asyncio
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Any, Union
import pandas as pd
import pickle
from sqlalchemy import create_engine
import sqlite3
import chromadb
from IPython.core.display_functions import display
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank, LongContextReorder,
)
from llama_index.core import SummaryIndex
from llama_index.core.query_engine import SubQuestionQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from pandas import DataFrame
from rich.markdown import Markdown
# from IPython.core.display import Markdown
from sqlalchemy import create_engine, Engine
from dotenv import load_dotenv, find_dotenv

# Llama Index imports
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings, SQLDatabase, SummaryIndex, )
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
    MarkdownElementNodeParser,
)
from llama_index.core.schema import BaseNode, IndexNode
from llama_index.readers.file import CSVReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.anthropic import Anthropic

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv(find_dotenv())
#=============================================================================
# File Paths
# File Paths
# File Paths
current_path = Path.cwd()
pdf_dirs = current_path.parent / "data" / "pdfs"
# pdf_dirs  = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/RAG/data/pdfs")
docs_path = current_path.parent / "data" / "llamadocs"
# docs_path = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/RAG/data/llamadocs")
chroma_path = current_path.parent / "data" / "chromadb"
# chroma_path = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/RAG/data/chromadb")
sqlitepath = current_path.parent / "data" / "sqlite" / "sqlite.db"
# sqlitepath = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/RAG/data/sqlite/sqlite.db")
# Tesla Data
nodes_path = current_path.parent / "data" / "nodes" / "tesla_esg_nodes.pkl"
# nodes_path = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/RAG/data/nodes/tesla_esg_nodes.pkl")


# Sales Data
sales_data_path = current_path.parent / "data" / "csv" / "kaggle_sample_superstore.csv"
# sales_data_path = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/RAG/data/csv/kaggle_sample_superstore.csv")
sales_nodes_path = current_path.parent / "data" / "nodes" / "sales_nodes.pkl"
# sales_nodes_path = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/RAG/data/nodes/sales_nodes.pkl")


parsing_instruction = """
The pdf documents are annual sustainability reports of companies. 
Always quantify your answer using the numbers cited in the relevant text in the document.
Parse every number embedded in the text in a format that is the easiest to use for calculations or retrieval.
Parse the numbers that indicates sustainiability metrics for that year as a list of sustainability metrics with relevanat text and numbers. 
"""

#=============================================================================
# =============================================================================
# create
# %%
# model_name = "gpt-3.5-turbo"
model_name = "claude-3-haiku-20240307"
# model_name="claude-3-opus-20240229"
embedding_model_name = "text-embedding-3-small"
# llm = OpenAI(temperature=0.1, model=model_name, max_tokens=512)
# llm = Anthropic(model="claude-3-opus-20240229", max_tokens=512, temperature=0.0)
llm = Anthropic(model= model_name, max_tokens=512, temperature=0.0,)
embed_model = OpenAIEmbedding(model=embedding_model_name)
reranker_model = "mixedbread-ai/mxbai-rerank-base-v1"
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.RerankerModel = reranker_model

# base node parser is a sentence splitter
text_splitter = SentenceSplitter()
Settings.text_splitter = text_splitter


# %%
# =============================================================================
# %%
def clean_column_names(df: DataFrame) -> DataFrame:
    # Define a regex pattern to match non-alphanumeric characters, spaces, and multiple spaces between words
    pattern = re.compile(r'(?<=\w)[^\w\s]+|(?<=\s)\s+|\s+')

    # Clean column names by replacing non-alphanumeric characters, spaces, and multiple spaces with underscores
    df.columns = [pattern.sub('_', col) for col in df.columns]
    return df


# =============================================================================
# Display Llama Prompt
# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))


# =============================================================================
# Load data
async def pdf_data_loader(filedir: Path, docs_path: Path,
                          filename: str, parsing_instruction: str = None, num_workers=None) -> List[Document]:
    """
    Loads and parses a PDF file using LlamaParse.

    This function initializes a LlamaParse parser with specified settings, including
    the result type as 'markdown' and an API key obtained from the environment variables.
    It then loads and parses the PDF file located at the given filepath using the parser,
    returning the parsed document(s).

    Parameters:
    - filepath (str): The path to the PDF file to be loaded and parsed.

    Returns:
    - str: The parsed document(s) in the specified result type.
    """
    filepaths = []
    for file_name in os.listdir(filedir):
        _, file_extension = os.path.splitext(file_name)
        file_path = Path(os.path.join(filedir, file_name))
        filepaths.append(str(file_path))
        if file_extension != ".pdf":
            raise ValueError("File must be a PDF file")

    parser = LlamaParse(
        api_key=os.getenv("LLAMA_PARSER_API_KEY"),
        result_type="markdown",
        parsing_instruction=parsing_instruction,
        language="en",
        num_workers=num_workers,
        verbose=True,
    )
    docs = await parser.aload_data(filepaths)
    # file_extractor = {".pdf": parser}
    # docs = SimpleDirectoryReader(
    #     filepath, file_extractor=file_extractor).load_data()

    if docs_path is not None and filename is not None:
        save_docs_file_path = Path(os.path.join(docs_path, filename))
        with open(save_docs_file_path, "wb") as file:
            pickle.dump(docs, file)

    return docs


def csv_excel_data_loader(
        filepath: Path,
        embed_cols: Optional[str] = None,
        embed_metadata: Optional[str] = None,
) -> Tuple[List[Document], Document]:
    """
    Reads .csv and .xlsx data from a file and processes columns for embedding and metadata extraction.

    This function reads data from a specified file (CSV or Excel) and processes columns
    for embedding and metadata extraction. It creates Document objects based on the specified
    columns and metadata, combining them into a single Document object if embedding is enabled.

    Parameters:
    - filepath (Path): The path to the file to read data from.
    - embed_cols (Optional[str]): Columns to process for embedding.
    - embed_metadata (Optional[str]): Columns to extract metadata from.

    Returns:
    - Tuple[List[Document], Document]: A tuple containing a list of Document objects created
      based on the specified columns and metadata, and a single Document object combining
      text from all documents if embedding is enabled.
    """
    docs = []
    document = None
    _, file_extension = os.path.splitext(filepath)

    if file_extension == ".csv":
        df = pd.read_csv(filepath)
    elif file_extension == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        raise ValueError("File must be a CSV or Excel file")
    df = clean_column_names(df)
    if embed_cols:
        for _, row in df.iterrows():
            to_metadata = {col: row[col] for col in embed_metadata if col in row}
            values_to_embed = {k: str(row[k]) for k in embed_cols if k in row}
            to_embed = "\n".join(
                f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items()
            )
            newdoc = Document(text=to_embed, metadata=to_metadata)
            docs.append(newdoc)

        document = Document(text="\n\n".join([doc.text for doc in docs]))

    elif file_extension == ".csv":
        docs = CSVReader(concat_rows=False).load_data(file=filepath)

    return docs, document


def load_data_to_sql_db(filepath: str, dbpath: str, tablename: str) -> DataFrame:
    """
    Loads data from a CSV file into an SQL database table.

    This function reads data from a specified CSV file into a pandas DataFrame. It then
    establishes a connection to an SQLite database located at the given database path.
    Using this connection, the function writes the contents of the DataFrame to a specified
    table within the SQLite database. If the table already exists, its contents are replaced
    with the data from the DataFrame.

    Parameters:
    - filepath (str): The path to the CSV/xlsx file containing the data to be loaded.
    - dbpath (str): The path to the SQLite database file where the data will be stored.
    - tablename (str): The name of the table within the SQLite database where the data
      from the CSV file will be written.

    Returns:
    - None
    """
    _, file_extension = os.path.splitext(filepath)
    if file_extension == ".csv":
        # Read the data from the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
    elif file_extension == ".xlsx":
        # Read the data from the Excel file into a pandas DataFrame
        df = pd.read_excel(filepath)

    else:
        raise ValueError("File must be a CSV or Excel file")
    df = clean_column_names(df)
    #     # Define a regex pattern to match non-alphanumeric characters, spaces, and multiple spaces between words
    # pattern = re.compile(r'(?<=\w)[^\w\s]+|(?<=\s)\s+|\s+')
    #
    # # Clean column names by replacing non-alphanumeric characters, spaces, and multiple spaces with underscores
    # df.columns = [pattern.sub('_', col) for col in df.columns]
    # Create a connection to the SQLite database and an engine
    conn = sqlite3.connect(dbpath)
    engine = create_engine("sqlite:///" + dbpath)

    # Write the DataFrame to a table in the SQLite database
    df.to_sql(tablename, engine, if_exists="replace", index=False)

    # It's good practice to close the connection when done
    conn.close()
    return df


# =============================================================================
# %%
# =============================================================================
# Get Nodes

def concat_node_object(node, node_object):
    if node_object is None:
        result = node
    elif isinstance(node_object, list):
        result = node + node_object
    else:
        raise ValueError("my_variable must be either None or a list")
    return result


def get_nodes(
        docs: List[Document],
        node_save_path: Optional[Union[str, Path]] = None,
        is_markdown: bool = False,
        num_workers: int = 8,
        base: bool = False,
) ->List[BaseNode]:
    """
    Extracts nodes from documents using either a SentenceWindowNodeParser or a base text splitter.

    Parameters:
    - docs (List[Document]): A list of Document objects from which nodes are to be extracted.
    - node_save_path (Optional[Union[str, Path]]): Path to save the nodes.
    - is_markdown (bool): Flag indicating if the nodes are in markdown format.
    - num_workers (int): Number of workers for processing.
    - base (bool): Flag indicating if base nodes should be extracted.

    Returns:
    - Tuple[List[BaseNode], Optional[List[IndexNode]]]: A tuple containing a list of BaseNode objects
      and an optional list of IndexNode objects.
    """
    if node_save_path and os.path.exists(node_save_path):
        with open(node_save_path, "rb") as file:
            return pickle.load(file)

    if base and not is_markdown:
        # Extract base nodes using the base text splitter
        nodes = SentenceSplitter.get_nodes_from_documents(docs)
        nodes_object = None
    else:
        if is_markdown:
            node_parser = MarkdownElementNodeParser(num_workers=num_workers)
            # Extract nodes using the selected node parser
            nodes = node_parser.get_nodes_from_documents(docs)
            nodes, nodes_object = node_parser.get_nodes_and_objects(nodes)
        else:
            # Initialize the SentenceWindowNodeParser with default settings
            node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,  # The size of the sentence window
                window_metadata_key="window",  # Metadata key for window information
                original_text_metadata_key="original_text",  # Metadata key for original text information
            )
            # Extract nodes using the selected node parser
            nodes = node_parser.get_nodes_from_documents(docs)
            nodes_object = None

        # if is_markdown:
        #     nodes, nodes_object = node_parser.get_nodes_and_objects(nodes)
        # else:
        #     nodes_object = None
    nodes = concat_node_object(nodes, nodes_object)
    with open(node_save_path, "wb") as file:
        pickle.dump(nodes, file)

    # return nodes, nodes_object
    return nodes


# =============================================================================
# %%
# Get Index
def get_index(vector_db_path, collection_name, nodes=None):
    db = chromadb.PersistentClient(path=vector_db_path)
    textsplitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
    title_extractor = TitleExtractor(nodes=5, llm=llm)
    # qa_extractor = QuestionsAnsweredExtractor(questions=3)
    summary_extractor = SummaryExtractor(summaries=["prev", "self"], llm=llm)
    keyword_extractor = (KeywordExtractor(keywords=10, llm=llm),)
    entity_extractor = EntityExtractor(
        prediction_threshold=0.5,
        label_entities=False,  # include the entity label in the metadata (can be erroneous)
        device="cpu",  # set to "cuda" if you have a GPU
    )
    # Concatenate the list and the variable
    # if nodes_object is None:
    #     nodes = nodes
    # elif isinstance(nodes_object, list):
    #     nodes = nodes + nodes_object
    # Check if the collection does not exist
    if collection_name not in [col.name for col in db.list_collections()]:
        print("building index", collection_name)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        vec_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=[
                textsplitter,
                summary_extractor,
                title_extractor,
                entity_extractor,
                keyword_extractor,
            ],
            show_progress=True,
        )
    else:
        # This block now correctly handles the case where the
        # collection already exists
        print("loading index", collection_name)
        chroma_collection = db.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vec_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
            show_progress=True,
        )

    return vec_index


# =============================================================================
# get query engine
def get_sentence_window_query_engine(
        index,
        similarity_top_k=6,
        rerank_top_n=2,
        window=True,
):
    rerank = SentenceTransformerRerank(top_n=rerank_top_n,
                                       model=reranker_model, )
    # prompt_compression = LongLLMLinguaPostprocessor(
    #     instruction_str="Given the context, please answer the final question",
    #     target_token=300,
    #     rank_method="longllmlingua",
    #     device_map="mps",
    #     additional_compress_kwargs={
    #         "condition_compare": True,
    #         "condition_in_question": "after",
    #         "context_budget": "+100",
    #         "reorder_context": "sort",  # enable document reorder
    #     }, )
    reorder = LongContextReorder()

    if window:
        # define postprocessors
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        # rerank = SentenceTransformerRerank(top_n=rerank_top_n,
        #                                    model=reranker_model, )

        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k,
                                             node_postprocessors=[postproc,
                                                                  rerank,
                                                                  # prompt_compression,
                                                                  reorder,
                                                                  ],
                                             alpha=0.5,
                                             vector_store_query_mode="hybrid",
                                             )
    else:
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k,
                                             node_postprocessors=[rerank,
                                                                  # prompt_compression,
                                                                  reorder,
                                                                  ],
                                             alpha=0.5,
                                             vector_store_query_mode="hybrid", )
    # simple_tool = QueryEngineTool.from_defaults(
    #     query_engine=query_engine,
    #     description="Useful when the query is relatively straightforward and "
    #                 "can be answered with direct information retrieval, "
    #                 "without the need for complex transformations.", )
    #
    # query_engine_tools = [
    #     QueryEngineTool(
    #         query_engine=query_engine,
    #         metadata=ToolMetadata(
    #             name="Tesla and ESG",
    #             description="Tesla 10K and Sustainability Report from Deloitte, "
    #                         "McKinsey and Tesla",
    #         )
    #
    #     ),
    # ]
    # sub_query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=query_engine_tools
    # )
    #
    # sub_question_tool = QueryEngineTool.from_defaults(
    #     query_engine=sub_query_engine,
    #     description="Useful when asking question about Tesla's 10k and sustainability report "
    #                 "of Tesla, Deloitte, and McKinsey",
    # )
    #
    # query_engine = RouterQueryEngine(
    #     selector=LLMSingleSelector.from_defaults(),
    #     query_engine_tools=[
    #         simple_tool,
    #         sub_question_tool,
    #         # multi_step_tool,
    #     ],
    #     verbose=True,
    # )

    return query_engine


# =============================================================================
# create text to sql function
# =============================================================================

def text_to_query_engine(all_table_names: List[str], engine: Engine,
                         temperature: float = 0.1, not_know_table: bool = True) -> Union[
    NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine]:
    """
    Convert text to a query engine for a SQL database.

    This function initializes the necessary components for querying a SQL database using OpenAI models. It sets up both
    the language model and the embedding model from OpenAI, configures the service context, and initializes the SQL database
    connection. Depending on whether the table to be queried is known ahead of time, it returns either a NLSQLTableQueryEngine
    or a SQLTableRetrieverQueryEngine.

    Args:
        model_name (str): The name of the OpenAI model to use.
        embedding_model_name (str): The name of the OpenAI embedding model to use.
        table_name (str): The name of the table in the SQL database to query.
        engine (str): The engine to use for the SQL database.
        temperature (float, optional): The temperature to use for the OpenAI model. Defaults to 0.1.
        not_know_table (bool, optional): Whether the table to query is not known ahead of time. Defaults to True.

    Returns:
        Tuple[Union[NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine],
        SQLDatabase, ServiceContext]: The query engine for the SQL database.
    """
    # Initialize the OpenAI model with the specified temperature and model name
    llm = OpenAI(temperature=temperature, model=model_name)

    # Initialize the OpenAI embedding model with the specified model name
    embed_model = OpenAIEmbedding(model=embedding_model_name)

    # Create a service context with the initialized models
    # service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    # Set the global service context for further use in the application
    # set_global_service_context(service_context)

    # Initialize the SQL database with the specified engine and include the table to be queried
    sql_database = SQLDatabase(engine, include_tables=all_table_names)

    if not_know_table:
        # If the table to query is not known ahead of time, use SQLTableRetrieverQueryEngine
        # This involves creating a mapping and schema objects for the SQL tables
        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = []
        for table_name in all_table_names:
            table_schema_objs.append(SQLTableSchema(table_name=table_name))

        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )
        # Initialize the query engine with the SQL database and the object index for retrieving tables
        query_engine = SQLTableRetrieverQueryEngine(
            sql_database, obj_index.as_retriever(similarity_top_k=1)
        )
    else:
        # If the table to query is known ahead of time, use NLSQLTableQueryEngine
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, tables=all_table_names)

    # Return the initialized query engine
    # return query_engine, sql_database
    return query_engine
#=============================================================================
# Run the final function
fulldocpath = Path.joinpath(docs_path, "10K_esg")
fulllnodespath = Path.joinpath(nodes_path)


def final_query(fulldocpath: Path, fulllnodespath: Path, tablename: str = "Sales"):
    if fulldocpath.exists():
        print(f"The path {fulldocpath} exists. Loading from disk...")
        with open(fulldocpath, "rb") as f:
            esg_docs = pickle.load(f)
    else:
        print(f"The path {fulldocpath} does not exist. Creating a new one...")
        esg_docs = asyncio.run(pdf_data_loader(pdf_dirs, docs_path, fulldocpath.name, parsing_instruction, 3))

    if fulllnodespath.exists():
        print(f"The path {fulllnodespath} exists. Loading from disk...")
        with open(fulllnodespath, "rb") as f:
            esg_nodes = pickle.load(f)
    else:
        print(f"The path {fulllnodespath} does not exist. Creating a new one...")
        esg_nodes = get_nodes(esg_docs, nodes_path, is_markdown=True,
                              num_workers=6)
    # Create index
    esg_index = get_index(str(chroma_path), fulldocpath.name, esg_nodes)
    summary_esg_index = SummaryIndex(esg_nodes)

    # Create query engine
    esg_query_engine = get_sentence_window_query_engine(esg_index, similarity_top_k=6,
                                                        rerank_top_n=3)

    summary_esg_query_engine = summary_esg_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True, )

    df = load_data_to_sql_db(str(sales_data_path), str(sqlitepath), tablename)
    conn = sqlite3.connect(str(sqlitepath))
    engine = create_engine("sqlite:///" + str(sqlitepath))
    sales_sql_query_engine = text_to_query_engine([tablename], engine)

    llm = Anthropic(model="claude-3-opus-20240229")
    # llm = OpenAI(temperature=0.1, model= "gpt-3.5-turbo")
    esg_tool = QueryEngineTool(
        query_engine=esg_query_engine,
        metadata=ToolMetadata(
            name="Tesla10K_Esg_Report_Query_Engine",
            description="Tool to query McKinsey and Deloitte annual sustainability report. And Teslas 2022 10K financial statement"))
    sql_tool = QueryEngineTool(
        query_engine=sales_sql_query_engine,
        metadata=ToolMetadata(
            name="SQl engine to query the sqlite sales database",
            description="Useful for translating a natural language query into a SQL query over"
                        " a table containing: product information"))
    summary_esg_tool = QueryEngineTool(
        query_engine=summary_esg_query_engine,
        metadata=ToolMetadata(
            name="Summary ESG Engine",
            description="Useful for summarizing the ESG report"))
    query_engine_tools = [
        esg_tool,
        summary_esg_tool,
        sql_tool]

    ## SUB QUERY
    sq_vec_summary_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools)
    return sq_vec_summary_engine
