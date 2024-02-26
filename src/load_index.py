import sqlite3
from typing import Tuple, List, Union

import pandas as pd
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, Document, SQLDatabase, \
    SimpleDirectoryReader
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
import chromadb
from llama_index.readers.file.tabular import CSVReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
from pathlib import Path

from sqlalchemy import create_engine
import asyncio


# %%
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-3-small"
llm = OpenAI(temperature=0.1, model=model_name)
embed_model = OpenAIEmbedding(model=embedding_model_name)
Settings.llm = llm
Settings.embed_model = embed_model

# base node parser is a sentence splitter
text_splitter = SentenceSplitter()
Settings.text_splitter = text_splitter

# %%
# ---------------------------



async def get_nodes(filepath):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text", )

    root, ext = os.path.splitext(filepath)
    if ext == '.pdf':
        parser = LlamaParse(result_type='markdown',
                            api_key=os.getenv("LLAMA_PARSER_API_KEY"),
                            verbose=True)
        docs = await parser.aload_data(filepath)

        # extract both the sentence window nodes and the base nodes
        nodes = node_parser.get_nodes_from_documents(docs)
        base_nodes = text_splitter.get_nodes_from_documents(docs)
    return nodes, base_nodes


def get_index(vector_db_path, collection_name, nodes=None):
    db = chromadb.PersistentClient(path=vector_db_path)

    # Check if the collection does not exist
    if collection_name not in [col.name for col in db.list_collections()]:
        print("building index", collection_name)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context,
                                 embed_model=embed_model, show_progress=True)
    else:
        # This block now correctly handles the case where the collection already exists
        chroma_collection = db.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index


def get_sentence_window_query_engine(
        sentence_index,
        similarity_top_k=6,
        rerank_top_n=2, ):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base")

    query_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank])
    return query_engine
#---------------------------
# %%
# ---------------------------
def load_data_to_sql_db(filepath: str, dbpath: str, tablename: str,
                        columns_to_embed: List[str] = None, columns_to_metadata: List[str] = None) -> \
        Tuple[sqlite3.Connection, any, pd.DataFrame, Document]:
    """
    Load data from a CSV file into a SQLite database and prepare documents for embedding.

    This function reads data from a CSV file, drops an unnecessary column, and writes the data to a specified
    SQLite database table. It also prepares a document for embedding by processing specified columns.

    Args:
        filepath (str): The path to the CSV file.
        dbpath (str): The path to the SQLite database.
        tablename (str): The name of the table to create in the database.
        columns_to_embed (List[str]): List of column names whose values will be embedded.
        columns_to_metadata (List[str]): List of column names to be included as metadata.

    Returns:
        Tuple[sqlite3.Connection, any, pd.DataFrame, Document]:
            - conn (sqlite3.Connection): The connection to the SQLite database.
            - engine (any): The SQLAlchemy engine for the SQLite database.
            - data (pd.DataFrame): The data from the CSV file as a DataFrame.
            - document (Document): A single document prepared for embedding, containing text and metadata.
    """
    # Read the data from the CSV file into a pandas DataFrame
    data = pd.read_csv(filepath)

    # Drop the 'Unnamed: 16' column from the DataFrame, if present
    data.drop(columns=["Unnamed: 16"], errors='ignore', inplace=True)

    # Create a connection to the SQLite database and an engine
    conn = sqlite3.connect(dbpath)
    engine = create_engine("sqlite:///" + dbpath)

    # Write the DataFrame to a table in the SQLite database
    data.to_sql(tablename, conn, if_exists='replace', index=False)

    # Initialize an empty list to hold Document objects
    docs = []

    # Iterate over each row in the DataFrame to process columns for embedding and metadata
    if columns_to_metadata:
        for _, row in data.iterrows():
            # Extract metadata from specified columns
            to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
            # Prepare text for embedding from specified columns
            values_to_embed = {k: str(row[k]) for k in columns_to_embed if k in row}
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            # Create a new Document object with text and metadata
            newDoc = Document(text=to_embed, metadata=to_metadata)
            docs.append(newDoc)

        # Combine text from all documents into a single Document object
        document = Document(text="\n\n".join([doc.text for doc in docs]))
    else:
        document = None

    # Return the database connection, engine, DataFrame, and the combined document
    return conn, engine, data, document


# %%
# -----------------------------------------------------------------------------
def text_to_query_engine(model_name: str, embedding_model_name: str, all_table_names: List[str], engine: str,
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


# %%
# -----------------------------------------------------------------------------
from llama_index.core.node_parser import SentenceWindowNodeParser


def build_sentence_window_index_vector_DB(
        document, client=None, chromapath=None, collection_name=None):
    db = chromadb.PersistentClient(path=chromapath)
    chroma_collection = db.get_or_create_collection(collection_name)

    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    # sentence_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embed_model,
    #     node_parser=node_parser,
    # )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(document, storage_context=storage_context,
                                            embed_model=embed_model,
                                            show_progress=True)
    # vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #
    # index = VectorStoreIndex.from_documents([document],
    #                                                  storage_context=storage_context)

    return index
